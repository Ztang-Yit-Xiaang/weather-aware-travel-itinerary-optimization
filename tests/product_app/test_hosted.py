"""Real hosted factory with offline providers and observable commit/ASGI ordering.

The shared product conftest supplies offline nearby transport and FastAPI cache
cleanup. Only the build fingerprint is frozen here: concurrent source edits by
other workers must not turn a boundary test into a build-drift test. The real
registry, core factory, security middleware, runtime and file writers still run.
Optional PostgreSQL restart coverage requires an explicit task-owned local DSN.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.routing import Mount

from itinerary_system.product_app import api as api_module
from itinerary_system.product_app import hosted, nearby_places
from itinerary_system.product_app import runtime as runtime_module
from itinerary_system.product_app.config import ProductConfigError
from itinerary_system.product_app.postgres_state import PostgresStateError, PostgresStateStore

ROOT = Path(__file__).resolve().parents[2]
API = "https://itinerary-test.up.railway.app"
FRONTEND = "https://owner-test.github.io"
SUPABASE = "https://itinerary-test.supabase.co"
OWNER = "b947697e-5587-4bf4-a726-6b7f98a4d963"
OTHER_OWNER = "7a6ca9bd-cdb9-4e2f-87fe-e1fb1e08cc36"
TOKEN = "Bearer synthetic-owner-token"
PRIVATE_MARKER = "synthetic-private-upstream-content"
DATABASE_URL = "postgresql://synthetic:synthetic-database-password@database.invalid/state"
ORS_KEY = "synthetic-private-ors-key"
OPENAI_KEY = "synthetic-private-openai-key"
PUBLIC_KEY = "sb_publishable_synthetic_test_key"


def anonymous_key(role="anon"):
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return part({"alg": "HS256"}) + "." + part({"role": role}) + ".synthetic-signature"


@pytest.fixture
def settings():
    return hosted.HostedSettings(
        api_origin=API,
        frontend_origin=FRONTEND,
        supabase_url=SUPABASE,
        owner_id=OWNER,
        database_url=DATABASE_URL,
        supabase_publishable_key=PUBLIC_KEY,
        ors_api_key=ORS_KEY,
        openai_api_key=OPENAI_KEY,
    )


@pytest.fixture(autouse=True)
def offline_network(monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("unexpected provider request")
        raise AssertionError("hosted boundary tests must not contact a network provider")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    yield calls
    assert calls == []


class FakeAuth:
    def __init__(self):
        self.calls = []
        self.failure = False

    def verify(self, value):
        self.calls.append(value)
        if self.failure:
            raise OSError(PRIVATE_MARKER)
        return value == TOKEN


class FakeORS:
    provider_id = "openrouteservice"

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def forbidden(*_args, **_kwargs):
            self.calls.append(name)
            raise AssertionError("health and owner boundary tests must not use ORS")

        return forbidden


class FakeStore:
    """Commit observer/failure injector; real PostgreSQL is tested separately."""

    def __init__(self, state_root):
        self.state_root = state_root
        self.events = []
        self.durable = {}
        self.poisoned = False
        self.fail_commit = False
        self.fail_probe = False

    def restore(self):
        self.events.append("restore")
        self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def probe(self):
        self.events.append("probe")
        if self.fail_probe:
            self.poisoned = True
        if self.poisoned:
            raise PostgresStateError("postgres_state_poisoned_restart_required")

    @contextmanager
    def transaction(self):
        if self.poisoned:
            raise PostgresStateError("postgres_state_poisoned_restart_required")
        self.events.append("begin")
        try:
            yield self
            if self.fail_commit:
                self.events.append("commit_failed")
                raise OSError(PRIVATE_MARKER)
            self.durable = {
                path.relative_to(self.state_root).as_posix(): path.read_bytes()
                for path in self.state_root.rglob("*")
                if path.is_file()
                and path.suffix in {".json", ".jsonl"}
                and not path.name.startswith(".tmp-")
                and "temporary" not in path.relative_to(self.state_root).parts
            }
            self.events.append("commit")
        except BaseException:
            self.poisoned = True
            self.events.append("rollback")
            raise


class SendObserver:
    def __init__(self, app, events):
        self.app, self.events = app, events

    async def __call__(self, scope, receive, send):
        async def observe(message):
            if scope["type"] == "http":
                if message["type"] == "http.response.start":
                    self.events.append(("response_start", message["status"]))
                elif message["type"] == "http.response.body":
                    self.events.append(("response_body", bytes(message.get("body", b""))))
            await send(message)

        await self.app(scope, receive, observe)


@pytest.fixture
def hosted_factory(tmp_path, monkeypatch, settings, offline_network):
    fingerprint = runtime_module.product_build_id(ROOT)
    monkeypatch.setattr(runtime_module, "product_build_id", lambda _root: fingerprint)
    created = []

    @contextmanager
    def start(*, configure=None, chosen_settings=None, store=None):
        cache = store.state_root if store is not None else tmp_path / f"cache_{len(created)}"
        store = store or FakeStore(cache)
        auth, router = FakeAuth(), FakeORS()
        app = hosted.create_hosted_app(
            settings=chosen_settings or settings,
            repository_root=ROOT,
            state_root=cache,
            store=store,
            auth=auth,
            router=router,
        )
        core = next(route.app for route in app.routes if isinstance(route, Mount) and route.path == "")
        bundle = SimpleNamespace(app=app, core=core, store=store, auth=auth, router=router, cache=cache)
        created.append(bundle)
        if configure is not None:
            configure(bundle)
        with TestClient(SendObserver(app, store.events), base_url=API, raise_server_exceptions=False) as client:
            bundle.client = client
            store.events.clear()
            yield bundle
        assert router.calls == []

    return start


def owner_headers(bundle, **extra):
    return {
        "Authorization": TOKEN,
        "Origin": FRONTEND,
        "X-Product-Build-Id": bundle.app.state.product_runtime.build_id,
        **extra,
    }


@pytest.mark.parametrize("authorization", [None, "Bearer synthetic-other-owner", "Basic synthetic"])
def test_private_api_requires_owner_without_entering_state_transaction(hosted_factory, authorization):
    with hosted_factory() as bundle:
        headers = {"Origin": FRONTEND}
        if authorization is not None:
            headers["Authorization"] = authorization
        response = bundle.client.post("/api/sessions", headers=headers, json={})
        assert response.status_code == 401
        assert response.json() == {"detail": "owner_sign_in_required"}
        assert response.headers["access-control-allow-origin"] == FRONTEND
        assert "access-control-allow-credentials" not in response.headers
        assert "begin" not in bundle.store.events
        assert not list((bundle.cache / "sessions").glob("*.json"))


def test_owner_can_read_real_registry_and_create_durable_session(hosted_factory):
    with hosted_factory() as bundle:
        response = bundle.client.get("/api/runs", headers=owner_headers(bundle))
        assert response.status_code == 200
        assert any(row["default"] for row in response.json()["runs"])
        created = bundle.client.post("/api/sessions", headers=owner_headers(bundle), json={})
        assert created.status_code == 200
        session = created.json()["session"]
        assert f"sessions/{session['session_id']}.json" in bundle.store.durable
        restored = bundle.client.get(
            f"/api/sessions/{session['session_id']}",
            headers=owner_headers(bundle, **{"X-Session-Token": created.json()["mutation_token"]}),
        )
        assert restored.status_code == 200
        assert restored.json()["session"]["session_id"] == session["session_id"]


@pytest.mark.parametrize("origin", [None, API, FRONTEND])
def test_allowed_origins_reach_real_api(hosted_factory, origin):
    with hosted_factory() as bundle:
        headers = owner_headers(bundle)
        if origin is None:
            headers.pop("Origin")
        else:
            headers["Origin"] = origin
        response = bundle.client.post("/api/sessions", headers=headers, json={})
        assert response.status_code == 200
        if origin is not None:
            assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_foreign_origin_rejected_before_auth_or_state(hosted_factory, method):
    with hosted_factory() as bundle:
        response = bundle.client.request(
            method, "/api/sessions", headers=owner_headers(bundle, Origin="https://evil.invalid")
        )
        assert response.status_code == 403
        assert response.json() == {"detail": "origin_not_allowed"}
        assert "access-control-allow-origin" not in response.headers
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


@pytest.mark.parametrize("path", ["/api/runs", "/healthz", "/static/js/deployment.js", "/public/map-style.json"])
def test_foreign_host_and_forwarded_host_cannot_reach_hosted_routes(hosted_factory, path):
    with hosted_factory() as bundle:
        response = bundle.client.get(
            path,
            headers={
                **owner_headers(bundle),
                "Host": "evil.invalid",
                "X-Forwarded-Host": API.removeprefix("https://"),
            },
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "host_not_allowed"}
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


def test_railway_health_host_exception_is_limited_to_health(hosted_factory):
    with hosted_factory() as bundle:
        healthy = bundle.client.get("/healthz", headers={"Host": "healthcheck.railway.app"})
        assert healthy.status_code == 200
        private = bundle.client.get("/api/runs", headers={**owner_headers(bundle), "Host": "healthcheck.railway.app"})
        assert private.status_code == 400
        assert not bundle.auth.calls


def test_cors_preflight_allows_all_session_headers_without_auth(hosted_factory):
    with hosted_factory() as bundle:
        response = bundle.client.options(
            "/api/conversations",
            headers={
                "Origin": FRONTEND,
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "Authorization,Content-Type,X-Session-Id,X-Session-Token,X-Product-Build-Id",
            },
        )
        assert response.status_code == 200
        allowed = {item.strip().lower() for item in response.headers["access-control-allow-headers"].split(",")}
        assert {"authorization", "content-type", "x-session-id", "x-session-token", "x-product-build-id"} <= allowed
        assert response.headers["access-control-allow-origin"] == FRONTEND
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


def test_foreign_origin_preflight_has_no_access_grant(hosted_factory):
    with hosted_factory() as bundle:
        response = bundle.client.options(
            "/api/sessions",
            headers={
                "Origin": "https://evil.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


def test_preflight_still_enforces_host_boundary(hosted_factory):
    with hosted_factory() as bundle:
        response = bundle.client.options(
            "/api/sessions",
            headers={
                "Host": "evil.invalid",
                "Origin": FRONTEND,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "host_not_allowed"}
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


def test_health_and_public_map_never_call_auth_routing_or_llm(hosted_factory, offline_network):
    with hosted_factory() as bundle:
        runtime = bundle.app.state.product_runtime
        runtime._map_checked_monotonic = 0
        for _ in range(2):
            response = bundle.client.get("/healthz")
            assert response.status_code == 200
            assert response.json()["status"] == "ready"
        assert bundle.auth.calls == []
        detailed = bundle.client.get("/api/health", headers=owner_headers(bundle))
        assert detailed.status_code == 200
        assert detailed.json()["components"]["routing"]["code"] == "ors_configured_not_live_probed"
        style = bundle.client.get("/public/map-style.json")
        assert style.status_code == 200
        assert style.json()["glyphs"].startswith(API + "/static/")
        assert bundle.router.calls == []
        assert offline_network == []


def test_failed_state_probe_makes_public_health_unavailable(hosted_factory):
    with hosted_factory() as bundle:
        bundle.store.fail_probe = True
        response = bundle.client.get("/healthz")
        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"
        assert bundle.store.poisoned
        assert not bundle.auth.calls


def test_hosted_interest_refresh_uses_labeled_catalog_without_online_fetch(hosted_factory, monkeypatch):
    calls = []

    def forbidden(_query):
        calls.append("nearby provider requested")
        raise AssertionError("hosted interest exploration must remain offline")

    monkeypatch.setattr(nearby_places, "_fetch", forbidden)
    with hosted_factory() as bundle:
        created = bundle.client.post("/api/sessions", json={}, headers=owner_headers(bundle))
        assert created.status_code == 200
        payload = created.json()
        session = payload["session"]
        plan = next(
            row for row in payload["workspace"]["geography"]["plans"] if row["plan_id"] == session["accepted_plan_id"]
        )
        leg = plan["validated_legs"]["features"][0]["properties"]
        response = bundle.client.get(
            f"/api/sessions/{session['session_id']}/map/interest-regions",
            headers=owner_headers(bundle, **{"X-Session-Token": payload["mutation_token"]}),
            params={
                "expected_revision": session["revision"],
                "day": leg["day"],
                "route_leg_id": leg["route_leg_id"],
                "refresh_places": "true",
            },
        )
        assert response.status_code == 200
        assert response.json()["place_data"]["status"] == "bundled_fallback"
        assert response.json()["place_data"]["error_code"] == "nearby_online_disabled_for_hosting"
        assert response.json()["place_data"]["fetched_at"] is None
        assert calls == []
        restored = bundle.client.get(
            f"/api/sessions/{session['session_id']}",
            headers=owner_headers(bundle, **{"X-Session-Token": payload["mutation_token"]}),
        )
        assert restored.status_code == 200
        assert restored.json()["session"]["revision"] == session["revision"]
        assert restored.json()["session"]["accepted_plan_id"] == session["accepted_plan_id"]


def test_commit_occurs_before_first_http_acknowledgement(hosted_factory):
    with hosted_factory() as bundle:
        response = bundle.client.post("/api/sessions", json={}, headers=owner_headers(bundle))
        assert response.status_code == 200
        events = bundle.store.events
        first_ack = next(index for index, event in enumerate(events) if isinstance(event, tuple))
        assert events.index("commit") < first_ack
        assert events[first_ack] == ("response_start", 200)
        session_id = response.json()["session"]["session_id"]
        assert f"sessions/{session_id}.json" in bundle.store.durable


def test_commit_failure_suppresses_success_body_and_poison_blocks_later_requests(hosted_factory):
    with hosted_factory() as bundle:
        durable_before = dict(bundle.store.durable)
        bundle.store.fail_commit = True
        response = bundle.client.post("/api/sessions", json={}, headers=owner_headers(bundle))
        assert response.status_code == 503
        assert response.json() == {"detail": "hosted_state_unavailable"}
        assert PRIVATE_MARKER not in response.text and "mutation_token" not in response.text
        assert ("response_start", 200) not in bundle.store.events
        assert bundle.store.durable == durable_before
        assert bundle.store.poisoned
        assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 503
        assert bundle.client.get("/healthz").status_code == 503


def test_sse_chunks_are_released_only_after_commit(hosted_factory):
    def configure(bundle):
        @bundle.core.get("/api/hosted-test-stream")
        async def stream():
            async def events():
                bundle.store.events.append("first_chunk")
                yield b"data: first\n\n"
                bundle.store.events.append("last_chunk")
                yield b"data: last\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

    with hosted_factory(configure=configure) as bundle:
        response = bundle.client.get("/api/hosted-test-stream", headers=owner_headers(bundle))
        assert response.status_code == 200
        assert response.content == b"data: first\n\ndata: last\n\n"
        events = bundle.store.events
        assert events.index("last_chunk") < events.index("commit") < events.index(("response_start", 200))


def test_response_size_failure_discards_buffer_and_poisons(hosted_factory, monkeypatch):
    monkeypatch.setattr(hosted, "MAX_HOSTED_RESPONSE_BYTES", 8)
    with hosted_factory() as bundle:
        response = bundle.client.get("/api/runs", headers=owner_headers(bundle))
        assert response.status_code == 503
        assert ("response_start", 200) not in bundle.store.events
        assert bundle.store.poisoned


def test_handler_failure_after_local_mutation_cannot_commit_poisoned_runtime(hosted_factory):
    def configure(bundle):
        @bundle.core.post("/api/hosted-test-failure")
        async def fail():
            (bundle.cache / "uncommitted.json").write_text('{"partial":true}', encoding="utf-8")
            raise RuntimeError(PRIVATE_MARKER)

    with hosted_factory(configure=configure) as bundle:
        response = bundle.client.post("/api/hosted-test-failure", headers=owner_headers(bundle))
        assert (response.status_code, bundle.store.poisoned, "uncommitted.json" in bundle.store.durable) == (
            503,
            True,
            False,
        )
        assert PRIVATE_MARKER not in response.text
        assert bundle.store.poisoned
        assert "uncommitted.json" not in bundle.store.durable
        assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 503


def test_client_disconnected_response_rolls_back_and_poisons(hosted_factory):
    def configure(bundle):
        @bundle.core.post("/api/hosted-test-disconnected")
        async def disconnected():
            (bundle.cache / "uncommitted.json").write_bytes(b'{"partial":true}')
            raise HTTPException(status_code=499, detail="preview_client_disconnected")

    with hosted_factory(configure=configure) as bundle:
        response = bundle.client.post("/api/hosted-test-disconnected", headers=owner_headers(bundle))
        assert response.status_code == 503
        assert response.json() == {"detail": "hosted_state_unavailable"}
        assert bundle.store.poisoned
        assert "uncommitted.json" not in bundle.store.durable
        assert ("response_start", 499) not in bundle.store.events


@pytest.mark.parametrize("commit_failure", [False, True], ids=["budget-committed", "database-failed"])
def test_expected_provider_503_preserves_budget_unless_database_commit_fails(hosted_factory, commit_failure):
    budget = b'{"schema_version":"ors-budget-v1","requests":[1800000000]}'

    def configure(bundle):
        @bundle.core.post("/api/hosted-test-provider-unavailable")
        async def unavailable():
            directory = bundle.cache / "routing_ors"
            directory.mkdir(exist_ok=True)
            (directory / "ors-budget.json").write_bytes(budget)
            raise HTTPException(status_code=503, detail="routing_api_unavailable")

    with hosted_factory(configure=configure) as bundle:
        bundle.store.fail_commit = commit_failure
        response = bundle.client.post("/api/hosted-test-provider-unavailable", headers=owner_headers(bundle))
        assert response.status_code == 503
        if commit_failure:
            assert response.json() == {"detail": "hosted_state_unavailable"}
            assert bundle.store.poisoned
            assert "routing_ors/ors-budget.json" not in bundle.store.durable
            assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 503
        else:
            assert response.json() == {"detail": "routing_api_unavailable"}
            assert not bundle.store.poisoned
            assert bundle.store.durable["routing_ors/ors-budget.json"] == budget
            assert bundle.store.events.index("commit") < bundle.store.events.index(("response_start", 503))
            assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 200


def test_timed_out_preview_cannot_leave_live_runtime_while_worker_writes_cache(hosted_factory, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(api_module, "PREVIEW_TIMEOUT_SECONDS", 0.05)
    with hosted_factory() as bundle:
        created = bundle.client.post("/api/sessions", json={}, headers=owner_headers(bundle))
        assert created.status_code == 200
        payload = created.json()
        session = payload["session"]
        service = bundle.app.state.product_runtime.service

        def delayed_preview(*_args, **kwargs):
            assert kwargs["publish"] is False
            started.set()
            try:
                assert release.wait(10)
                # A background compiler/ORS cache write can happen even when
                # proposal publication has explicitly been suppressed.
                (bundle.cache / "late-worker-cache.json").write_bytes(b'{"late":true}')
                return {"unpublished": True}
            finally:
                finished.set()

        monkeypatch.setattr(service, "preview", delayed_preview)
        try:
            response = bundle.client.post(
                f"/api/sessions/{session['session_id']}/preview",
                json={"expected_revision": session["revision"]},
                headers=owner_headers(bundle, **{"X-Session-Token": payload["mutation_token"]}),
            )
            assert started.wait(5)
        finally:
            release.set()
            assert finished.wait(5)
        assert (bundle.cache / "late-worker-cache.json").read_bytes() == b'{"late":true}'
        assert (response.status_code, bundle.store.poisoned) == (503, True)
        assert "late-worker-cache.json" not in bundle.store.durable
        assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 503


def test_auth_outage_is_sanitized_503_without_poisoning_unchanged_store(hosted_factory):
    with hosted_factory() as bundle:
        bundle.auth.failure = True
        response = bundle.client.post("/api/sessions", json={}, headers=owner_headers(bundle))
        assert response.status_code == 503
        assert response.json() == {"detail": "hosted_auth_unavailable"}
        assert PRIVATE_MARKER not in response.text
        assert "begin" not in bundle.store.events
        assert not bundle.store.poisoned
        bundle.auth.failure = False
        assert bundle.client.get("/api/runs", headers=owner_headers(bundle)).status_code == 200


@pytest.mark.parametrize("key", [PUBLIC_KEY, anonymous_key()], ids=["publishable", "legacy-anon"])
def test_public_javascript_exposes_only_browser_configuration_and_no_secrets(hosted_factory, settings, key):
    with hosted_factory(chosen_settings=replace(settings, supabase_publishable_key=key)) as bundle:
        response = bundle.client.get("/static/js/deployment.js")
        assert response.status_code == 200
        prefix = "window.PRODUCT_DEPLOYMENT = "
        assert response.text.startswith(prefix) and response.text.endswith(";\n")
        payload = json.loads(response.text[len(prefix) : -2])
        assert payload == {
            "api_origin": API,
            "supabase_url": SUPABASE,
            "supabase_publishable_key": key,
            "static_frontend": True,
        }
        for secret in (DATABASE_URL, ORS_KEY, OPENAI_KEY, OWNER, PRIVATE_MARKER):
            assert secret not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert "javascript" in response.headers["content-type"]
        assert not bundle.auth.calls
        assert "begin" not in bundle.store.events


def test_settings_repr_hides_keys_database_credentials_and_optional_llm_key(settings):
    for secret in (DATABASE_URL, ORS_KEY, OPENAI_KEY, PUBLIC_KEY):
        assert secret not in repr(settings)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("sb_secret_synthetic_private_key", id="secret-key"),
        pytest.param(anonymous_key("service_role"), id="service-role-jwt"),
        pytest.param(anonymous_key("authenticated"), id="authenticated-jwt"),
        pytest.param("a.not-json.c", id="malformed-jwt"),
        pytest.param("sb_publishable_unsafe\nheader", id="control-character"),
        pytest.param("sb_publishable_" + "x" * 8192, id="oversized-key"),
    ],
)
def test_public_key_validator_rejects_privileged_and_malformed_keys(key):
    with pytest.raises(ProductConfigError, match="supabase_publishable_key_invalid"):
        hosted.public_supabase_key(key)


@pytest.mark.parametrize("key", [PUBLIC_KEY, anonymous_key()], ids=["publishable", "legacy-anon"])
def test_settings_from_environment_validates_public_key_and_defaults_to_unpaid_mode(settings, monkeypatch, key):
    values = {
        "HOSTED_API_ORIGIN": API,
        "HOSTED_FRONTEND_ORIGIN": FRONTEND,
        "SUPABASE_URL": SUPABASE,
        "HOSTED_OWNER_ID": OWNER,
        "DATABASE_URL": DATABASE_URL,
        "SUPABASE_PUBLISHABLE_KEY": key,
        "ORS_API_KEY": ORS_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "PRODUCT_COPILOT_ADAPTER": "deterministic",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HOSTED_ENABLE_PAID_OPENAI", raising=False)
    checked = hosted.HostedSettings.from_environment()
    assert checked.supabase_publishable_key == key
    assert checked.copilot_adapter == "deterministic"
    assert checked.openai_api_key is None
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_secret_synthetic_private_key")
    with pytest.raises(ProductConfigError, match="supabase_publishable_key_invalid"):
        hosted.HostedSettings.from_environment()
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", key)
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "openai")
    with pytest.raises(ProductConfigError, match="paid_openai_not_enabled"):
        hosted.HostedSettings.from_environment()


class AuthResponse:
    def __init__(self, content, status=200):
        self.content, self.status = content, status
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit):
        self.read_limits.append(limit)
        return self.content[:limit]


def auth_opener(monkeypatch, response=None, error=None):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append((request, kwargs))
            if error is not None:
                raise error
            return response

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return calls


@pytest.mark.parametrize(
    "payload,authorized",
    [
        ({"id": OWNER}, True),
        ({"id": OTHER_OWNER}, False),
        ({"id": None}, False),
        ([], False),
    ],
)
def test_supabase_auth_resolves_owner_from_bounded_server_response(settings, monkeypatch, payload, authorized):
    response = AuthResponse(json.dumps(payload).encode())
    calls = auth_opener(monkeypatch, response=response)
    assert hosted.SupabaseOwnerAuth(settings).verify(TOKEN) is authorized
    request, options = calls[0]
    assert request.full_url == SUPABASE + "/auth/v1/user"
    assert request.get_header("Authorization") == TOKEN
    assert request.get_header("Apikey") == PUBLIC_KEY
    assert options == {"timeout": 5}
    assert response.read_limits == [65_537]


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_supabase_auth_distinguishes_rejected_tokens_from_upstream_outage(settings, monkeypatch, code):
    error = urllib.error.HTTPError(SUPABASE + "/auth/v1/user", code, PRIVATE_MARKER, {}, io.BytesIO(b"private"))
    auth_opener(monkeypatch, error=error)
    auth = hosted.SupabaseOwnerAuth(settings)
    if code in {401, 403}:
        assert auth.verify(TOKEN) is False
    else:
        with pytest.raises(RuntimeError, match="hosted_auth_unavailable") as caught:
            auth.verify(TOKEN)
        assert PRIVATE_MARKER not in str(caught.value)


@pytest.mark.parametrize("bad_json", [False, True])
def test_supabase_auth_network_or_corrupt_response_is_sanitized_outage(settings, monkeypatch, bad_json):
    auth_opener(monkeypatch, response=AuthResponse(b"not JSON"), error=None if bad_json else OSError(PRIVATE_MARKER))
    with pytest.raises(RuntimeError, match="hosted_auth_unavailable") as caught:
        hosted.SupabaseOwnerAuth(settings).verify(TOKEN)
    assert PRIVATE_MARKER not in str(caught.value)


@pytest.mark.parametrize("header", ["", "Basic synthetic", "Bearer " + "x" * 8193])
def test_supabase_auth_rejects_bad_header_before_transport(settings, monkeypatch, header):
    calls = auth_opener(monkeypatch)
    assert hosted.SupabaseOwnerAuth(settings).verify(header) is False
    assert calls == []


def test_hosted_session_survives_real_postgres_restart(hosted_factory, tmp_path):
    dsn = os.environ.get("ITINERARY_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("explicit task-owned local PostgreSQL DSN required for hosted restart integration")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict

    options = conninfo_to_dict(dsn)
    if (
        options.get("host") not in {"127.0.0.1", "localhost", "::1"}
        or options.get("hostaddr")
        not in {
            None,
            "127.0.0.1",
            "::1",
        }
        or options.get("service")
        or not options.get("dbname", "").startswith("itinerary_state_test_")
    ):
        pytest.fail("hosted integration requires an explicitly selected disposable local test database")
    namespace = str(uuid4())

    def connect(value):
        return psycopg.connect(value, autocommit=False, prepare_threshold=None)

    # Migration is deliberately a test setup operation, never application boot.
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            (ROOT / "deploy/supabase/001_itinerary_state.sql").read_text(encoding="utf-8"), prepare=False
        )
    try:
        first = PostgresStateStore(tmp_path / "pg_first", dsn, namespace, connect)
        first.events = []  # ASGI observation only; not a production adapter hook.
        with hosted_factory(store=first) as bundle:
            response = bundle.client.post("/api/sessions", headers=owner_headers(bundle), json={})
            assert response.status_code == 200
            payload = response.json()
            session_id = payload["session"]["session_id"]
            token = payload["mutation_token"]
            selection = bundle.client.post(
                f"/api/sessions/{session_id}/selection",
                headers=owner_headers(bundle, **{"X-Session-Token": token}),
                json={"expected_revision": payload["session"]["revision"], "selected_day": 2},
            )
            assert selection.status_code == 200
        restarted = PostgresStateStore(tmp_path / "pg_restarted", dsn, namespace, connect)
        restarted.events = []
        with hosted_factory(store=restarted) as bundle:
            response = bundle.client.get(
                f"/api/sessions/{session_id}",
                headers=owner_headers(bundle, **{"X-Session-Token": token}),
            )
            assert response.status_code == 200
            assert response.json()["session"]["session_id"] == session_id
            assert response.json()["session"]["selected_day"] == 2
    finally:
        with connect(dsn) as connection:
            connection.execute("DELETE FROM itinerary_private.state_namespaces WHERE namespace = %s", (namespace,))
