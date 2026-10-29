"""Adversarial checks for the zero-provider E2E process boundary.

No test in this module contacts a provider or starts an application server.
HTTP checks use MockTransport; the native socket target is documentation-only
address space and must be rejected synchronously by the audit hook.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import os
import socket
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "e2e1_provider_isolation", ROOT / "scripts" / "e2e1_provider_isolation.py",
)
assert SPEC and SPEC.loader
isolation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(isolation)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "isolation_test_launcher", ROOT / "scripts" / "run_product_app.py",
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)


@pytest.fixture(autouse=True)
def deterministic_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "")


def _runtime(adapter: object | None = None, *, configured: str = "deterministic") -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(copilot_adapter=configured),
        service=SimpleNamespace(copilot=adapter if adapter is not None else DeterministicCopilotAdapter()),
    )


def _complete(guard: object) -> None:
    guard.check_adapter(_runtime())
    guard.record["server_started"] = True
    guard.record["server_stopped"] = True


def test_local_connection_counter_does_not_write_receipt_in_socket_hook(tmp_path, monkeypatch):
    """Injected Windows sharing error must not become a map connection failure."""
    with isolation.ProviderIsolation(tmp_path / "isolation.json", "local-counter") as guard:
        _complete(guard)
        with monkeypatch.context() as fault:
            fault.setattr(guard, "_write", lambda: (_ for _ in ()).throw(PermissionError(13, "injected sharing violation")))
            guard._audit_socket("socket.connect", (SimpleNamespace(family=socket.AF_INET), ("127.0.0.1", 8080)))
        assert guard.record["allowed_local_connections"] == 1
    record = json.loads((tmp_path / "isolation.json").read_text(encoding="utf-8"))
    assert record["allowed_local_connections"] == 1
    assert record["finalized"] is True


def test_failed_receipt_write_is_latched_even_if_later_flush_recovers(tmp_path, monkeypatch):
    output = tmp_path / "isolation.json"
    with isolation.ProviderIsolation(output, "failed-receipt") as guard:
        _complete(guard)
        with monkeypatch.context() as fault:
            fault.setattr(Path, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "injected sharing violation")))
            with pytest.raises(isolation.ProviderIsolationError, match="e2e1_telemetry_write_failed"):
                guard._write()
        assert guard.record["failure_code"] == "e2e1_telemetry_write_failed"
        dispatched = []
        with httpx.Client(transport=httpx.MockTransport(lambda request: dispatched.append(request))) as client:
            with pytest.raises(isolation.ProviderIsolationError):
                client.get("https://api.openai.com/v1/responses")
        assert dispatched == []
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["finalized"] is False
    assert record["failure_code"] == "e2e1_telemetry_write_failed"
    assert len(record["blocked_attempts"]) == 1


def test_transient_windows_replace_denial_retries_without_weakening_receipt(tmp_path, monkeypatch):
    output = tmp_path / "isolation.json"
    original = Path.replace
    attempts = []

    def briefly_locked(path, target):
        attempts.append(str(target))
        if len(attempts) < 3:
            raise PermissionError(13, "injected transient sharing violation")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    monkeypatch.setattr(isolation.time, "sleep", lambda _: None)
    with isolation.ProviderIsolation(output, "transient-replace") as guard:
        _complete(guard)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert len(attempts) >= 3
    assert record["finalized"] is True
    assert record["failure_code"] is None


def test_environment_is_copied_and_poisoned_configuration_is_neutralized() -> None:
    inherited = {
        "PRODUCT_COPILOT_ADAPTER": "openai",
        "OPENAI_API_KEY": "synthetic-inherited-key-not-a-credential",
        "PRODUCT_MAP_BASE_URL": "http://127.0.0.1:8080",
        "UNRELATED_SETTING": "preserve-me",
    }
    original = inherited.copy()

    protected = isolation.deterministic_environment(inherited)

    assert protected is not inherited
    assert inherited == original
    assert protected == {
        **original, "PRODUCT_COPILOT_ADAPTER": "deterministic", "OPENAI_API_KEY": "",
        "PYTHONIOENCODING": "utf-8",
    }


def test_local_env_loader_cannot_reintroduce_provider_or_key(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text(
        "PRODUCT_COPILOT_ADAPTER=openai\n"
        "OPENAI_API_KEY=synthetic-file-key-not-a-credential\n"
        "PRODUCT_MAP_BASE_URL=http://127.0.0.1:8080\n",
        encoding="utf-8",
    )
    environment = isolation.deterministic_environment({})

    loaded = launcher.load_local_environment(tmp_path, environ=environment)

    assert loaded == 1
    assert environment["PRODUCT_COPILOT_ADAPTER"] == "deterministic"
    assert environment["OPENAI_API_KEY"] == ""
    assert environment["PRODUCT_MAP_BASE_URL"] == "http://127.0.0.1:8080"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_external_http_is_blocked_before_transport_dispatch(tmp_path: Path, asynchronous: bool) -> None:
    dispatched: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        dispatched.append(str(request.url))
        return httpx.Response(200, json={"must_not_run": True})

    async def async_request() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            await client.post("https://api.openai.com/v1/responses", json={"input": "synthetic"})

    with isolation.ProviderIsolation(tmp_path / "isolation.json", "external-boundary") as guard:
        _complete(guard)
        with pytest.raises(RuntimeError):
            if asynchronous:
                asyncio.run(async_request())
            else:
                with httpx.Client(transport=httpx.MockTransport(transport)) as client:
                    client.post("https://api.openai.com/v1/responses", json={"input": "synthetic"})
        assert len(guard.record["blocked_attempts"]) == 1
        assert guard.record["blocked_attempts"][0]["boundary"] == (
            "httpx.AsyncClient.send" if asynchronous else "httpx.Client.send"
        )
        assert guard.record["sent_provider_calls"] == 0
        assert dispatched == []

    # A complete observation can contain a blocked attempt; the enclosing
    # runner must reject its nonzero count rather than falsifying the receipt.
    assert guard.record["finalized"] is True
    assert len(guard.record["blocked_attempts"]) == 1
    assert guard.record["sent_provider_calls"] == 0


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_plain_http_loopback_remains_available(tmp_path: Path, host: str, asynchronous: bool) -> None:
    dispatched: list[str] = []
    url = f"http://{host}:8127/api/health"

    def transport(request: httpx.Request) -> httpx.Response:
        dispatched.append(str(request.url))
        return httpx.Response(200, json={"ready": True})

    async def async_request() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            return await client.get(url)

    with isolation.ProviderIsolation(tmp_path / "isolation.json", "local-boundary") as guard:
        _complete(guard)
        if asynchronous:
            response = asyncio.run(async_request())
        else:
            with httpx.Client(transport=httpx.MockTransport(transport)) as client:
                response = client.get(url)
        assert response.status_code == 200
        assert response.json() == {"ready": True}
        assert dispatched == [url]
        assert guard.record["blocked_attempts"] == []
        assert guard.record["sent_provider_calls"] == 0

    assert guard.record["finalized"] is True


@pytest.mark.parametrize("url", [
    "http://example.invalid/v1/responses",
    "http://localhost.example.invalid/v1/responses",
    "http://127.0.0.1.example.invalid/v1/responses",
    "https://localhost/v1/responses",
])
def test_local_lookalikes_and_non_http_schemes_are_not_allowed(tmp_path: Path, url: str) -> None:
    dispatched = []

    def transport(request: httpx.Request) -> httpx.Response:
        dispatched.append(request)
        return httpx.Response(200)

    with isolation.ProviderIsolation(tmp_path / "isolation.json", "lookalike-boundary") as guard:
        _complete(guard)
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            with pytest.raises(RuntimeError):
                client.get(url)
        assert dispatched == []
        assert len(guard.record["blocked_attempts"]) == 1
        assert guard.record["sent_provider_calls"] == 0


def test_socket_connection_is_blocked_before_network_io(tmp_path: Path) -> None:
    with isolation.ProviderIsolation(tmp_path / "isolation.json", "socket-boundary") as guard:
        _complete(guard)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.05)
            with pytest.raises(RuntimeError):
                connection.connect(("203.0.113.1", 443))
        assert len(guard.record["blocked_attempts"]) == 1
        assert guard.record["blocked_attempts"][0]["boundary"] == "socket.connect"
        assert guard.record["sent_provider_calls"] == 0
        assert guard.record["allowed_local_connections"] == 0


def test_socket_hook_records_local_connections_without_opening_one(tmp_path: Path) -> None:
    with isolation.ProviderIsolation(tmp_path / "isolation.json", "local-socket-boundary") as guard:
        _complete(guard)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            # Drive the same audit event without needing a listening service.
            sys.audit("socket.connect", connection, ("127.0.0.1", 8127))
        assert guard.record["allowed_local_connections"] == 1
        assert guard.record["blocked_attempts"] == []


@pytest.mark.parametrize("malformation", ["config", "object", "subclass", "name", "missing_service", "key"])
def test_runtime_adapter_check_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str,
) -> None:
    runtime = _runtime()
    if malformation == "config":
        runtime.config.copilot_adapter = "openai"
    elif malformation == "object":
        runtime.service.copilot = SimpleNamespace(provider_name="deterministic")
    elif malformation == "subclass":
        class PretendDeterministic(DeterministicCopilotAdapter):
            pass
        runtime.service.copilot = PretendDeterministic()
    elif malformation == "name":
        runtime.service.copilot.provider_name = "openai"
    elif malformation == "missing_service":
        del runtime.service
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-reintroduced-key-not-a-credential")

    with pytest.raises(RuntimeError):
        with isolation.ProviderIsolation(tmp_path / "isolation.json", "adapter-rejected") as guard:
            guard.check_adapter(runtime)

    assert guard.record["finalized"] is False
    assert guard.record["failure_code"]
    assert guard.record["sent_provider_calls"] == 0


@pytest.mark.parametrize("missing", ["adapter_check", "server_started", "server_stopped"])
def test_incomplete_lifecycle_never_claims_finalized(tmp_path: Path, missing: str) -> None:
    with isolation.ProviderIsolation(tmp_path / "isolation.json", "incomplete-lifecycle") as guard:
        if missing != "adapter_check":
            guard.check_adapter(_runtime())
        guard.record["server_started"] = missing != "server_started"
        guard.record["server_stopped"] = missing != "server_stopped"

    assert guard.record["finalized"] is False
    assert guard.record["failure_code"]
    assert guard.record["sent_provider_calls"] == 0


def test_exception_preserves_failure_and_does_not_finalize(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic application failure"):
        with isolation.ProviderIsolation(tmp_path / "isolation.json", "failed-lifecycle") as guard:
            _complete(guard)
            raise ValueError("synthetic application failure")

    assert guard.record["finalized"] is False
    assert guard.record["failure_code"]
    assert guard.record["sent_provider_calls"] == 0


def test_successful_receipt_has_process_identity_and_observed_lifecycle(tmp_path: Path) -> None:
    with isolation.ProviderIsolation(tmp_path / "isolation.json", "successful-lifecycle") as guard:
        _complete(guard)

    assert guard.record["run_id"] == "successful-lifecycle"
    assert guard.record["process_id"] == os.getpid()
    assert guard.record["installed_at"]
    assert guard.record["finalized_at"]
    assert guard.record["finalized"] is True
    assert guard.record["effective_adapter"] == "deterministic"
    assert guard.record["actual_adapter"] == "DeterministicCopilotAdapter"
    assert guard.record["api_key_disabled"] is True
    assert guard.record["app_initialized"] is True
    assert guard.record["server_started"] is True
    assert guard.record["server_stopped"] is True
    assert guard.record["blocked_attempts"] == []
    assert guard.record["sent_provider_calls"] == 0
    receipts = list(tmp_path.rglob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8")) == guard.record


@pytest.mark.parametrize("failed", [False, True], ids=["normal-exit", "exception-exit"])
def test_http_patches_are_restored_and_audit_hook_is_inert_after_exit(tmp_path: Path, failed: bool) -> None:
    original_sync = httpx.Client.send
    original_async = httpx.AsyncClient.send
    dispatched = []

    def transport(request: httpx.Request) -> httpx.Response:
        dispatched.append(str(request.url))
        return httpx.Response(200)

    try:
        with isolation.ProviderIsolation(tmp_path / "isolation.json", "cleanup-boundary") as guard:
            _complete(guard)
            assert httpx.Client.send is not original_sync
            assert httpx.AsyncClient.send is not original_async
            if failed:
                raise ValueError("synthetic exit")
    except ValueError:
        assert failed is True

    assert httpx.Client.send is original_sync
    assert httpx.AsyncClient.send is original_async
    before = dict(guard.record)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        assert client.get("https://example.invalid/fake-only").status_code == 200
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        # Audit hooks cannot be removed; their active context must be cleared.
        sys.audit("socket.connect", connection, ("203.0.113.1", 443))
    assert dispatched == ["https://example.invalid/fake-only"]
    assert guard.record == before


def test_repeated_guard_contexts_do_not_install_process_lifetime_hooks_per_context(tmp_path, monkeypatch):
    installed = []
    original = sys.addaudithook

    def observed(hook):
        original(hook)
        installed.append(hook)
    monkeypatch.setattr(sys, "addaudithook", observed)
    for index in range(10):
        with isolation.ProviderIsolation(tmp_path / f"guard_{index}.json", f"guard-{index}") as guard:
            _complete(guard)
    assert len(installed) <= 1


@pytest.mark.parametrize("failed", [False, True])
def test_exited_guard_is_not_retained_by_the_process_audit_hook(tmp_path, failed):
    def exercise():
        guard = isolation.ProviderIsolation(tmp_path / "guard.json", "released")
        try:
            with guard:
                _complete(guard)
                if failed:
                    raise ValueError("synthetic lifecycle failure")
        except ValueError:
            pass
        return weakref.ref(guard)
    reference = exercise()
    gc.collect()
    assert reference() is None


def test_closing_inner_guard_keeps_outer_native_socket_boundary_active(tmp_path):
    with isolation.ProviderIsolation(tmp_path / "outer.json", "outer") as outer:
        _complete(outer)
        with isolation.ProviderIsolation(tmp_path / "inner.json", "inner") as inner:
            _complete(inner)
        inner_before = json.loads(json.dumps(inner.record))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            with pytest.raises(isolation.ProviderIsolationError):
                sys.audit("socket.connect", connection, ("203.0.113.1", 443))
        assert len(outer.record["blocked_attempts"]) == 1
        assert inner.record == inner_before
