"""Disposable E2E app bootstrap: deny external Python HTTP/socket dispatch.

This is an application-process guard, not an OS or browser firewall. No request
bodies, credentials, inherited environment values, or arbitrary URLs are logged.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = ["httpx.Client.send", "httpx.AsyncClient.send", "socket.connect"]
_AUDIT_LOCK = threading.RLock()
_AUDIT_GUARDS: list[ProviderIsolation] = []
_AUDIT_INSTALLED = False


def _dispatch_socket_audit(event: str, arguments: tuple) -> None:
    if event != "socket.connect":
        return
    with _AUDIT_LOCK:
        guards = tuple(_AUDIT_GUARDS)
    for guard in guards:
        guard._audit_socket(event, arguments)


def _deactivate_audit_guard(guard: ProviderIsolation) -> None:
    with _AUDIT_LOCK:
        guard.active = False
        if guard in _AUDIT_GUARDS:
            _AUDIT_GUARDS.remove(guard)


class ProviderIsolationError(RuntimeError):
    """Secret-free isolation failure."""


def deterministic_environment(environ: dict[str, str]) -> dict[str, str]:
    environment = dict(environ)
    environment["PRODUCT_COPILOT_ADAPTER"] = "deterministic"
    # Empty, not absent: the normal local-env loader must not restore a key.
    environment["OPENAI_API_KEY"] = ""
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _loopback(host: object) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


class ProviderIsolation:
    def __init__(self, output: Path, run_id: str) -> None:
        self.output = output
        self.active = False
        self._lock = threading.RLock()
        self._patches = ExitStack()
        self.record = {
            "schema_version": "e2e1-provider-isolation-v1",
            "run_id": run_id,
            "process_id": os.getpid(),
            "scope": "application_server",
            "guard_boundaries": BOUNDARIES,
            "installed_at": None,
            "finalized_at": None,
            "finalized": False,
            "effective_adapter": None,
            "actual_adapter": None,
            "api_key_disabled": False,
            "app_initialized": False,
            "server_started": False,
            "server_stopped": False,
            "blocked_attempts": [],
            "sent_provider_calls": 0,
            "allowed_local_connections": 0,
            "failure_code": None,
        }

    def _write(self) -> None:
        with self._lock:
            try:
                temporary = self.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                for attempt in range(8):
                    try:
                        temporary.replace(self.output)
                        break
                    except PermissionError:
                        if attempt == 7:
                            raise
                        # Windows can briefly retain the destination while a
                        # scanner or reader releases its handle. Retry only the
                        # atomic replace; all persistent write failures still
                        # fail closed below.
                        time.sleep(0.025 * (attempt + 1))
            except OSError as error:
                self.record["failure_code"] = "e2e1_telemetry_write_failed"
                self.record["finalized"] = False
                raise ProviderIsolationError("e2e1_telemetry_write_failed") from error

    def _deny(self, boundary: str) -> None:
        with self._lock:
            self.record["blocked_attempts"].append({
                "sequence": len(self.record["blocked_attempts"]) + 1,
                "boundary": boundary,
                "reason": "non_loopback_target",
                "at": _now(),
            })
            self._write()
        raise ProviderIsolationError("e2e1_external_dispatch_blocked")

    def _audit_socket(self, event: str, arguments: tuple) -> None:
        if not self.active or event != "socket.connect":
            return
        connection, address = arguments
        if connection.family not in (socket.AF_INET, socket.AF_INET6):
            return
        if not isinstance(address, tuple) or not _loopback(address[0]):
            self._deny("socket.connect")
        with self._lock:
            self.record["allowed_local_connections"] += 1
            # Persist the count at lifecycle checkpoints, not inside socket.connect.
            # A file-sharing failure must not masquerade as a map network failure.

    def __enter__(self):
        # Install before importing the application or constructing any adapter.
        global _AUDIT_INSTALLED

        self.active = True
        try:
            with _AUDIT_LOCK:
                if not _AUDIT_INSTALLED:
                    sys.addaudithook(_dispatch_socket_audit)
                    _AUDIT_INSTALLED = True
                _AUDIT_GUARDS.append(self)
            import httpx

            sync_send = httpx.Client.send
            async_send = httpx.AsyncClient.send

            def guarded_send(client, request, **kwargs):
                if request.url.scheme != "http" or not _loopback(request.url.host):
                    self._deny("httpx.Client.send")
                return sync_send(client, request, **kwargs)

            async def guarded_async_send(client, request, **kwargs):
                if request.url.scheme != "http" or not _loopback(request.url.host):
                    self._deny("httpx.AsyncClient.send")
                return await async_send(client, request, **kwargs)

            self._patches.enter_context(patch.object(httpx.Client, "send", guarded_send))
            self._patches.enter_context(patch.object(httpx.AsyncClient, "send", guarded_async_send))
            self.record["installed_at"] = _now()
            self._write()
            return self
        except BaseException:
            _deactivate_audit_guard(self)
            self._patches.close()
            self.record["failure_code"] = self.record["failure_code"] or "e2e1_guard_installation_failed"
            self._write()
            raise

    def check_adapter(self, runtime) -> None:
        from itinerary_system.product_app.copilot import DeterministicCopilotAdapter

        adapter = getattr(getattr(runtime, "service", None), "copilot", None)
        if (getattr(getattr(runtime, "config", None), "copilot_adapter", None) != "deterministic"
                or type(adapter) is not DeterministicCopilotAdapter
                or getattr(adapter, "provider_name", None) != "deterministic"
                or os.environ.get("OPENAI_API_KEY") != ""):
            raise ProviderIsolationError("e2e1_actual_adapter_mismatch")
        self.record.update(
            effective_adapter="deterministic", actual_adapter="DeterministicCopilotAdapter",
            api_key_disabled=True, app_initialized=True,
        )
        self._write()

    def __exit__(self, exc_type, exc_value, traceback):
        self.record["finalized_at"] = _now()
        self.record["finalized"] = bool(
            exc_type is None and self.record["app_initialized"]
            and self.record["server_started"] and self.record["server_stopped"]
            and self.record["failure_code"] is None
        )
        if not self.record["finalized"]:
            self.record["failure_code"] = self.record["failure_code"] or "e2e1_guard_lifecycle_incomplete"
        try:
            self._write()
        finally:
            _deactivate_audit_guard(self)
            self._patches.close()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.environ.update(deterministic_environment(os.environ))
    for import_root in (ROOT / "src", ROOT / "scripts"):
        sys.path.insert(0, str(import_root))
    try:
        with ProviderIsolation(args.output, args.run_id) as guard:
            import uvicorn
            from run_product_app import load_local_environment

            from itinerary_system.product_app.api import create_product_app

            load_local_environment(ROOT)
            app = create_product_app(state_root=args.state_root, application_port=args.port)
            guard.check_adapter(app.state.product_runtime)
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info"))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                if sys.stdin.readline().strip() != "stop":
                    raise ProviderIsolationError("e2e1_shutdown_command_missing")
            finally:
                guard.record["server_started"] = server.started
                server.should_exit = True
                thread.join(timeout=15)
                guard.record["server_stopped"] = not thread.is_alive()
            if thread.is_alive() or not server.started:
                raise ProviderIsolationError("e2e1_server_lifecycle_incomplete")
        return 0
    except Exception:
        print("e2e1_guarded_server_failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
