from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from itinerary_system.product_app.runtime import product_build_id

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = ROOT / "scripts" / "run_product_app.py"


def load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("w1_product_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def component(name: str, status: str, required: bool, code: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "required_for_core": required,
        "code": code,
        "checked_at": "2026-08-03T00:00:00+00:00",
    }


def health_payload(*, status: str = "ready", core_ready: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "product-health-v2",
        "product_id": "itinerary-repair-copilot",
        "product_version": "0.2.0",
        "build_id": product_build_id(ROOT),
        "status": status,
        "ready": core_ready,
        "core_ready": core_ready,
        "default_run": "demo",
        "legacy_enabled": False,
        "components": {
            "registry": component("registry", "ready" if core_ready else "failed", True, "registry_ready" if core_ready else "registry_unavailable"),
            "default_workspace": component(
                "default_workspace",
                "ready" if core_ready else "failed",
                True,
                "workspace_ready" if core_ready else "registry_required",
            ),
            "state_store": component("state_store", "ready", True, "state_store_ready"),
            "map": component(
                "map",
                "degraded" if status == "degraded" else "ready",
                False,
                "maplibre_unavailable" if status == "degraded" else "maplibre_ready",
            ),
            "openai": component(
                "openai", "disabled", False, "deterministic_adapter_selected"
            ),
        },
    }


def test_launcher_health_validator_accepts_complete_ready_degraded_and_failed_contracts() -> None:
    launcher = load_launcher()

    assert launcher._valid_product_health(health_payload())
    assert launcher._valid_product_health(health_payload(status="degraded"))
    assert launcher._valid_product_health(health_payload(status="failed", core_ready=False))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="wrong"),
        lambda value: value.update(product_id="other-product"),
        lambda value: value.update(product_version="0.1.0"),
        lambda value: value.update(build_id="outdated"),
        lambda value: value.update(ready=False),
        lambda value: value.update(status="failed"),
        lambda value: value["components"].pop("map"),
        lambda value: value["components"]["map"].update(code="raw path C:\\secret"),
        lambda value: value["components"]["map"].update(status="unknown"),
    ],
)
def test_launcher_health_validator_rejects_incomplete_or_inconsistent_contracts(
    mutate: Any,
) -> None:
    launcher = load_launcher()
    payload = health_payload()
    mutate(payload)
    assert launcher._valid_product_health(payload) is False


def test_launcher_health_fetch_rejects_oversized_and_wrong_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = load_launcher()

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit: int) -> bytes:
            return self.body[:limit]

    wrong = health_payload()
    wrong["product_id"] = "wrong"
    monkeypatch.setattr(
        launcher.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(json.dumps(wrong).encode()),
    )
    assert launcher.health("http://127.0.0.1:8127") is None

    oversized = b"x" * (launcher.HEALTH_RESPONSE_LIMIT + 1)
    monkeypatch.setattr(
        launcher.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(oversized),
    )
    assert launcher.health("http://127.0.0.1:8127") is None


@pytest.mark.parametrize(
    ("payload", "expected_exit"),
    [
        (health_payload(), 0),
        (health_payload(status="degraded"), 0),
        (health_payload(status="failed", core_ready=False), 3),
    ],
)
def test_launcher_reuses_only_matching_product_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    expected_exit: int,
) -> None:
    launcher = load_launcher()
    opened: list[str] = []
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            host="127.0.0.1",
            port=8127,
            open=True,
            enable_legacy=False,
            state_root=tmp_path / "state",
        ),
    )
    monkeypatch.setattr(launcher, "health", lambda url, **kwargs: payload)
    monkeypatch.setattr(launcher, "port_in_use", lambda host, port: True)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        launcher,
        "create_product_app",
        lambda **kwargs: pytest.fail("matching service must be reused"),
    )

    assert launcher.main() == expected_exit
    assert opened == ["http://127.0.0.1:8127/app"]


def test_launcher_wrong_or_malformed_health_on_occupied_port_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = load_launcher()
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            host="127.0.0.1",
            port=9127,
            open=False,
            enable_legacy=False,
            state_root=tmp_path / "state",
        ),
    )
    monkeypatch.setattr(launcher, "health", lambda url, **kwargs: None)
    monkeypatch.setattr(launcher, "port_in_use", lambda host, port: True)
    monkeypatch.setattr(launcher, "wait_for_matching_health", lambda url: None)
    assert launcher.main() == 2


def test_launcher_passes_explicit_fixed_host_and_port_to_app_and_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = load_launcher()
    captured: dict[str, Any] = {}
    app = object()
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            host="127.0.0.1",
            port=9127,
            open=False,
            enable_legacy=True,
            state_root=tmp_path / "state",
        ),
    )
    monkeypatch.setattr(launcher, "health", lambda url, **kwargs: None)
    monkeypatch.setattr(launcher, "port_in_use", lambda host, port: False)

    def create(**kwargs: Any) -> object:
        captured["create"] = kwargs
        return app

    def run(value: object, **kwargs: Any) -> None:
        captured["run"] = (value, kwargs)

    monkeypatch.setattr(launcher, "create_product_app", create)
    monkeypatch.setattr(launcher.uvicorn, "run", run)

    assert launcher.main() == 0
    assert captured["create"]["application_host"] == "127.0.0.1"
    assert captured["create"]["application_port"] == 9127
    assert captured["create"]["state_root"] == tmp_path / "state"
    assert captured["run"] == (
        app,
        {"host": "127.0.0.1", "port": 9127, "log_level": "info"},
    )


def test_open_when_ready_opens_only_after_validated_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = load_launcher()
    responses = [None, health_payload(status="degraded")]
    opened: list[str] = []
    monkeypatch.setattr(launcher, "health", lambda url, **kwargs: responses.pop(0))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)

    assert launcher.open_when_ready("http://127.0.0.1:8127") == "core_ready"
    assert opened == ["http://127.0.0.1:8127/app"]


def test_launcher_reuses_a_real_service_after_map_health_cache_expires(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    state_root = tmp_path / "state"
    command = [
        sys.executable,
        str(LAUNCHER_PATH),
        "--port",
        str(port),
        "--state-root",
        str(state_root),
    ]
    environment = os.environ.copy()
    environment["PRODUCT_COPILOT_ADAPTER"] = "deterministic"
    for name in (
        "PRODUCT_APP_ORIGIN",
        "PRODUCT_MAP_PROVIDER",
        "PRODUCT_MAP_BASE_URL",
        "OPENAI_API_KEY",
        "MAPBOX_ATLAS_LICENSE",
    ):
        environment.pop(name, None)

    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15.0
        payload: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=2.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (OSError, ValueError, urllib.error.URLError):
                time.sleep(0.1)
        assert payload is not None
        assert payload["schema_version"] == "product-health-v2"
        assert payload["core_ready"] is True

        time.sleep(2.1)
        reused = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        assert reused.returncode == 0, reused.stderr
        assert "degraded mode" in reused.stdout
    finally:
        server.terminate()
        try:
            server.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5.0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.1)
    else:
        raise AssertionError("launcher test server did not release its fixed port")
