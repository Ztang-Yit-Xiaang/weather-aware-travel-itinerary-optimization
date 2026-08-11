"""Launch the loopback-only Itinerary Repair Copilot."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import MutableMapping
from pathlib import Path

import uvicorn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

LOCAL_ENV_MAX_BYTES = 65_536
LOCAL_ENV_MAX_LINES = 128
LOCAL_ENV_MAX_LINE_BYTES = 8_192
LOCAL_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
LOCAL_ENV_ALLOWED_KEYS = frozenset(
    {
        "MAPBOX_ATLAS_LICENSE",
        "OPENAI_API_KEY",
        "OPENAI_COPILOT_HISTORY_CHARACTERS",
        "OPENAI_COPILOT_HISTORY_MESSAGES",
        "OPENAI_COPILOT_MODEL",
        "OPENAI_COPILOT_TIMEOUT_SECONDS",
        "PRODUCT_APP_ORIGIN",
        "PRODUCT_COPILOT_ADAPTER",
        "PRODUCT_MAP_BASE_URL",
        "PRODUCT_MAP_PROVIDER",
    }
)


class LocalEnvironmentError(ValueError):
    """A stable, secret-free local environment loading failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def load_local_environment(
    repository_root: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Load a bounded ``.env.local`` without shell parsing or overwrites."""

    target_environment = os.environ if environ is None else environ
    try:
        root = repository_root.resolve()
        path = root / ".env.local"
        if path.is_symlink():
            raise LocalEnvironmentError("local_environment_symlink_not_allowed")
        if not path.exists():
            return 0
        resolved = path.resolve(strict=True)
        if resolved.parent != root:
            raise LocalEnvironmentError("local_environment_outside_repository")
        if not resolved.is_file():
            raise LocalEnvironmentError("local_environment_not_regular_file")
        size = resolved.stat().st_size
        if size > LOCAL_ENV_MAX_BYTES:
            raise LocalEnvironmentError("local_environment_too_large")
        raw = resolved.read_bytes()
    except LocalEnvironmentError:
        raise
    except OSError as exc:
        raise LocalEnvironmentError("local_environment_unreadable") from exc
    if len(raw) > LOCAL_ENV_MAX_BYTES:
        raise LocalEnvironmentError("local_environment_too_large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalEnvironmentError("local_environment_invalid_utf8") from exc

    lines = text.splitlines()
    if len(lines) > LOCAL_ENV_MAX_LINES:
        raise LocalEnvironmentError("local_environment_too_many_lines")
    parsed: dict[str, str] = {}
    for line in lines:
        if len(line.encode("utf-8")) > LOCAL_ENV_MAX_LINE_BYTES:
            raise LocalEnvironmentError("local_environment_line_too_large")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise LocalEnvironmentError("local_environment_malformed")
        key, value = line.split("=", 1)
        if LOCAL_ENV_KEY.fullmatch(key) is None or "\x00" in value or key in parsed:
            raise LocalEnvironmentError("local_environment_malformed")
        if key not in LOCAL_ENV_ALLOWED_KEYS:
            raise LocalEnvironmentError("local_environment_key_not_allowed")
        parsed[key] = value

    loaded = 0
    for key, value in parsed.items():
        if key not in target_environment:
            target_environment[key] = value
            loaded += 1
    return loaded


if __name__ == "__main__":
    try:
        load_local_environment(REPOSITORY_ROOT)
    except LocalEnvironmentError as exc:
        print(
            f"Itinerary Repair Copilot rejected .env.local ({exc.code}).",
            file=sys.stderr,
        )
        raise SystemExit(3) from None

from itinerary_system.product_app.api import PRODUCT_ID, create_product_app  # noqa: E402
from itinerary_system.product_app.config import ProductConfigError  # noqa: E402
from itinerary_system.product_app.registry import RegistryError  # noqa: E402
from itinerary_system.product_app.runtime import (  # noqa: E402
    PRODUCT_VERSION,
    product_build_id,
)
from itinerary_system.product_dashboard_models import ProductDashboardValidationError  # noqa: E402

HEALTH_SCHEMA = "product-health-v2"
EXPECTED_BUILD_ID = product_build_id(REPOSITORY_ROOT)
HEALTH_WAIT_SECONDS = 8.0
HEALTH_POLL_SECONDS = 0.1
HEALTH_REQUEST_SECONDS = 6.0
HEALTH_RESPONSE_LIMIT = 65_536
VALID_TOP_LEVEL_STATUSES = frozenset({"ready", "degraded", "failed"})
VALID_COMPONENT_STATUSES = frozenset({"ready", "degraded", "failed", "disabled"})
STABLE_CODE = re.compile(r"[a-z0-9_]{1,128}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=_port_number, default=8127)
    parser.add_argument("--open", action="store_true", help="Open /app after the health check passes.")
    parser.add_argument("--enable-legacy", action="store_true", help="Enable labeled debug-only legacy routes.")
    parser.add_argument("--state-root", type=Path, default=REPOSITORY_ROOT / ".product_app_state")
    return parser.parse_args()


def _port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def health(url: str, *, timeout_seconds: float = HEALTH_REQUEST_SECONDS) -> dict | None:
    """Return only a complete, matching product-health-v2 payload."""

    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=timeout_seconds) as response:
            raw = response.read(HEALTH_RESPONSE_LIMIT + 1)
        if len(raw) > HEALTH_RESPONSE_LIMIT:
            return None
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, urllib.error.URLError):
        return None
    return payload if _valid_product_health(payload) else None


def _valid_product_health(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != HEALTH_SCHEMA or payload.get("product_id") != PRODUCT_ID:
        return False
    status = payload.get("status")
    core_ready = payload.get("core_ready")
    if status not in VALID_TOP_LEVEL_STATUSES or not isinstance(core_ready, bool):
        return False
    if not isinstance(payload.get("ready"), bool) or payload["ready"] is not core_ready:
        return False
    if (status == "failed") is core_ready:
        return False
    if payload.get("product_version") != PRODUCT_VERSION:
        return False
    if payload.get("build_id") != EXPECTED_BUILD_ID:
        return False
    if payload.get("default_run") is not None and not isinstance(payload["default_run"], str):
        return False
    if not isinstance(payload.get("legacy_enabled"), bool):
        return False
    components = payload.get("components")
    required = {"registry", "default_workspace", "state_store", "map", "openai"}
    if not isinstance(components, dict) or not required.issubset(components):
        return False
    for name, component in components.items():
        if not isinstance(name, str) or not isinstance(component, dict):
            return False
        if component.get("name") != name or component.get("status") not in VALID_COMPONENT_STATUSES:
            return False
        if not isinstance(component.get("required_for_core"), bool):
            return False
        code = component.get("code")
        if not isinstance(code, str) or STABLE_CODE.fullmatch(code) is None:
            return False
        if not isinstance(component.get("checked_at"), str) or not component["checked_at"]:
            return False
    return True


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) == 0


def _limited_component_codes(payload: dict) -> str:
    components = payload.get("components", {})
    entries = [
        f"{name}={component['code']}"
        for name, component in sorted(components.items())
        if component["status"] in {"degraded", "failed"}
    ]
    return ", ".join(entries) if entries else "none"


def _report_health(url: str, payload: dict) -> None:
    if payload["core_ready"]:
        if payload["status"] == "degraded":
            print(
                "Itinerary Repair Copilot is running in degraded mode: "
                f"{_limited_component_codes(payload)}"
            )
        else:
            print(f"Itinerary Repair Copilot is ready at {url}/app")
        return
    print(
        "Itinerary Repair Copilot requires recovery: "
        f"{_limited_component_codes(payload)}",
        file=sys.stderr,
    )
    print(f"Recovery page: {url}/app", file=sys.stderr)


def open_when_ready(url: str) -> str:
    """Wait at most eight seconds, then open either the app or its recovery UI."""

    payload = wait_for_matching_health(url)
    if payload is not None:
        _report_health(url, payload)
        webbrowser.open(f"{url}/app")
        return "core_ready" if payload["core_ready"] else "recovery"
    print(
        "The service started, but no matching product-health-v2 response was received within eight seconds.",
        file=sys.stderr,
    )
    return "timeout"


def wait_for_matching_health(url: str) -> dict | None:
    """Retry matching product health within the launcher's fixed wait budget."""

    deadline = time.monotonic() + HEALTH_WAIT_SECONDS
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        payload = health(url, timeout_seconds=min(HEALTH_REQUEST_SECONDS, remaining))
        if payload is not None:
            return payload
        time.sleep(min(HEALTH_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    return None


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}"
    if port_in_use(args.host, args.port):
        existing = wait_for_matching_health(url)
        if existing is not None:
            _report_health(url, existing)
            if args.open:
                webbrowser.open(f"{url}/app")
            return 0 if existing["core_ready"] else 3
        print(
            f"Port {args.port} is occupied by a different, unhealthy, or out-of-date process. "
            "Stop the existing Itinerary Repair Copilot with Ctrl+C, then run this launcher again. "
            "An explicit --port may be used for an independent test service.",
            file=sys.stderr,
        )
        return 2

    try:
        app = create_product_app(
            repository_root=REPOSITORY_ROOT,
            state_root=args.state_root,
            enable_legacy=args.enable_legacy,
            application_host=args.host,
            application_port=args.port,
        )
    except (ProductConfigError, RegistryError, ProductDashboardValidationError):
        print(
            "Itinerary Repair Copilot could not initialize its validated runtime configuration.",
            file=sys.stderr,
        )
        return 3
    if args.open:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    print(f"Itinerary Repair Copilot: {url}/app")
    print("Press Ctrl+C to stop the local service.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
