"""Validated runtime configuration for the loopback-only product service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ProductConfigError(ValueError):
    """Raised with a stable code when product configuration is unsafe."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65_535:
        raise ProductConfigError("invalid_application_port")
    return value


def _loopback_http_url(value: str, *, expected_port: int | None = None) -> str:
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise ProductConfigError("invalid_loopback_url") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is None
        or (expected_port is not None and port != expected_port)
    ):
        raise ProductConfigError("invalid_loopback_url")
    return f"http://{parsed.hostname}:{port}"


def _bounded_environment_integer(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    error_code: str,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value.isascii() or not value.isdecimal():
        raise ProductConfigError(error_code)
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ProductConfigError(error_code)
    return parsed


@dataclass(frozen=True)
class ProductRuntimeConfig:
    """Configuration with secrets excluded from repr and public contracts."""

    repository_root: Path
    registry_path: Path
    state_root: Path
    application_host: str = "127.0.0.1"
    application_port: int = 8127
    application_origin: str = "http://127.0.0.1:8127"
    map_provider: str = "maplibre_pmtiles"
    map_base_url: str = "http://127.0.0.1:8080"
    routing_base_url: str = "http://127.0.0.1:5000"
    mapbox_atlas_license: str | None = field(default=None, repr=False)
    copilot_adapter: str = "deterministic"
    openai_model: str = "gpt-5.6-terra"
    openai_api_key: str | None = field(default=None, repr=False)
    openai_timeout_seconds: int = 30
    openai_history_messages: int = 12
    openai_history_characters: int = 12_000
    enable_legacy: bool = False
    additional_allowed_authorities: tuple[str, ...] = ()

    @classmethod
    def from_environment(
        cls,
        *,
        repository_root: Path,
        registry_path: Path,
        state_root: Path,
        application_host: str = "127.0.0.1",
        application_port: int = 8127,
        enable_legacy: bool = False,
        additional_allowed_authorities: tuple[str, ...] = (),
    ) -> ProductRuntimeConfig:
        if application_host != "127.0.0.1":
            raise ProductConfigError("non_loopback_host_not_allowed")
        checked_port = _port(application_port)
        default_origin = f"http://{application_host}:{checked_port}"
        origin = _loopback_http_url(os.environ.get("PRODUCT_APP_ORIGIN", default_origin))
        if urlparse(origin).port != checked_port:
            raise ProductConfigError("application_origin_port_mismatch")
        map_provider = os.environ.get("PRODUCT_MAP_PROVIDER", "maplibre_pmtiles").strip().lower()
        if map_provider not in {"maplibre_pmtiles", "mapbox_atlas_v3"}:
            raise ProductConfigError("invalid_map_provider")
        map_base_url = _loopback_http_url(
            os.environ.get("PRODUCT_MAP_BASE_URL", "http://127.0.0.1:8080"),
            expected_port=8080,
        )
        routing_base_url = _loopback_http_url(
            os.environ.get("PRODUCT_ROUTING_BASE_URL", "http://127.0.0.1:5000")
        )
        adapter = os.environ.get("PRODUCT_COPILOT_ADAPTER", "deterministic").strip().lower()
        if adapter not in {"deterministic", "openai"}:
            raise ProductConfigError("invalid_copilot_adapter")
        model = os.environ.get("OPENAI_COPILOT_MODEL", "gpt-5.6-terra").strip()
        if not model:
            raise ProductConfigError("invalid_openai_model")
        timeout_seconds = _bounded_environment_integer(
            "OPENAI_COPILOT_TIMEOUT_SECONDS",
            default=30,
            minimum=1,
            maximum=60,
            error_code="invalid_openai_copilot_timeout_seconds",
        )
        history_messages = _bounded_environment_integer(
            "OPENAI_COPILOT_HISTORY_MESSAGES",
            default=12,
            minimum=0,
            maximum=20,
            error_code="invalid_openai_copilot_history_messages",
        )
        history_characters = _bounded_environment_integer(
            "OPENAI_COPILOT_HISTORY_CHARACTERS",
            default=12_000,
            minimum=0,
            maximum=20_000,
            error_code="invalid_openai_copilot_history_characters",
        )
        authorities = tuple(str(item).strip().lower() for item in additional_allowed_authorities)
        if any(not item or "/" in item or "@" in item for item in authorities):
            raise ProductConfigError("invalid_additional_authority")
        return cls(
            repository_root=repository_root.resolve(),
            registry_path=registry_path.resolve(),
            state_root=state_root.resolve(),
            application_host=application_host,
            application_port=checked_port,
            application_origin=origin,
            map_provider=map_provider,
            map_base_url=map_base_url,
            routing_base_url=routing_base_url,
            mapbox_atlas_license=os.environ.get("MAPBOX_ATLAS_LICENSE") or None,
            copilot_adapter=adapter,
            openai_model=model,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_timeout_seconds=timeout_seconds,
            openai_history_messages=history_messages,
            openai_history_characters=history_characters,
            enable_legacy=enable_legacy,
            additional_allowed_authorities=authorities,
        )

    @property
    def allowed_authorities(self) -> frozenset[str]:
        port = self.application_port
        return frozenset(
            {f"127.0.0.1:{port}", f"localhost:{port}", *self.additional_allowed_authorities}
        )

    @property
    def allowed_origins(self) -> frozenset[str]:
        port = self.application_port
        return frozenset({f"http://127.0.0.1:{port}", f"http://localhost:{port}"})
