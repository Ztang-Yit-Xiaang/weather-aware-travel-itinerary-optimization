"""Truthful runtime readiness orchestration for the local product."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from fastapi import HTTPException

from .config import ProductRuntimeConfig
from .conversations import ConversationError
from .map_asset_contract import (
    EXPECTED_STYLE_ATTRIBUTION,
    required_fontstacks,
    valid_provenance,
)
from .models import ComponentHealthV1, MapConfigurationV2, ProductHealthV2
from .persistence import LocalStateLayout
from .registry import ProductRunRegistry, RegistryError
from .routing_runtime import RuntimeRoutingError, RuntimeRoutingService
from .service import ProductService

PRODUCT_ID = "itinerary-repair-copilot"
PRODUCT_VERSION = "0.2.0"
MAP_CACHE_SECONDS = 2.0
MAP_TIMEOUT_SECONDS = 1.0

BUILD_FINGERPRINT_SUFFIXES = frozenset({
    ".css",
    ".html",
    ".js",
    ".json",
    ".py",
    ".svg",
    ".webmanifest",
})


def product_build_id(repository_root: Path) -> str:
    """Fingerprint the browser/backend contract loaded by one server process."""

    digest = hashlib.sha256()
    product_root = repository_root / "src" / "itinerary_system" / "product_app"
    candidates = [repository_root / "scripts" / "run_product_app.py"]
    candidates.extend(
        path
        for path in product_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in BUILD_FINGERPRINT_SUFFIXES
    )
    for path in sorted(candidates):
        relative_path = path.relative_to(repository_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _component(name: str, status: str, required: bool, code: str) -> ComponentHealthV1:
    return ComponentHealthV1(name=name, status=status, required_for_core=required, code=code, checked_at=_now())


class ProductRuntime:
    """Owns initialization and sanitized component readiness."""

    def __init__(self, config: ProductRuntimeConfig) -> None:
        self.config = config
        self.build_id = product_build_id(config.repository_root)
        self.registry: ProductRunRegistry | None = None
        self.service: ProductService | None = None
        self.routing: RuntimeRoutingService | None = None
        self.state_layout = LocalStateLayout(config.state_root)
        self._components: dict[str, ComponentHealthV1] = {}
        self._initialized = False
        self._map_checked_monotonic = 0.0
        self._cache_lock = threading.RLock()

    def initialize(self) -> ProductHealthV2:
        with self._cache_lock:
            if self._initialized:
                return self.health()
            service_state_error: str | None = None
            try:
                self.registry = ProductRunRegistry(self.config.repository_root, self.config.registry_path)
            except RegistryError as exc:
                self._components["registry"] = _component("registry", "failed", True, exc.code)
                self._components["default_workspace"] = _component(
                    "default_workspace", "failed", True, "registry_required"
                )
            else:
                self._components["registry"] = _component("registry", "ready", True, "registry_ready")
                try:
                    self.service = ProductService(
                        self.registry,
                        self.config.state_root,
                        runtime_config=self.config,
                    )
                    self.service.load(self.registry.default.run_id)
                except ConversationError as exc:
                    self.service = None
                    service_state_error = exc.code
                    self._components["default_workspace"] = _component(
                        "default_workspace", "failed", True, "product_service_unavailable"
                    )
                except Exception:
                    self.service = None
                    self._components["default_workspace"] = _component(
                        "default_workspace", "failed", True, "workspace_invalid"
                    )
                else:
                    self._components["default_workspace"] = _component(
                        "default_workspace", "ready", True, "workspace_ready"
                    )
            state_readiness = self.state_layout.initialize()
            legacy = state_readiness.legacy or self.state_layout.detect_legacy()
            if service_state_error is not None:
                self._components["state_store"] = _component(
                    "state_store", "failed", True, service_state_error
                )
            elif state_readiness.ready:
                has_legacy = bool(legacy.workspace_pointer_count or legacy.decision_file_count)
                self._components["state_store"] = _component(
                    "state_store",
                    "degraded" if has_legacy else "ready",
                    True,
                    "legacy_state_deferred" if has_legacy else state_readiness.code,
                )
            else:
                self._components["state_store"] = _component(
                    "state_store", "failed", True, state_readiness.code
                )
            self._components["openai"] = self._openai_health()
            self._components["map"] = self._probe_map()
            try:
                self.routing = RuntimeRoutingService(
                    base_url=self.config.routing_base_url,
                    timeout_seconds=1.0,
                )
            except RuntimeRoutingError as exc:
                self._components["routing"] = _component(
                    "routing", "degraded", False, exc.code
                )
            else:
                self._components["routing"] = self._probe_routing()
            self._map_checked_monotonic = time.monotonic()
            self._initialized = True
            return self._health_payload()

    def health(self, *, force: bool = False) -> ProductHealthV2:
        if not self._initialized:
            return self.initialize()
        with self._cache_lock:
            if force or time.monotonic() - self._map_checked_monotonic >= MAP_CACHE_SECONDS:
                self._components["map"] = self._probe_map()
                if self.routing is not None:
                    self._components["routing"] = self._probe_routing()
                self._map_checked_monotonic = time.monotonic()
            return self._health_payload()

    def require_service(self) -> ProductService:
        health = self.health()
        if not health.core_ready or self.service is None:
            raise HTTPException(status_code=503, detail="product_core_not_ready")
        return self.service

    def require_routing(self) -> RuntimeRoutingService:
        """Return a validated loopback router or a stable unavailable response."""

        self.health()
        with self._cache_lock:
            component = self._components.get("routing")
            if self.routing is not None and component is not None and component.status == "ready":
                return self.routing
            if self.routing is not None:
                component = self._probe_routing()
                self._components["routing"] = component
                self._map_checked_monotonic = time.monotonic()
                if component.status == "ready":
                    return self.routing
            code = component.code if component is not None else "routing_unavailable"
            raise HTTPException(status_code=503, detail=code)

    def map_configuration(self) -> MapConfigurationV2:
        map_health = self.health().components["map"]
        contract = self._map_contract()
        return MapConfigurationV2(status=map_health.status, **contract)

    def _health_payload(self) -> ProductHealthV2:
        core = tuple(component for component in self._components.values() if component.required_for_core)
        core_ready = bool(core) and all(component.status != "failed" for component in core)
        if not core_ready:
            status = "failed"
        elif any(component.status == "degraded" for component in self._components.values()):
            status = "degraded"
        else:
            status = "ready"
        default_run = self.registry.default.run_id if self.registry is not None else None
        return ProductHealthV2(
            product_id=PRODUCT_ID,
            product_version=PRODUCT_VERSION,
            build_id=self.build_id,
            status=status,
            core_ready=core_ready,
            default_run=default_run,
            legacy_enabled=self.config.enable_legacy,
            components=dict(self._components),
        )

    def _openai_health(self) -> ComponentHealthV1:
        if self.config.copilot_adapter == "deterministic":
            return _component("openai", "disabled", False, "deterministic_adapter_selected")
        if not self.config.openai_api_key:
            return _component("openai", "degraded", False, "openai_not_configured")
        if self.service is not None:
            if (
                self.service.copilot is not None
                and self.service.copilot.provider_name == "openai"
            ):
                return _component("openai", "ready", False, "openai_configured")
            return _component(
                "openai",
                "degraded",
                False,
                self.service.copilot_unavailable_code or "openai_transport_unavailable",
            )
        if self.registry is not None:
            return _component("openai", "degraded", False, "openai_transport_unavailable")
        try:
            self._construct_openai_adapter()
        except Exception:
            return _component("openai", "degraded", False, "openai_transport_unavailable")
        return _component("openai", "ready", False, "openai_configured")

    def _construct_openai_adapter(self) -> object:
        """Validate local provider construction without making a billed request."""

        from .openai_copilot import OpenAICopilotAdapter

        return OpenAICopilotAdapter(
            model=self.config.openai_model,
            api_key=self.config.openai_api_key,
            timeout_seconds=self.config.openai_timeout_seconds,
            history_messages=self.config.openai_history_messages,
            history_characters=self.config.openai_history_characters,
        )

    def _probe_routing(self) -> ComponentHealthV1:
        """Exercise one bounded nearest-road query without mutating product state."""

        if self.routing is None:
            return _component("routing", "degraded", False, "routing_unavailable")
        try:
            result = self.routing.nearest("routing_health_probe", (34.0522, -118.2437))
        except RuntimeRoutingError as exc:
            return _component("routing", "degraded", False, exc.code)
        if result.validation_state == "rejected":
            return _component("routing", "degraded", False, "routing_probe_snap_too_far")
        return _component("routing", "ready", False, "runtime_osrm_ready")

    def _map_contract(self) -> dict[str, Any]:
        base = self.config.map_base_url
        if self.config.map_provider == "mapbox_atlas_v3":
            return {
                "provider": "mapbox_atlas_v3",
                "base_url": base,
                "style_url": f"{base}/data/styles/mapbox/streets-v12.json",
                "script_url": f"{base}/mapbox-gl-js/mapbox-gl.js",
                "stylesheet_url": f"{base}/mapbox-gl-js/mapbox-gl.css",
                "protocol_script_url": None,
                "provenance_url": None,
                "attribution": "Mapbox Atlas",
                "attribution_url": None,
                "runtime_license": self.config.mapbox_atlas_license,
            }
        return {
            "provider": "maplibre_pmtiles",
            "base_url": base,
            "style_url": f"{base}/styles/protomaps-light.json",
            "script_url": f"{base}/maplibre/maplibre-gl.js",
            "stylesheet_url": f"{base}/maplibre/maplibre-gl.css",
            "protocol_script_url": f"{base}/pmtiles/pmtiles.js",
            "provenance_url": f"{base}/provenance.json",
            "attribution": "Protomaps | © OpenStreetMap contributors",
            "attribution_url": "https://www.openstreetmap.org/copyright",
            "runtime_license": None,
        }

    def _probe_map(self) -> ComponentHealthV1:
        contract = self._map_contract()
        provider = self.config.map_provider
        if provider == "mapbox_atlas_v3" and not self.config.mapbox_atlas_license:
            return _component("map", "degraded", False, "atlas_runtime_license_missing")
        try:
            if self._fetch_checked("/healthz")[2] != 200:
                return _component("map", "degraded", False, "map_health_invalid")
            if self._fetch_checked(contract["script_url"])[2] != 200:
                return _component("map", "degraded", False, "map_script_unavailable")
            if self._fetch_checked(contract["stylesheet_url"])[2] != 200:
                return _component("map", "degraded", False, "map_stylesheet_unavailable")
            protocol_url = contract["protocol_script_url"]
            if protocol_url is not None and self._fetch_checked(protocol_url)[2] != 200:
                return _component("map", "degraded", False, "pmtiles_protocol_unavailable")
            style_bytes, _, status = self._fetch_checked(contract["style_url"])
            if status != 200:
                return _component("map", "degraded", False, "map_style_unavailable")
            style = json.loads(style_bytes.decode("utf-8"))
            pmtiles_url = self._validate_style_and_find_pmtiles(style)
            if pmtiles_url is None:
                return _component("map", "degraded", False, "pmtiles_source_missing")
            for asset_kind, asset_url in self._style_probe_urls(style):
                if self._fetch_checked(asset_url)[2] != 200:
                    return _component("map", "degraded", False, f"map_{asset_kind}_unavailable")
            provenance_url = contract["provenance_url"]
            if provenance_url is not None:
                provenance_bytes, _, provenance_status = self._fetch_checked(provenance_url)
                if provenance_status != 200 or not self._valid_provenance(
                    provenance_bytes, pmtiles_url
                ):
                    return _component("map", "degraded", False, "map_provenance_invalid")
            range_body, headers, range_status = self._fetch_checked(pmtiles_url, range_request=True)
            content_range = headers.get("Content-Range", "")
            if range_status != 206:
                return _component("map", "degraded", False, "pmtiles_range_status_invalid")
            if "bytes" not in headers.get("Accept-Ranges", "").lower():
                return _component("map", "degraded", False, "pmtiles_accept_ranges_missing")
            range_match = re.fullmatch(r"bytes 0-1023/(\d+)", content_range)
            if range_match is None or int(range_match.group(1)) <= 1023:
                return _component("map", "degraded", False, "pmtiles_content_range_invalid")
            if headers.get("Content-Length") != "1024" or len(range_body) != 1024:
                return _component("map", "degraded", False, "pmtiles_range_length_invalid")
            if range_body[:7] != b"PMTiles" or range_body[7] != 3:
                return _component("map", "degraded", False, "pmtiles_header_invalid")
        except json.JSONDecodeError:
            return _component("map", "degraded", False, "map_style_invalid")
        except UnicodeDecodeError:
            return _component("map", "degraded", False, "map_style_invalid")
        except ValueError as exc:
            return _component("map", "degraded", False, str(exc))
        except (OSError, urllib.error.URLError, TimeoutError):
            code = "maplibre_unavailable" if provider == "maplibre_pmtiles" else "atlas_unavailable"
            return _component("map", "degraded", False, code)
        code = "maplibre_ready" if provider == "maplibre_pmtiles" else "atlas_ready"
        return _component("map", "ready", False, code)

    def _fetch(self, path_or_url: str, *, range_request: bool = False) -> tuple[bytes, Any, int]:
        url = path_or_url if path_or_url.startswith("http://") else urljoin(f"{self.config.map_base_url}/", path_or_url.lstrip("/"))
        if not self._is_map_url(url):
            raise ValueError("map_url_not_loopback")
        headers = {"Origin": self.config.application_origin}
        if range_request:
            headers["Range"] = "bytes=0-1023"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=MAP_TIMEOUT_SECONDS) as response:
            if not self._is_map_url(response.geturl()):
                raise ValueError("map_redirect_not_loopback")
            return response.read(1_048_576), response.headers, response.status

    def _fetch_checked(
        self, path_or_url: str, *, range_request: bool = False
    ) -> tuple[bytes, Any, int]:
        body, headers, status = self._fetch(path_or_url, range_request=range_request)
        if headers.get("Access-Control-Allow-Origin") != self.config.application_origin:
            raise ValueError("map_cors_origin_invalid")
        return body, headers, status

    def _validate_style_and_find_pmtiles(self, style: Any) -> str | None:
        if not isinstance(style, dict):
            raise ValueError("map_style_invalid")
        imports = style.get("imports", [])
        if not isinstance(imports, list) or imports:
            raise ValueError("map_style_imports_not_supported")
        pmtiles_url: str | None = None
        resources: list[str] = []
        for field_name in ("sprite", "glyphs"):
            value = style.get(field_name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("map_style_invalid")
                resources.append(value)
        sources = style.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("map_style_invalid")
        for source in sources.values():
            if not isinstance(source, dict):
                raise ValueError("map_style_invalid")
            if isinstance(source.get("url"), str):
                resources.append(source["url"])
            if isinstance(source.get("data"), str):
                resources.append(source["data"])
            if (
                self.config.map_provider == "maplibre_pmtiles"
                and ".pmtiles" in str(source.get("url", "")).lower()
            ):
                attribution = source.get("attribution")
                if attribution != EXPECTED_STYLE_ATTRIBUTION:
                    raise ValueError("map_attribution_invalid")
            tiles = source.get("tiles", [])
            if not isinstance(tiles, list) or any(not isinstance(item, str) for item in tiles):
                raise ValueError("map_style_invalid")
            resources.extend(tiles)
        for value in resources:
            candidate = value.removeprefix("pmtiles://")
            candidate_url = candidate if candidate.startswith("http://") else urljoin(
                f"{self.config.map_base_url}/", candidate.lstrip("/")
            )
            if not self._is_map_url(candidate_url):
                raise ValueError("map_style_url_not_loopback")
            if ".pmtiles" in candidate.lower():
                pmtiles_url = candidate_url
        return pmtiles_url

    def _style_probe_urls(self, style: dict[str, Any]) -> list[tuple[str, str]]:
        sprite = style.get("sprite")
        glyphs = style.get("glyphs")
        fontstacks = required_fontstacks(style)
        if not isinstance(sprite, str) or not isinstance(glyphs, str) or fontstacks is None:
            raise ValueError("map_style_assets_invalid")
        probes = [
            ("sprite", f"{sprite}.json"),
            ("sprite", f"{sprite}.png"),
            ("sprite", f"{sprite}@2x.json"),
            ("sprite", f"{sprite}@2x.png"),
        ]
        for fontstack in sorted(fontstacks):
            glyph_url = glyphs.replace("{fontstack}", quote(fontstack, safe=","))
            probes.append(("glyph", glyph_url.replace("{range}", "0-255")))
        return probes

    def _valid_provenance(self, body: bytes, pmtiles_url: str) -> bool:
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        artifact_path = urlparse(pmtiles_url).path
        return self._is_map_url(pmtiles_url) and valid_provenance(
            value, expected_artifact_url=artifact_path
        )

    def _is_map_url(self, url: str) -> bool:
        parsed = urlparse(url)
        configured = urlparse(self.config.map_base_url)
        return (
            parsed.scheme == "http"
            and parsed.hostname == configured.hostname
            and parsed.port == configured.port
            and parsed.username is None
            and parsed.password is None
        )
