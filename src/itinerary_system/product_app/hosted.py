"""Explicit owner-only hosting; the ordinary launcher remains loopback-only."""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .config import ProductConfigError, ProductRuntimeConfig
from .models import ComponentHealthV1
from .runtime import PRODUCT_ID, ProductRuntime

MAX_HOSTED_RESPONSE_BYTES = 16 * 1024 * 1024


def public_supabase_key(value: str) -> str:
    """Only a publishable key or the legacy anonymous JWT may reach a browser."""
    if isinstance(value, str) and len(value) <= 8192:
        if re.fullmatch(r"sb_publishable_[\w-]+", value, flags=re.ASCII):
            return value
        if re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", value, flags=re.ASCII):
            try:
                part = value.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
                if isinstance(claims, dict) and claims.get("role") == "anon":
                    return value
            except (ValueError, UnicodeError):
                pass
    raise ProductConfigError("supabase_publishable_key_invalid")


def _https_origin(value: str, code: str) -> str:
    try:
        url = urllib.parse.urlsplit(value)
        valid = (
            url.scheme == "https"
            and bool(url.hostname)
            and url.port in {None, 443}
            and len(url.hostname) <= 253
            and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in url.hostname.split("."))
        )
    except ValueError:
        valid = False
    if not valid or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment:
        raise ProductConfigError(code)
    return f"https://{url.hostname}"


@dataclass(frozen=True)
class HostedSettings:
    api_origin: str
    frontend_origin: str
    supabase_url: str
    owner_id: str
    database_url: str = field(repr=False)
    supabase_publishable_key: str = field(repr=False)
    ors_api_key: str = field(repr=False)
    copilot_adapter: str = "deterministic"
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-5.6-terra"

    @classmethod
    def from_environment(cls) -> HostedSettings:
        def required(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise ProductConfigError(name.lower() + "_required")
            return value

        api = _https_origin(required("HOSTED_API_ORIGIN"), "hosted_api_origin_invalid")
        frontend = _https_origin(required("HOSTED_FRONTEND_ORIGIN"), "hosted_frontend_origin_invalid")
        supabase = _https_origin(required("SUPABASE_URL"), "supabase_url_invalid")
        if not urllib.parse.urlsplit(supabase).hostname.endswith(".supabase.co"):
            raise ProductConfigError("supabase_url_invalid")
        try:
            owner = str(UUID(required("HOSTED_OWNER_ID")))
        except ValueError:
            raise ProductConfigError("hosted_owner_id_invalid") from None
        adapter = os.environ.get("PRODUCT_COPILOT_ADAPTER", "deterministic")
        if adapter not in {"deterministic", "openai"}:
            raise ProductConfigError("invalid_copilot_adapter")
        if adapter == "openai" and os.environ.get("HOSTED_ENABLE_PAID_OPENAI") != "true":
            raise ProductConfigError("paid_openai_not_enabled")
        return cls(
            api_origin=api,
            frontend_origin=frontend,
            supabase_url=supabase,
            owner_id=owner,
            database_url=required("DATABASE_URL"),
            supabase_publishable_key=public_supabase_key(required("SUPABASE_PUBLISHABLE_KEY")),
            ors_api_key=required("ORS_API_KEY"),
            copilot_adapter=adapter,
            openai_api_key=required("OPENAI_API_KEY") if adapter == "openai" else None,
            openai_model=os.environ.get("OPENAI_COPILOT_MODEL", "gpt-5.6-terra"),
        )


@dataclass(frozen=True)
class HostedRuntimeConfig(ProductRuntimeConfig):
    frontend_origin: str = ""

    @property
    def allowed_authorities(self) -> frozenset[str]:
        return frozenset({urllib.parse.urlsplit(self.application_origin).netloc})

    @property
    def allowed_origins(self) -> frozenset[str]:
        return frozenset({self.application_origin, self.frontend_origin})


class HostedProductRuntime(ProductRuntime):
    def __init__(self, config: HostedRuntimeConfig, router: Any) -> None:
        super().__init__(config, routing=router)

    def initialize(self):
        result = super().initialize()
        if self.service is not None:
            # Shared public Overpass is not the hosted app's discovery backend.
            # Keep the labelled catalog/cache until a suitable service is configured.
            self.service.nearby_places.online_enabled = False
        return result

    def _probe_routing(self) -> ComponentHealthV1:
        # Health checks must not consume the owner's routing or LLM quota.
        return ComponentHealthV1("routing", "ready", False, "ors_configured_not_live_probed")

    def _probe_map(self) -> ComponentHealthV1:
        vendor = self.config.repository_root / "src/itinerary_system/product_app/static/vendor"
        ready = all(
            (vendor / name).is_file()
            for name in ("maplibre-gl.js", "maplibre-gl.css", "fonts/Noto Sans Regular/0-255.pbf")
        )
        return ComponentHealthV1(
            "map",
            "ready" if ready else "degraded",
            False,
            "hosted_raster_configured" if ready else "hosted_map_assets_missing",
        )

    def _map_contract(self) -> dict[str, Any]:
        base = self.config.application_origin
        return {
            "provider": "maplibre_raster",
            "base_url": base,
            "style_url": base + "/public/map-style.json",
            "script_url": base + "/static/vendor/maplibre-gl.js",
            "stylesheet_url": base + "/static/vendor/maplibre-gl.css",
            "protocol_script_url": None,
            "provenance_url": None,
            "attribution": "© OpenStreetMap contributors",
            "attribution_url": "https://www.openstreetmap.org/copyright",
            "runtime_license": None,
            "range_requests_required": False,
        }


class SupabaseOwnerAuth:
    """Resolve the token through Supabase Auth, then enforce the configured owner."""

    def __init__(self, settings: HostedSettings) -> None:
        self.settings = settings

    def verify(self, authorization: str) -> bool:
        if not authorization.startswith("Bearer ") or not 8 <= len(authorization) <= 8192:
            return False
        request = urllib.request.Request(
            self.settings.supabase_url + "/auth/v1/user",
            headers={
                "Authorization": authorization,
                "apikey": self.settings.supabase_publishable_key,
                "Accept": "application/json",
            },
        )

        class RejectRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_args, **_kwargs):
                raise ValueError("auth_redirect_rejected")

        try:
            with urllib.request.build_opener(RejectRedirect()).open(request, timeout=5) as response:
                payload = response.read(65_537)
                if response.status != 200 or len(payload) > 65_536:
                    return False
            user = json.loads(payload)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return False
            raise RuntimeError("hosted_auth_unavailable") from None
        except (OSError, ValueError, http.client.HTTPException):
            raise RuntimeError("hosted_auth_unavailable") from None
        return isinstance(user, dict) and user.get("id") == self.settings.owner_id


class HostedBoundary:
    """Authenticate and durably commit state before acknowledging an API result."""

    def __init__(self, app: Any, *, settings: HostedSettings, store: Any, auth: Any, lock: asyncio.Lock) -> None:
        self.app, self.settings, self.store, self.auth, self.lock = app, settings, store, auth, lock

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        path = scope.get("path", "")
        if not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return
        if headers.get(b"origin") not in {None, self.settings.api_origin, self.settings.frontend_origin}:
            await JSONResponse({"detail": "origin_not_allowed"}, 403)(scope, receive, send)
            return
        try:
            authorized = await asyncio.to_thread(self.auth.verify, headers.get(b"authorization", ""))
        except Exception:
            await JSONResponse({"detail": "hosted_auth_unavailable"}, 503)(scope, receive, send)
            return
        if not authorized:
            await JSONResponse({"detail": "owner_sign_in_required"}, 401)(scope, receive, send)
            return
        messages, total = [], 0

        async def buffer(message):
            nonlocal total
            if message.get("type") == "http.response.start" and message.get("status") in {499, 500, 504}:
                # The core sanitizes exceptions into HTTP 500. A timed-out preview
                # may also leave an unpublished worker writing its disposable cache.
                # Discard that runtime instead of committing uncertain local state.
                raise RuntimeError("hosted_handler_failed_restart_required")
            total += len(message.get("body", b""))
            if total > MAX_HOSTED_RESPONSE_BYTES:
                raise ValueError("hosted_response_limit_exceeded")
            messages.append(message)

        async with self.lock:
            try:
                with self.store.transaction():
                    await self.app(scope, receive, buffer)
            except Exception:
                # Store transactions poison a failed/stale runtime. Never acknowledge
                # a mutation whose durable outcome is unknown, or expose DB credentials.
                await JSONResponse({"detail": "hosted_state_unavailable"}, 503)(scope, receive, send)
                return
        for message in messages:
            await send(message)


def create_hosted_app(
    *,
    settings: HostedSettings | None = None,
    repository_root: Path | None = None,
    state_root: Path | None = None,
    store: Any = None,
    auth: Any = None,
    router: Any = None,
) -> FastAPI:
    """One owner, one process, PostgreSQL durability, and explicitly bounded providers."""
    settings = settings or HostedSettings.from_environment()
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    ephemeral = tempfile.TemporaryDirectory(prefix="itinerary-hosted-") if state_root is None else None
    cache = state_root or Path(ephemeral.name)
    public_settings = {
        "api_origin": settings.api_origin,
        "supabase_url": settings.supabase_url,
        "supabase_publishable_key": public_supabase_key(settings.supabase_publishable_key),
        "static_frontend": True,
    }
    if store is None:
        from .postgres_state import PostgresStateStore

        store = PostgresStateStore(cache, settings.database_url, settings.owner_id)
    store.restore()
    if router is None:
        from .ors_runtime import ORSRuntimeRoutingService

        router = ORSRuntimeRoutingService(api_key=settings.ors_api_key, cache_root=cache / "routing_ors")
    config = HostedRuntimeConfig(
        repository_root=root,
        registry_path=root / "configs/product_app_registry.json",
        state_root=cache,
        application_origin=settings.api_origin,
        frontend_origin=settings.frontend_origin,
        map_provider="maplibre_raster",
        copilot_adapter=settings.copilot_adapter,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    runtime = HostedProductRuntime(config, router)
    from .api import create_product_app

    with store.transaction():
        core = create_product_app(repository_root=root, state_root=cache, runtime=runtime)
    serial = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            async with core.router.lifespan_context(core):
                with store.transaction():
                    pass
                yield
        finally:
            if ephemeral is not None:
                ephemeral.cleanup()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.product_runtime, app.state.hosted_store = runtime, store

    @app.get("/static/js/deployment.js")
    def deployment_config():
        return Response(
            "window.PRODUCT_DEPLOYMENT = " + json.dumps(public_settings) + ";\n", media_type="text/javascript"
        )

    @app.get("/healthz")
    async def healthz():
        try:
            async with serial:
                store.probe()
            ready = runtime.health().core_ready
        except Exception:
            ready = False
        return JSONResponse(
            {"product_id": PRODUCT_ID, "build_id": runtime.build_id, "status": "ready" if ready else "unavailable"},
            200 if ready else 503,
        )

    @app.get("/public/map-style.json")
    def map_style():
        return {
            "version": 8,
            "glyphs": settings.api_origin + "/static/vendor/fonts/{fontstack}/{range}.pbf",
            "sources": {
                "osm": {
                    "type": "raster",
                    "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    "tileSize": 256,
                    "attribution": "© OpenStreetMap contributors",
                },
                "labels": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}},
            },
            "layers": [
                {"id": "osm", "type": "raster", "source": "osm"},
                {
                    "id": "label-font",
                    "type": "symbol",
                    "source": "labels",
                    "layout": {"text-field": ["get", "name"], "text-font": ["Noto Sans Regular"]},
                },
            ],
        }

    @app.middleware("http")
    async def hosted_headers(request, call_next):
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob: https://tile.openstreetmap.org https://www.nps.gov; "
            f"connect-src 'self' https://tile.openstreetmap.org {settings.supabase_url}; "
            "font-src 'self'; worker-src 'self' blob:; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/", core)
    app.add_middleware(
        HostedBoundary, settings=settings, store=store, auth=auth or SupabaseOwnerAuth(settings), lock=serial
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin, settings.api_origin],
        allow_methods=["GET", "HEAD", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Session-Id", "X-Session-Token", "X-Product-Build-Id"],
        expose_headers=["Content-Disposition"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def hosted_host(request, call_next):
        authority = urllib.parse.urlsplit(settings.api_origin).netloc
        host = request.headers.get("host", "").lower()
        railway_health = request.url.path == "/healthz" and host == "healthcheck.railway.app"
        if host != authority.lower() and not railway_health:
            return JSONResponse({"detail": "host_not_allowed"}, 400)
        return await call_next(request)

    return app
