"""Host, origin, response-header, and bounded-request security boundary."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException, Request
from starlette.responses import Response

from .config import ProductRuntimeConfig

MAX_JSON_BODY_BYTES = 20_000


class ProductSecurityBoundary:
    def __init__(self, config: ProductRuntimeConfig) -> None:
        self.config = config

    def validate_host(self, request: Request) -> None:
        authority = request.headers.get("host", "").strip().lower()
        if authority not in self.config.allowed_authorities:
            raise HTTPException(status_code=400, detail="host_not_allowed")

    def validate_origin(self, request: Request) -> None:
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return
        origin = request.headers.get("origin")
        if origin is not None and origin not in self.config.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")

    def apply_headers(self, request: Request, response: Response) -> Response:
        map_origin = self.config.map_base_url
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' {map_origin}; "
            f"style-src 'self' {map_origin}; "
            f"img-src 'self' data: blob: {map_origin}; "
            f"font-src 'self' {map_origin}; "
            f"connect-src 'self' {map_origin}; "
            "worker-src 'self' blob:; manifest-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif (
            request.url.path == "/app"
            or request.url.path.startswith("/app/")
            or request.url.path.startswith("/static/")
        ):
            response.headers["Cache-Control"] = "no-cache"
        return response

    async def read_bounded_json(self, request: Request, limit: int = MAX_JSON_BODY_BYTES) -> dict[str, Any]:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                parsed_length = int(content_length)
                if parsed_length < 0:
                    raise HTTPException(status_code=400, detail="invalid_content_length")
                if parsed_length > limit:
                    raise HTTPException(status_code=413, detail="request_too_large")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid_content_length") from exc
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                raise HTTPException(status_code=413, detail="request_too_large")
            body.extend(chunk)
        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid_json") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="object_body_required")
        return payload


def validate_envelope(
    payload: dict[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    return payload


def validate_identifier(value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    text = value.strip()
    if (
        not text
        or len(text) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", text) is None
    ):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    return text
