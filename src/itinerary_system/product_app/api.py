"""FastAPI boundary for the loopback-only Itinerary Repair Copilot."""

from __future__ import annotations

import io
import math
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..product_dashboard_models import safe_run_relative_path
from .candidate_discovery import (
    DEFAULT_LIMIT as POI_CANDIDATE_DEFAULT_LIMIT,
)
from .candidate_discovery import (
    DEFAULT_MAX_DETOUR_MINUTES,
    POICandidateDiscoveryError,
)
from .config import ProductRuntimeConfig
from .conversations import ConversationError
from .map_interactions import MapInteractionError, MapInteractionService
from .registry import RegistryError
from .runtime import PRODUCT_ID as RUNTIME_PRODUCT_ID
from .runtime import PRODUCT_VERSION, ProductRuntime
from .security import ProductSecurityBoundary, validate_envelope, validate_identifier
from .service import CopilotServiceError
from .workspace import WorkspaceError

CREATE_FIELDS = frozenset({"run_id"})
SELECTION_FIELDS = frozenset(
    {
        "expected_revision",
        "selected_day",
        "selected_stop_id",
        "selected_segment_id",
        "selected_candidate_id",
        "selected_alternative_id",
    }
)
OPERATION_FIELDS = frozenset({"expected_revision", "type", "target", "parameters", "source", "evidence_refs"})
REVISION_FIELDS = frozenset({"expected_revision"})
MESSAGE_FIELDS = frozenset({"expected_revision", "client_message_id", "message"})
DELETE_ALL_CONVERSATIONS_FIELDS = frozenset({"expected_revision", "confirmation"})
PERMISSION_FIELDS = frozenset({"expected_revision", "permission", "decision", "proposal_id", "scope", "expires_at"})
SNAP_PREVIEW_FIELDS = frozenset(
    {
        "expected_revision",
        "longitude",
        "latitude",
        "operation_intent",
        "selected_day",
        "selected_route_segment_id",
        "travel_mode",
        "target_waypoint_id",
    }
)
SNAP_CONFIRM_FIELDS = frozenset(
    {"expected_revision", "name", "role", "duration", "warning_acknowledged"}
)
SNAP_CONFIRM_REQUIRED_FIELDS = frozenset({"expected_revision", "warning_acknowledged"})
SNAP_PREVIEW_REQUIRED_FIELDS = frozenset(
    {
        "expected_revision",
        "longitude",
        "latitude",
        "operation_intent",
        "selected_day",
        "travel_mode",
    }
)
SNAP_OPERATION_INTENTS = frozenset(
    {
        "explore_only",
        "add_custom_waypoint",
        "add_route_waypoint",
        "replace_stop_near_location",
        "relocate_custom_waypoint",
    }
)
PRODUCT_ID = RUNTIME_PRODUCT_ID


def create_product_app(
    *,
    repository_root: Path | None = None,
    registry_path: Path | None = None,
    state_root: Path | None = None,
    enable_legacy: bool = False,
    application_host: str = "127.0.0.1",
    application_port: int = 8127,
    additional_allowed_authorities: tuple[str, ...] = (),
) -> FastAPI:
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    registry_file = registry_path or root / "configs" / "product_app_registry.json"
    local_state = state_root or root / ".product_app_state"
    config = ProductRuntimeConfig.from_environment(
        repository_root=root,
        registry_path=registry_file,
        state_root=local_state,
        application_host=application_host,
        application_port=application_port,
        enable_legacy=enable_legacy,
        additional_allowed_authorities=additional_allowed_authorities,
    )
    runtime = ProductRuntime(config)
    runtime.initialize()
    security = ProductSecurityBoundary(config)
    static_dir = Path(__file__).resolve().parent / "static"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.initialize()
        yield

    app = FastAPI(
        title="Itinerary Repair Copilot",
        version=PRODUCT_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.product_runtime = runtime
    app.state.product_service = runtime.service

    @app.middleware("http")
    async def product_security(request: Request, call_next):
        try:
            security.validate_host(request)
            security.validate_origin(request)
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            return security.apply_headers(request, response)
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - sanitize unexpected application failures
            response = JSONResponse(
                {"detail": "internal_server_error"},
                status_code=500,
            )
        return security.apply_headers(request, response)

    @app.exception_handler(WorkspaceError)
    async def workspace_error(_: Request, exc: WorkspaceError):
        return JSONResponse({"detail": exc.code}, status_code=exc.status_code)

    @app.exception_handler(RegistryError)
    async def registry_error(_: Request, exc: RegistryError):
        return JSONResponse({"detail": exc.code}, status_code=404)

    @app.exception_handler(ConversationError)
    async def conversation_error(_: Request, exc: ConversationError):
        return JSONResponse(
            {"detail": exc.code},
            status_code=_conversation_error_status(exc.code),
        )

    @app.exception_handler(CopilotServiceError)
    async def copilot_service_error(_: Request, exc: CopilotServiceError):
        return JSONResponse({"detail": exc.code}, status_code=exc.status_code)

    @app.exception_handler(POICandidateDiscoveryError)
    async def poi_candidate_discovery_error(
        _: Request, exc: POICandidateDiscoveryError
    ):
        return JSONResponse({"detail": exc.code}, status_code=exc.status_code)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return runtime.health().as_dict()

    @app.get("/api/map/config")
    def map_config() -> dict[str, Any]:
        return runtime.map_configuration().as_dict()

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        service = runtime.require_service()
        return {"runs": [record.as_dict() for record in service.registry.all()]}

    @app.get("/api/runs/{run_id}/workspace")
    def run_workspace(run_id: str) -> dict[str, Any]:
        validate_identifier(run_id)
        return runtime.require_service().workspace_view(run_id)

    @app.post("/api/sessions")
    async def create_session(request: Request) -> dict[str, Any]:
        payload = validate_envelope(await security.read_bounded_json(request), allowed=CREATE_FIELDS)
        service = runtime.require_service()
        requested_run = payload.get("run_id")
        run_id = validate_identifier(requested_run) if requested_run is not None else service.registry.default.run_id
        assert run_id is not None
        session, mutation_token, _ = service.create_session(run_id)
        return {
            "session": session.public_dict(),
            "mutation_token": mutation_token,
            "workspace": service.workspace_view(run_id),
            "registry": service.registry.get(run_id).as_dict(),
        }

    @app.get("/api/sessions/{session_id}")
    def restore_session(
        session_id: str,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _validate_session_id(session_id)
        service = runtime.require_service()
        session = service.workspace.authenticate_readonly(
            session_id,
            x_session_token,
        )
        service.validate_session_draft(session)
        service.validate_cached_role_proposal(session, x_session_token)
        session = service.workspace.authenticate(session_id, x_session_token)
        return {
            "session": session.public_dict(),
            "workspace": service.workspace_view(session.run_id),
            "registry": service.registry.get(session.run_id).as_dict(),
        }

    @app.post("/api/sessions/{session_id}/selection")
    async def selection(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=SELECTION_FIELDS,
            required=frozenset({"expected_revision"}),
        )
        _validate_revision(payload)
        _validate_selection(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        day = payload.get("selected_day")
        _, view = service.load(session.run_id)
        if day is not None and not 1 <= day <= view["trip"]["day_count"]:
            raise WorkspaceError("invalid_selected_day")
        stop_id = payload.get("selected_stop_id")
        if stop_id and stop_id not in service.valid_stops(session.run_id):
            raise WorkspaceError("invalid_selected_stop")
        candidate_id = payload.get("selected_candidate_id")
        if candidate_id:
            capabilities = service.workspace_view(session.run_id)["draft_capabilities"]
            valid_candidates = {
                str(row.get("candidate_id") or "")
                for row in capabilities.get("candidate_choices") or ()
                if isinstance(row, dict) and row.get("candidate_id")
            }
            if candidate_id not in valid_candidates:
                raise WorkspaceError("invalid_selected_candidate", 422)
        segment_id = payload.get("selected_segment_id")
        if segment_id and segment_id not in service.valid_segments(session.run_id):
            raise WorkspaceError("invalid_selected_segment")
        alternative_id = payload.get("selected_alternative_id")
        if alternative_id and alternative_id not in service.valid_alternatives(session.run_id):
            raise WorkspaceError("invalid_selected_alternative")
        service.workspace.select(session, payload)
        return {"session": session.public_dict()}

    @app.post("/api/sessions/{session_id}/map/snap-preview")
    async def snap_preview(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Return a road-snap classification without appending a draft operation."""

        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=SNAP_PREVIEW_FIELDS,
            required=SNAP_PREVIEW_REQUIRED_FIELDS,
        )
        _validate_revision(payload)
        _validate_snap_preview(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        service.workspace.expect_revision(session, payload["expected_revision"])
        _, view = service.load(session.run_id)
        selected_day = payload["selected_day"]
        if not 1 <= selected_day <= view["trip"]["day_count"]:
            raise HTTPException(status_code=422, detail="request_validation_failed")
        segment_id = payload.get("selected_route_segment_id")
        predecessor = None
        successor = None
        target_waypoint_id = payload.get("target_waypoint_id")
        if payload["operation_intent"] == "relocate_custom_waypoint":
            existing = service.resolve_custom_waypoint(session, target_waypoint_id)
            insertion = existing["insertion"]
            if selected_day != existing["day"]:
                raise WorkspaceError("waypoint_insertion_context_mismatch", 409)
            if segment_id is not None and segment_id != insertion["route_leg_id"]:
                raise WorkspaceError("waypoint_insertion_context_mismatch", 409)
            segment_id = insertion["route_leg_id"]
        if segment_id is not None:
            predecessor, successor = service.resolve_validated_route_context(
                run_id=session.run_id,
                accepted_plan_id=session.accepted_plan_id,
                route_leg_id=segment_id,
                selected_day=selected_day,
            )
        try:
            if payload["operation_intent"] == "relocate_custom_waypoint":
                preview_entity_id = str(target_waypoint_id)
            elif payload["operation_intent"] in {"add_custom_waypoint", "add_route_waypoint"}:
                preview_entity_id = f"waypoint_{uuid4().hex}"
            else:
                preview_entity_id = f"map_preview_{uuid4().hex}"
            preview = MapInteractionService(runtime.require_routing()).preview(
                entity_id=preview_entity_id,
                raw_coordinate=(payload["latitude"], payload["longitude"]),
                operation_intent=payload["operation_intent"],
                predecessor=predecessor,
                successor=successor,
                travel_mode=payload["travel_mode"],
            )
        except MapInteractionError as exc:
            raise HTTPException(status_code=503, detail=exc.code) from None
        service.store_map_snap_preview(
            session=session,
            preview=preview,
            selected_day=selected_day,
            route_leg_id=segment_id,
            predecessor_id=predecessor[0] if predecessor else None,
            successor_id=successor[0] if successor else None,
            travel_mode=payload["travel_mode"],
            target_waypoint_id=target_waypoint_id,
        )
        result = preview.as_dict()
        result.update(
            {
                "session_revision": session.revision,
                "selected_day": selected_day,
                "selected_route_segment_id": segment_id,
                "travel_mode": payload["travel_mode"],
                "persisted": False,
            }
        )
        return result

    @app.post("/api/sessions/{session_id}/map/snap-previews/{snap_preview_id}/confirm")
    async def confirm_snap_preview(
        session_id: str,
        snap_preview_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=SNAP_CONFIRM_FIELDS,
            required=SNAP_CONFIRM_REQUIRED_FIELDS,
        )
        _validate_revision(payload)
        _validate_snap_confirmation(payload)
        _validate_session_id(session_id)
        if re.fullmatch(r"snap_[0-9a-f]{32}", snap_preview_id) is None:
            raise HTTPException(status_code=422, detail="request_validation_failed")
        service = runtime.require_service()
        session = service.workspace.authenticate(session_id, x_session_token)
        operation = service.confirm_map_snap_preview(
            session=session,
            snap_preview_id=snap_preview_id,
            payload=payload,
        )
        return {
            "confirmed": True,
            "feedback_tier": "route_checked",
            "evaluated_repair": False,
            "operation": operation.as_dict(),
            "session": session.public_dict(),
        }

    @app.post("/api/sessions/{session_id}/draft/operations")
    async def add_operation(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=OPERATION_FIELDS,
            required=frozenset({"expected_revision", "type", "target", "parameters"}),
        )
        _validate_revision(payload)
        _validate_operation(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        _, view = service.load(session.run_id)
        capabilities = service.workspace_view(session.run_id)["draft_capabilities"]
        operation = service.workspace.add_operation(
            session,
            payload,
            valid_stop_ids=service.valid_stops(session.run_id),
            day_count=view["trip"]["day_count"],
            parent_stop_ids=set(capabilities.get("parent_targets") or ()),
            candidate_ids={
                str(row.get("candidate_id") or "")
                for row in capabilities.get("candidate_choices") or ()
                if isinstance(row, dict) and row.get("candidate_id")
            },
            candidate_replacements={
                str(row["candidate_id"]): str(row["replaces_stop_id"])
                for row in capabilities.get("candidate_choices") or ()
                if isinstance(row, dict) and row.get("candidate_id") and row.get("replaces_stop_id")
            },
            route_leg_ids=service.valid_segments_for_plan(session.run_id, session.accepted_plan_id),
            protected_stop_ids=service.protected_stop_ids(session.run_id),
            parent_day_by_stop=service.parent_day_by_stop(session.run_id),
            parent_order_by_day=service.parent_order_by_day(session.run_id),
            parent_duration_by_stop=service.parent_duration_by_stop(session.run_id),
            parent_time_window_by_stop=service.parent_time_window_by_stop(
                session.run_id
            ),
            parent_role_by_stop=service.parent_role_by_stop(session.run_id),
            role_constraints_by_stop=service.role_constraints_by_stop(
                session.run_id
            ),
        )
        return {"operation": operation.as_dict(), "session": session.public_dict()}

    @app.post("/api/sessions/{session_id}/draft/impact-preview")
    async def draft_impact_preview(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.draft_impact_preview(session, payload["expected_revision"])

    @app.post("/api/sessions/{session_id}/draft/undo")
    async def undo(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        operation = service.workspace.undo(session, payload["expected_revision"])
        return {"undone": operation.as_dict(), "session": session.public_dict()}

    @app.post("/api/sessions/{session_id}/preview")
    async def preview(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        proposal = service.preview(
            session_id,
            payload["expected_revision"],
            mutation_token=x_session_token,
        )
        return {"proposal": proposal, "session": session.public_dict()}

    @app.post("/api/sessions/{session_id}/copilot/messages")
    async def copilot_message(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=MESSAGE_FIELDS,
            required=MESSAGE_FIELDS,
        )
        _validate_revision(payload)
        client_message_id = payload.get("client_message_id")
        if not isinstance(client_message_id, str) or not re.fullmatch(
            r"client_message_[0-9a-f]{32}", client_message_id
        ):
            raise HTTPException(status_code=422, detail="request_validation_failed")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 2_000:
            raise HTTPException(status_code=422, detail="request_validation_failed")
        service = runtime.require_service()
        _validate_session_id(session_id)
        return await service.interpret_copilot_message(
            session_id=session_id,
            mutation_token=x_session_token,
            expected_revision=payload["expected_revision"],
            client_message_id=client_message_id,
            message=message,
        )

    @app.get("/api/sessions/{session_id}/conversation")
    def conversation(
        session_id: str,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        record = service.read_conversation(session)
        return {"conversation": record.as_dict(), "session": session.public_dict()}

    @app.delete("/api/sessions/{session_id}/conversation")
    async def delete_conversation(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        deleted = service.delete_conversation(session, payload["expected_revision"])
        return {"deleted": deleted, "session": session.public_dict()}

    @app.delete("/api/conversations")
    async def delete_all_conversations(
        request: Request,
        x_session_id: str | None = Header(default=None),
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if not request.headers.get("origin"):
            raise HTTPException(status_code=403, detail="origin_not_allowed")
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=DELETE_ALL_CONVERSATIONS_FIELDS,
            required=DELETE_ALL_CONVERSATIONS_FIELDS,
        )
        _validate_revision(payload)
        if payload.get("confirmation") != "delete_all_conversations":
            raise HTTPException(status_code=422, detail="request_validation_failed")
        if not isinstance(x_session_id, str):
            raise HTTPException(status_code=403, detail="invalid_session_token")
        _validate_session_id(x_session_id)
        service = runtime.require_service()
        session = service.workspace.authenticate(x_session_id, x_session_token)
        deleted_count = service.delete_all_conversations(session, payload["expected_revision"])
        return {"deleted_count": deleted_count, "session": session.public_dict()}

    @app.post("/api/sessions/{session_id}/permissions")
    async def permission(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=PERMISSION_FIELDS,
            required=frozenset({"expected_revision", "permission", "decision", "proposal_id", "scope"}),
        )
        _validate_revision(payload)
        _validate_permission(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        decision = service.workspace.append_permission(session, payload)
        return {"decision": decision, "session": session.public_dict()}

    @app.get("/api/sessions/{session_id}/map/poi-candidates")
    def poi_candidates(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        query = _poi_candidate_query(request)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.poi_candidates(session, **query)

    @app.get("/api/sessions/{session_id}/alternatives")
    def alternatives(session_id: str, x_session_token: str | None = Header(default=None)) -> dict[str, Any]:
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        _, view = service.load(session.run_id)
        return {
            "baseline": {
                "id": "keep_original",
                "label": "Keep original",
                "ranking_eligible": False,
                "status": "accepted_parent",
            },
            "alternatives": view["alternatives"],
            "comparison": view["comparison"],
        }

    @app.post("/api/sessions/{session_id}/accept")
    async def accept(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        service.workspace.expect_revision(session, payload["expected_revision"])
        raise HTTPException(status_code=409, detail="acceptance_not_enabled_until_w5")

    @app.post("/api/sessions/{session_id}/keep-original")
    async def keep_original(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        service.workspace.expect_revision(session, payload["expected_revision"])
        raise HTTPException(status_code=409, detail="acceptance_not_enabled_until_w5")

    @app.get("/api/runs/{run_id}/evidence-bundle")
    def evidence_bundle(run_id: str, plan_id: str | None = None) -> StreamingResponse:
        validate_identifier(run_id)
        if plan_id is not None:
            validate_identifier(plan_id)
        service = runtime.require_service()
        bundle = service.evidence_bundle(run_id, plan_id)
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
            for category, paths in sorted(bundle.source_paths.items()):
                for relative_text in paths:
                    artifact = safe_run_relative_path(bundle.run_dir, relative_text)
                    archive.write(artifact, f"{category}/{Path(relative_text).name}")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": (f'attachment; filename="{run_id}-{plan_id or "baseline"}-evidence.zip"')},
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="product-static")

    if enable_legacy and runtime.registry is not None:
        legacy_dir = runtime.registry.run_dir(runtime.registry.default.run_id) / "dashboard_product"
        if legacy_dir.is_dir():
            app.mount("/legacy/product-v6", StaticFiles(directory=legacy_dir, html=True), name="legacy-product")

            @app.get("/legacy/folium")
            def legacy_folium() -> RedirectResponse:
                return RedirectResponse("/legacy/product-v6/", status_code=307)

            @app.get("/legacy/modular")
            def legacy_modular() -> RedirectResponse:
                return RedirectResponse("/legacy/product-v6/", status_code=307)

    @app.get("/")
    def root_redirect() -> RedirectResponse:
        return RedirectResponse("/app", status_code=307)

    @app.get("/app")
    @app.get("/app/{client_path:path}")
    def app_shell(client_path: str = "") -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


async def _revision_payload(request: Request, security: ProductSecurityBoundary) -> dict[str, Any]:
    payload = validate_envelope(
        await security.read_bounded_json(request), allowed=REVISION_FIELDS, required=REVISION_FIELDS
    )
    _validate_revision(payload)
    return payload


def _validate_revision(payload: dict[str, Any]) -> None:
    revision = payload.get("expected_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise HTTPException(status_code=422, detail="request_validation_failed")


def _poi_candidate_query(request: Request) -> dict[str, Any]:
    allowed = {
        "expected_revision",
        "day",
        "route_leg_id",
        "replacement_target_id",
        "role",
        "max_detour_minutes",
        "limit",
    }
    values: dict[str, str] = {}
    for key, value in request.query_params.multi_items():
        if key not in allowed:
            raise HTTPException(status_code=422, detail="poi_candidate_query_field_unknown")
        if key in values:
            raise HTTPException(status_code=422, detail="poi_candidate_query_field_duplicate")
        values[key] = value
    if not {"expected_revision", "day", "route_leg_id"}.issubset(values):
        raise HTTPException(status_code=422, detail="poi_candidate_query_field_missing")
    expected_revision = _query_integer(values["expected_revision"], "expected_revision_invalid")
    day = _query_integer(values["day"], "selected_day_invalid")
    route_leg_id = validate_identifier(values["route_leg_id"])
    replacement_target_id = values.get("replacement_target_id")
    if replacement_target_id is not None:
        replacement_target_id = validate_identifier(replacement_target_id)
    role = values.get("role")
    if role is not None:
        role = validate_identifier(role)
    maximum_detour = _query_float(
        values.get("max_detour_minutes"),
        default=DEFAULT_MAX_DETOUR_MINUTES,
        code="poi_candidate_max_detour_invalid",
    )
    limit = (
        _query_integer(values["limit"], "poi_candidate_limit_invalid")
        if "limit" in values
        else POI_CANDIDATE_DEFAULT_LIMIT
    )
    return {
        "expected_revision": expected_revision,
        "day": day,
        "route_leg_id": route_leg_id,
        "replacement_target_id": replacement_target_id,
        "role": role,
        "maximum_detour_minutes": maximum_detour,
        "limit": limit,
    }


def _query_integer(value: str, code: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]{0,9}", value):
        raise HTTPException(status_code=422, detail=code)
    return int(value)


def _query_float(value: str | None, *, default: float, code: str) -> float:
    if value is None:
        return default
    if len(value) > 32:
        raise HTTPException(status_code=422, detail=code)
    try:
        converted = float(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=code) from exc
    if not math.isfinite(converted):
        raise HTTPException(status_code=422, detail=code)
    return converted


def _validate_session_id(value: Any) -> str:
    session_id = validate_identifier(value)
    if re.fullmatch(r"session_[0-9a-f]{32}", session_id) is None:
        raise HTTPException(status_code=422, detail="invalid_session_id")
    return session_id


def _validate_selection(payload: dict[str, Any]) -> None:
    day = payload.get("selected_day")
    if day is not None and (isinstance(day, bool) or not isinstance(day, int)):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    for field in SELECTION_FIELDS - {"expected_revision", "selected_day"}:
        if field in payload and payload[field] is not None:
            validate_identifier(payload[field])


def _validate_operation(payload: dict[str, Any]) -> None:
    validate_identifier(payload.get("type"))
    validate_identifier(payload.get("target"))
    if not isinstance(payload.get("parameters"), dict):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    _validate_finite_json(payload["parameters"])
    source = payload.get("source", "map")
    if not isinstance(source, str) or not source.strip() or len(source) > 128:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    evidence_refs = payload.get("evidence_refs", [])
    if (
        not isinstance(evidence_refs, list)
        or len(evidence_refs) > 16
        or any(not isinstance(item, str) or not item.strip() or len(item) > 256 for item in evidence_refs)
    ):
        raise HTTPException(status_code=422, detail="request_validation_failed")


def _validate_snap_preview(payload: dict[str, Any]) -> None:
    for field, lower, upper in (
        ("latitude", -90.0, 90.0),
        ("longitude", -180.0, 180.0),
    ):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            raise HTTPException(status_code=422, detail="request_validation_failed")
    intent = payload.get("operation_intent")
    if intent not in SNAP_OPERATION_INTENTS:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    day = payload.get("selected_day")
    if isinstance(day, bool) or not isinstance(day, int):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    segment_id = payload.get("selected_route_segment_id")
    if segment_id is not None:
        validate_identifier(segment_id)
    target_waypoint_id = payload.get("target_waypoint_id")
    if intent == "relocate_custom_waypoint":
        if target_waypoint_id is None:
            raise HTTPException(status_code=422, detail="request_validation_failed")
        target = validate_identifier(target_waypoint_id)
        if re.fullmatch(r"waypoint_[0-9a-f]{32}", target) is None:
            raise HTTPException(status_code=422, detail="request_validation_failed")
    elif target_waypoint_id is not None:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    if payload.get("travel_mode") != "driving":
        raise HTTPException(status_code=422, detail="request_validation_failed")


def _validate_snap_confirmation(payload: dict[str, Any]) -> None:
    acknowledgement = payload.get("warning_acknowledged")
    if not isinstance(acknowledgement, bool):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    name = payload.get("name")
    if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    role = payload.get("role")
    if role is not None and (
        not isinstance(role, str)
        or role
        not in {"attraction", "activity", "meal", "rest_stop", "scenic_stop", "route_waypoint"}
    ):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    duration = payload.get("duration")
    if duration is not None:
        if not isinstance(duration, dict):
            raise HTTPException(status_code=422, detail="request_validation_failed")
        _validate_finite_json(duration)


def _validate_finite_json(value: Any, *, depth: int = 0) -> None:
    """Reject non-finite or excessively nested operation parameters."""

    if depth > 8:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    if isinstance(value, float) and not math.isfinite(value):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item, depth=depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_finite_json(item, depth=depth + 1)
        return
    raise HTTPException(status_code=422, detail="request_validation_failed")


def _validate_permission(payload: dict[str, Any]) -> None:
    for field in ("permission", "decision", "proposal_id", "scope"):
        validate_identifier(payload.get(field))
    if payload["decision"] not in {"granted", "denied"}:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    expires_at = payload.get("expires_at")
    if expires_at is not None and (not isinstance(expires_at, str) or len(expires_at) > 64):
        raise HTTPException(status_code=422, detail="request_validation_failed")


def _conversation_error_status(code: str) -> int:
    if code == "unknown_conversation":
        return 404
    if code in {"conversation_session_mismatch", "conversation_not_owned"}:
        return 403
    if code in {"message_id_conflict", "conversation_invalid", "conversation_session_conflict"}:
        return 409
    if code == "conversation_capacity_reached":
        return 429
    if code.startswith("invalid_") or code in {"conversation_turn_limit_reached"}:
        return 422
    return 503
