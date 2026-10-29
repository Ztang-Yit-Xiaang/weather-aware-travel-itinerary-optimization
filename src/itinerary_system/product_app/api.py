"""FastAPI boundary for the loopback-only Itinerary Repair Copilot."""

from __future__ import annotations

import asyncio
import io
import json
import math
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore, Event
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..product_dashboard_models import safe_run_relative_path
from .acceptance import AcceptanceError
from .candidate_discovery import (
    DEFAULT_LIMIT as POI_CANDIDATE_DEFAULT_LIMIT,
)
from .candidate_discovery import (
    DEFAULT_MAX_DETOUR_MINUTES,
    POICandidateDiscoveryError,
)
from .config import ProductRuntimeConfig
from .conversations import ConversationError
from .copilot import CopilotAdapter
from .map_interactions import MapInteractionError, MapInteractionService
from .registry import RegistryError
from .runtime import PRODUCT_ID as RUNTIME_PRODUCT_ID
from .runtime import PRODUCT_VERSION, ProductRuntime
from .security import ProductSecurityBoundary, validate_envelope, validate_identifier
from .service import CopilotServiceError
from .workspace import WorkspaceError, WorkspaceStore

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
MUTATION_FIELDS = frozenset({"mutation_id"})
ACCEPTANCE_FIELDS = frozenset({"expected_revision", "confirmation", "idempotency_key"})
MESSAGE_FIELDS = frozenset({"expected_revision", "client_message_id", "message"})
COPILOT_STREAM_SCHEMA_VERSION = "copilot-stream-v1"
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
SNAP_CONFIRM_FIELDS = frozenset({"expected_revision", "name", "role", "duration", "warning_acknowledged"})
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
PREVIEW_MAX_WORKERS = 2
PREVIEW_TIMEOUT_SECONDS = 60.0


def create_product_app(
    *,
    repository_root: Path | None = None,
    registry_path: Path | None = None,
    state_root: Path | None = None,
    enable_legacy: bool = False,
    application_host: str = "127.0.0.1",
    application_port: int = 8127,
    additional_allowed_authorities: tuple[str, ...] = (),
    copilot: CopilotAdapter | None = None,
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
    runtime = ProductRuntime(config, copilot=copilot)
    runtime.initialize()
    security = ProductSecurityBoundary(config)
    static_dir = Path(__file__).resolve().parent / "static"
    # StaticFiles streams from disk after middleware checks. Freeze these small assets
    # at startup instead so a concurrent source edit cannot mix browser generations.
    static_assets = {
        path.relative_to(static_dir).as_posix(): path.read_bytes() for path in static_dir.rglob("*") if path.is_file()
    }
    static_assets["index.html"] = static_assets["index.html"].replace(
        b"__PRODUCT_BUILD_ID__", runtime.build_id.encode("ascii")
    )
    runtime.require_current_build()
    preview_executor = ThreadPoolExecutor(max_workers=PREVIEW_MAX_WORKERS, thread_name_prefix="itinerary-preview")
    preview_slots = BoundedSemaphore(PREVIEW_MAX_WORKERS)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.initialize()
        try:
            yield
        finally:
            preview_executor.shutdown(wait=False, cancel_futures=False)

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
            if request.url.path == "/app" or request.url.path.startswith(("/app/", "/static/", "/api/")):
                await asyncio.to_thread(runtime.require_current_build)
            security.validate_browser_build(request, runtime.build_id)
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

    @app.exception_handler(AcceptanceError)
    async def acceptance_error(_: Request, exc: AcceptanceError):
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
    async def poi_candidate_discovery_error(_: Request, exc: POICandidateDiscoveryError):
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
        session, mutation_token, workspace = service.create_session(run_id)
        return {
            "session": session.public_dict(),
            "mutation_token": mutation_token,
            "workspace": workspace,
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
            "workspace": service.workspace_view_for_session(session),
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
            allowed=SELECTION_FIELDS | MUTATION_FIELDS,
            required=frozenset({"expected_revision"}),
        )
        _validate_revision(payload)
        _validate_mutation_id(payload)
        _validate_selection(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate_readonly(session_id, x_session_token)
        replay = service.workspace.replay_mutation(session, "selection", payload) if "mutation_id" in payload else None
        if replay:
            return _mutation_response(service, session, replay, x_session_token)
        day = payload.get("selected_day")
        if day is not None and day not in service.parent_order_by_day(session.run_id):
            proposal = session.proposal or {}
            compiler = service.compiler_for_session(session)
            compiled = (
                compiler.compile(session.draft, accepted_plan_id=session.accepted_plan_id, read_only=True)
                if compiler
                and proposal.get("execution_mode") == "trip_replan_independent_evaluation"
                and proposal.get("state") == "eligible"
                else None
            )
            child = compiled.child_plan or {} if compiled else {}
            if not (
                compiled
                and compiled.state == "eligible"
                and child.get("plan_id") == proposal.get("child_plan_id")
                and child.get("content_hash") == proposal.get("child_plan_content_hash")
                and any(s.get("day") == day for s in child.get("selected_stops", ()))
            ):
                raise WorkspaceError("invalid_selected_day")
        stop_id = payload.get("selected_stop_id")
        if stop_id and stop_id not in service.valid_stops(session.run_id):
            proposal = session.proposal or {}
            preview_stop = False
            if (
                proposal.get("execution_mode") == "trip_replan_independent_evaluation"
                and proposal.get("state") == "eligible"
            ):
                compiler = service.compiler_for_session(session)
                compiled = (
                    compiler.compile(session.draft, accepted_plan_id=session.accepted_plan_id, read_only=True)
                    if compiler
                    else None
                )
                child = compiled.child_plan or {} if compiled else {}
                preview_stop = bool(
                    compiled
                    and compiled.state == "eligible"
                    and child.get("plan_id") == proposal.get("child_plan_id")
                    and child.get("content_hash") == proposal.get("child_plan_content_hash")
                    and any(
                        (row.get("poi_id") or row.get("stop_id")) == stop_id and (day is None or row.get("day") == day)
                        for row in child.get("selected_stops", ())
                    )
                )
            if not preview_stop:
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
        if (
            segment_id
            and segment_id not in service.valid_segments(session.run_id)
            and segment_id not in service.valid_segments_for_plan(session.run_id, session.accepted_plan_id)
        ):
            proposal = session.proposal or {}
            preview_segment = False
            if (
                proposal.get("execution_mode") == "trip_replan_independent_evaluation"
                and proposal.get("state") == "eligible"
            ):
                compiler = service.compiler_for_session(session)
                compiled = (
                    compiler.compile(session.draft, accepted_plan_id=session.accepted_plan_id, read_only=True)
                    if compiler
                    else None
                )
                child = (compiled.child_plan or {}) if compiled else {}
                if (
                    compiled
                    and compiled.state == "eligible"
                    and child.get("plan_id") == proposal.get("child_plan_id")
                    and child.get("content_hash") == proposal.get("child_plan_content_hash")
                ):
                    edges = {
                        (row["origin_id"], row["destination_id"])
                        for row in compiled.route_legs
                        if day is None or row.get("day") == day
                    }
                    preview_segment = any(
                        cell.get("route_leg_id") == segment_id
                        and cell.get("road_validated") is True
                        and cell.get("fallback_used") is False
                        and (cell.get("origin_id"), cell.get("destination_id")) in edges
                        for cell in compiled.source_bundle.route_matrix.get("cells", ())
                    )
            if not preview_segment:
                raise WorkspaceError("invalid_selected_segment")
        alternative_id = payload.get("selected_alternative_id")
        if alternative_id and alternative_id not in service.valid_alternatives(session.run_id):
            raise WorkspaceError("invalid_selected_alternative")
        service.select(session, payload, x_session_token)
        if "mutation_id" in payload:
            return _mutation_response(
                service, session, service.workspace.replay_mutation(session, "selection", payload), x_session_token
            )
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
        service.ensure_editable_baseline(session)
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
                session=session,
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
        service.ensure_editable_baseline(session)
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
        raw = await security.read_bounded_json(request)
        batch = isinstance(raw, dict) and "operations" in raw
        payload = validate_envelope(
            raw,
            allowed=(frozenset({"expected_revision", "operations"}) if batch else OPERATION_FIELDS) | MUTATION_FIELDS,
            required=frozenset({"expected_revision", "operations"})
            if batch
            else frozenset({"expected_revision", "type", "target", "parameters"}),
        )
        _validate_revision(payload)
        _validate_mutation_id(payload)
        rows = payload["operations"] if batch else [payload]
        if not isinstance(rows, list) or not 1 <= len(rows) <= WorkspaceStore.MAX_DRAFT_OPERATIONS:
            raise WorkspaceError("invalid_draft_operations", 422)
        for row in rows:
            if batch:
                validate_envelope(
                    row,
                    allowed=OPERATION_FIELDS - {"expected_revision"},
                    required=frozenset({"type", "target", "parameters"}),
                )
            _validate_operation(row)

        def blocking_work():
            service = runtime.require_service()
            _validate_session_id(session_id)
            authenticate = (
                service.workspace.authenticate_readonly if "mutation_id" in payload else service.workspace.authenticate
            )
            session = authenticate(session_id, x_session_token)
            replay = (
                service.workspace.replay_mutation(session, "operations", payload) if "mutation_id" in payload else None
            )
            if replay:
                return _mutation_response(service, session, replay, x_session_token)
            service.ensure_editable_baseline(session)
            _, view = service.load(session.run_id)
            capabilities = service.workspace_view(session.run_id)["draft_capabilities"]
            operation = service.workspace.add_operation(
                session,
                {**rows[0], "expected_revision": payload["expected_revision"]},
                additional_operations=tuple(rows[1:]),
                mutation_request=payload if "mutation_id" in payload else None,
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
                parent_time_window_by_stop=service.parent_time_window_by_stop(session.run_id),
                parent_role_by_stop=service.parent_role_by_stop(session.run_id),
                role_constraints_by_stop=service.role_constraints_by_stop(session.run_id),
            )
            if "mutation_id" in payload:
                return _mutation_response(
                    service, session, service.workspace.replay_mutation(session, "operations", payload), x_session_token
                )
            return (
                {
                    "operations": [item.as_dict() for item in session.draft[-len(rows) :]],
                    "session": session.public_dict(),
                }
                if batch
                else {"operation": operation.as_dict(), "session": session.public_dict()}
            )

        return await asyncio.to_thread(blocking_work)

    @app.post("/api/sessions/{session_id}/draft/impact-preview")
    async def draft_impact_preview(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)

        def blocking_work():
            service = runtime.require_service()
            _validate_session_id(session_id)
            session = service.workspace.authenticate(session_id, x_session_token)
            return service.draft_impact_preview(session, payload["expected_revision"])

        return await asyncio.to_thread(blocking_work)

    @app.post("/api/sessions/{session_id}/draft/undo")
    async def undo(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request),
            allowed=REVISION_FIELDS | MUTATION_FIELDS,
            required=REVISION_FIELDS,
        )
        _validate_revision(payload)
        _validate_mutation_id(payload)
        service = runtime.require_service()
        _validate_session_id(session_id)
        authenticate = (
            service.workspace.authenticate_readonly if "mutation_id" in payload else service.workspace.authenticate
        )
        session = authenticate(session_id, x_session_token)
        replay = service.workspace.replay_mutation(session, "undo", payload) if "mutation_id" in payload else None
        if replay:
            return _mutation_response(service, session, replay, x_session_token)
        operation = service.workspace.undo(
            session, payload["expected_revision"], mutation_id=payload.get("mutation_id")
        )
        if "mutation_id" in payload:
            return _mutation_response(
                service, session, service.workspace.replay_mutation(session, "undo", payload), x_session_token
            )
        return {"undone": operation.as_dict(), "session": session.public_dict()}

    @app.get("/api/sessions/{session_id}/mutations/{mutation_id}")
    async def mutation_status(
        session_id: str, mutation_id: str, x_session_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _validate_session_id(session_id)
        WorkspaceStore.validate_mutation_id(mutation_id)

        def blocking_work():
            service = runtime.require_service()
            session = service.workspace.authenticate_readonly(session_id, x_session_token)
            receipt = next((row for row in session.mutation_receipts if row["mutation_id"] == mutation_id), None)
            # Unseen means no receipt at this read; an earlier request may still be in flight.
            return {
                "schema_version": "product-mutation-status-v1",
                "state": "applied" if receipt else "unseen",
                "mutation_id": mutation_id,
                "mutation_receipt": {key: value for key, value in receipt.items() if key != "result"}
                if receipt
                else None,
                "result": receipt["result"] if receipt else None,
            }

        return await asyncio.to_thread(blocking_work)

    @app.post("/api/sessions/{session_id}/mutations/{mutation_id}/cancel")
    async def cancel_pending_mutation(
        session_id: str,
        mutation_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = validate_envelope(
            await security.read_bounded_json(request), allowed={"action", "request"}, required={"action", "request"}
        )
        action, original = payload["action"], payload["request"]
        if (
            not isinstance(action, str)
            or action not in {"selection", "operations", "undo"}
            or not isinstance(original, dict)
            or original.get("mutation_id") != mutation_id
        ):
            raise HTTPException(status_code=422, detail="request_validation_failed")
        _validate_session_id(session_id)
        WorkspaceStore.validate_mutation_id(mutation_id)
        _validate_revision(original)
        service = runtime.require_service()
        session = service.workspace.authenticate_readonly(session_id, x_session_token)
        receipt = service.resolve_pending_mutation(session, action, original, x_session_token)
        return {
            "schema_version": "product-mutation-resolution-v1",
            "state": "applied" if receipt else "not_applied",
            "mutation_id": mutation_id,
            "action": action,
            "expected_revision": original["expected_revision"],
            "resolved_revision": session.revision,
            "mutation_receipt": {key: value for key, value in receipt.items() if key != "result"} if receipt else None,
            "result": receipt["result"] if receipt else None,
        }

    @app.post("/api/sessions/{session_id}/preview")
    async def preview(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _revision_payload(request, security)
        _validate_session_id(session_id)
        service = await asyncio.to_thread(runtime.require_service)

        def authenticated_context():
            # Readiness, authentication and accepted-artifact validation perform
            # blocking file/network I/O, even when the optimizer itself is async.
            session = service.workspace.authenticate(session_id, x_session_token)
            service.ensure_editable_baseline(session)
            with service.workspace._lock:
                return session, session.revision

        _, source_revision = await asyncio.to_thread(authenticated_context)
        if not preview_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="preview_capacity_reached", headers={"Retry-After": "1"})

        def submit_preview():
            try:
                work = preview_executor.submit(
                    service.preview,
                    session_id,
                    payload["expected_revision"],
                    mutation_token=x_session_token,
                    publish=False,
                )
            except BaseException:
                preview_slots.release()
                raise
            # Register release before exposing the work to the awaiting request.
            # A cancelled request cannot stop pending submission or a running worker.
            work.add_done_callback(lambda _: preview_slots.release())
            return work

        submission = asyncio.create_task(asyncio.to_thread(submit_preview))
        submission.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        work = await asyncio.shield(submission)
        result = asyncio.wrap_future(work)
        result.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            proposal = await asyncio.wait_for(asyncio.shield(result), timeout=PREVIEW_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="preview_execution_timeout") from None
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="preview_client_disconnected")
        session, _ = await asyncio.to_thread(authenticated_context)
        # A disconnect/cancellation during post-completion I/O must not publish.
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="preview_client_disconnected")
        abandoned = Event()

        def publish_result():
            # The revision guard, cached-result decision and durable mutation are
            # one in-process transaction. set_proposal also rechecks the persisted
            # revision under its file lock. Once this commit begins, it is atomic.
            with service.workspace._lock:
                if abandoned.is_set():
                    raise HTTPException(status_code=499, detail="preview_client_disconnected")
                service.workspace.expect_revision(session, source_revision)
                if session.proposal != proposal:
                    service.workspace.set_proposal(session, proposal, payload["expected_revision"])
                return {"proposal": proposal, "session": session.public_dict()}

        try:
            return await asyncio.to_thread(publish_result)
        except asyncio.CancelledError:
            abandoned.set()
            raise

    @app.post("/api/sessions/{session_id}/copilot/messages")
    async def copilot_message(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> Any:
        payload = await _copilot_message_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        if not _accepts_event_stream(request):
            return await service.interpret_copilot_message(
                session_id=session_id,
                mutation_token=x_session_token,
                expected_revision=payload["expected_revision"],
                client_message_id=payload["client_message_id"],
                message=payload["message"],
            )

        request_id = f"copilot_request_{uuid4().hex}"

        async def events():
            sequence = 1
            yield _copilot_stream_event(
                {
                    "schema_version": COPILOT_STREAM_SCHEMA_VERSION,
                    "type": "message.accepted",
                    "sequence": sequence,
                    "request_id": request_id,
                    "client_message_id": payload["client_message_id"],
                }
            )
            sequence += 1
            yield _copilot_stream_event(
                {
                    "schema_version": COPILOT_STREAM_SCHEMA_VERSION,
                    "type": "assistant.status",
                    "sequence": sequence,
                    "request_id": request_id,
                    "client_message_id": payload["client_message_id"],
                    "state": "interpreting",
                    "message": "Reading the current itinerary and validating a typed suggestion.",
                }
            )
            await asyncio.sleep(0)
            if await request.is_disconnected():
                return
            try:
                result = await service.interpret_copilot_message(
                    session_id=session_id,
                    mutation_token=x_session_token,
                    expected_revision=payload["expected_revision"],
                    client_message_id=payload["client_message_id"],
                    message=payload["message"],
                )
            except asyncio.CancelledError:
                raise
            except CopilotServiceError as exc:
                yield _copilot_stream_error(
                    request_id,
                    payload["client_message_id"],
                    sequence + 1,
                    exc.code,
                    exc.status_code,
                )
                return
            except WorkspaceError as exc:
                yield _copilot_stream_error(
                    request_id,
                    payload["client_message_id"],
                    sequence + 1,
                    exc.code,
                    exc.status_code,
                )
                return
            except ConversationError as exc:
                yield _copilot_stream_error(
                    request_id,
                    payload["client_message_id"],
                    sequence + 1,
                    exc.code,
                    _conversation_error_status(exc.code),
                )
                return
            except Exception:  # noqa: BLE001 - never stream provider or path details
                yield _copilot_stream_error(
                    request_id,
                    payload["client_message_id"],
                    sequence + 1,
                    "internal_server_error",
                    500,
                )
                return
            if await request.is_disconnected():
                return
            yield _copilot_stream_event(
                {
                    "schema_version": COPILOT_STREAM_SCHEMA_VERSION,
                    "type": "turn.completed",
                    "sequence": sequence + 1,
                    "request_id": request_id,
                    "client_message_id": payload["client_message_id"],
                    "payload": result,
                }
            )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Vary": "Accept",
                "X-Accel-Buffering": "no",
            },
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

    @app.get("/api/sessions/{session_id}/poi/search")
    def statewide_poi_search(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        query = _statewide_poi_query(request)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.statewide_poi_search(session, **query)

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

    @app.get("/api/sessions/{session_id}/map/interest-regions")
    def interest_regions(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> JSONResponse:
        query = _poi_candidate_query(request, interest_regions=True)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        # The service already returns JSON-native values. Avoid traversing every
        # GeoJSON coordinate through FastAPI's generic object encoder a second time.
        return JSONResponse(service.interest_regions(session, **query))

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
        payload = await _acceptance_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.accept_evaluated(
            session,
            expected_revision=payload["expected_revision"],
            confirmation=payload["confirmation"],
            idempotency_key=payload["idempotency_key"],
            mutation_token=x_session_token,
        )

    @app.post("/api/sessions/{session_id}/keep-original")
    async def keep_original(
        session_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _acceptance_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.keep_original_plan(
            session,
            expected_revision=payload["expected_revision"],
            confirmation=payload["confirmation"],
            idempotency_key=payload["idempotency_key"],
        )

    @app.get("/api/sessions/{session_id}/acceptance-decisions")
    def acceptance_decisions(
        session_id: str,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        service = runtime.require_service()
        _validate_session_id(session_id)
        session = service.workspace.authenticate_readonly(session_id, x_session_token)
        return service.acceptance_history(session)

    @app.post("/api/sessions/{session_id}/acceptance-decisions/{decision_id}/rollback")
    async def rollback_acceptance(
        session_id: str,
        decision_id: str,
        request: Request,
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = await _acceptance_payload(request, security)
        service = runtime.require_service()
        _validate_session_id(session_id)
        if re.fullmatch(r"decision_[0-9a-f]{32}", decision_id) is None:
            raise HTTPException(status_code=422, detail="request_validation_failed")
        session = service.workspace.authenticate(session_id, x_session_token)
        return service.rollback_acceptance(
            session,
            target_decision_id=decision_id,
            expected_revision=payload["expected_revision"],
            confirmation=payload["confirmation"],
            idempotency_key=payload["idempotency_key"],
        )

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

    @app.api_route("/static/{asset_path:path}", methods=["GET", "HEAD"])
    def static_asset(asset_path: str, request: Request) -> Response:
        body = static_assets.get(asset_path)
        if body is None:
            raise HTTPException(status_code=404, detail="Not Found")
        media_type = mimetypes.guess_type(asset_path)[0] or "application/octet-stream"
        return Response(
            body if request.method != "HEAD" else b"", media_type=media_type, headers={"Content-Length": str(len(body))}
        )

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
    def app_shell(client_path: str = "") -> HTMLResponse:
        return HTMLResponse(static_assets["index.html"])

    return app


async def _acceptance_payload(
    request: Request,
    security: ProductSecurityBoundary,
) -> dict[str, Any]:
    payload = validate_envelope(
        await security.read_bounded_json(request),
        allowed=ACCEPTANCE_FIELDS,
        required=ACCEPTANCE_FIELDS,
    )
    _validate_revision(payload)
    if not isinstance(payload.get("confirmation"), str):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    key = payload.get("idempotency_key")
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{16,128}", key):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    return payload


async def _copilot_message_payload(
    request: Request,
    security: ProductSecurityBoundary,
) -> dict[str, Any]:
    payload = validate_envelope(
        await security.read_bounded_json(request),
        allowed=MESSAGE_FIELDS,
        required=MESSAGE_FIELDS,
    )
    _validate_revision(payload)
    client_message_id = payload.get("client_message_id")
    if not isinstance(client_message_id, str) or not re.fullmatch(r"client_message_[0-9a-f]{32}", client_message_id):
        raise HTTPException(status_code=422, detail="request_validation_failed")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 2_000:
        raise HTTPException(status_code=422, detail="request_validation_failed")
    return payload


def _accepts_event_stream(request: Request) -> bool:
    accepted = request.headers.get("accept", "")
    return any(media_type.split(";", 1)[0].strip().lower() == "text/event-stream" for media_type in accepted.split(","))


def _copilot_stream_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _copilot_stream_error(
    request_id: str,
    client_message_id: str,
    sequence: int,
    code: str,
    status_code: int,
) -> str:
    return _copilot_stream_event(
        {
            "schema_version": COPILOT_STREAM_SCHEMA_VERSION,
            "type": "turn.failed",
            "sequence": sequence,
            "request_id": request_id,
            "client_message_id": client_message_id,
            "code": code,
            "status": status_code,
            "retryable": status_code in {408, 429, 502, 503, 504}
            and code
            not in {
                "openai_timeout",
                "openai_provider_unavailable",
                "copilot_provider_unavailable",
                "copilot_outcome_unknown",
                "copilot_dispatch_result_write_failed",
            },
        }
    )


def _validate_mutation_id(payload: dict[str, Any]) -> None:
    if "mutation_id" in payload:
        WorkspaceStore.validate_mutation_id(payload["mutation_id"])


def _mutation_response(
    service: Any, session: Any, receipt: dict[str, Any], mutation_token: str | None
) -> dict[str, Any]:
    service.validate_session_draft(session)
    service.validate_cached_role_proposal(session, mutation_token)
    return {
        **receipt["result"],
        "session": session.public_dict(),
        "mutation_receipt": {key: value for key, value in receipt.items() if key != "result"},
    }


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


def _statewide_poi_query(request: Request) -> dict[str, Any]:
    allowed = {"q", "limit"}
    values: dict[str, str] = {}
    for key, value in request.query_params.multi_items():
        if key not in allowed:
            raise HTTPException(status_code=422, detail="poi_search_query_field_unknown")
        if key in values:
            raise HTTPException(status_code=422, detail="poi_search_query_field_duplicate")
        values[key] = value
    query = values.get("q", "")
    if len(query) > 120:
        raise HTTPException(status_code=422, detail="poi_search_query_invalid")
    limit = _query_integer(values["limit"], "poi_search_limit_invalid") if "limit" in values else 10
    if not 1 <= limit <= 20:
        raise HTTPException(status_code=422, detail="poi_search_limit_invalid")
    return {"query": query, "limit": limit}


def _poi_candidate_query(request: Request, *, interest_regions: bool = False) -> dict[str, Any]:
    allowed = {
        "expected_revision",
        "day",
        "route_leg_id",
        "replacement_target_id",
        "role",
        "max_detour_minutes",
        "limit",
    }
    if interest_regions:
        allowed.remove("limit")
        allowed.add("nature_preference")
        allowed.add("plan_id")
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
    result = {
        "expected_revision": expected_revision,
        "day": day,
        "route_leg_id": route_leg_id,
        "replacement_target_id": replacement_target_id,
        "role": role,
        "maximum_detour_minutes": maximum_detour,
        "limit": limit,
    }
    if interest_regions:
        del result["limit"]
        preference = _query_float(values.get("nature_preference"), default=0.0, code="interest_preference_invalid")
        if not -1 <= preference <= 1:
            raise HTTPException(status_code=422, detail="interest_preference_invalid")
        result["nature_preference"] = preference
        if "plan_id" in values:
            result["plan_id"] = validate_identifier(values["plan_id"])
    return result


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
        or role not in {"attraction", "activity", "meal", "rest_stop", "scenic_stop", "route_waypoint"}
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
    if code.startswith("invalid_") or code in {
        "conversation_turn_limit_reached",
        "conversation_size_limit_reached",
    }:
        return 422
    return 503
