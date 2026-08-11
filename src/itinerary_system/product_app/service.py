"""Application boundary over immutable product-dashboard artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..product_dashboard_adapter import load_product_dashboard_source
from ..product_dashboard_view_models import build_product_dashboard_view_model
from ..research_artifacts import stable_content_hash
from .candidate_discovery import discover_poi_candidates
from .conversations import ConversationError, ConversationRepository
from .copilot import (
    CopilotAdapter,
    CopilotContractError,
    DeterministicCopilotAdapter,
    validate_copilot_interpretation,
)
from .copilot_prompt import PROMPT_VERSION, prompt_sha256
from .draft_compiler import CANONICAL_OPERATIONS, FrozenDraftCompiler
from .evaluated_stop_edits import build_role_constraint_evidence
from .geography_v2 import build_geographic_workspace_v2
from .interaction_contracts import (
    DRAFT_ONLY_TYPED_EDIT_OPERATIONS,
    EVALUATED_TYPED_EDIT_OPERATIONS,
    DraftOperationTypeV2,
    typed_edit_capabilities,
)
from .map_interactions import SnapPreviewV1
from .models import (
    ConversationV1,
    CopilotContextV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
    DraftOperationV1,
    ProductSession,
)
from .product_demo import load_product_demo_package
from .registry import ProductRunRegistry
from .workspace import WorkspaceError, WorkspaceStore

if TYPE_CHECKING:
    from .config import ProductRuntimeConfig


class CopilotServiceError(ValueError):
    """Stable, secret-free orchestration failure returned by the API."""

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass
class _StoredSnapPreview:
    preview: SnapPreviewV1
    session_id: str
    session_revision: int
    accepted_plan_id: str
    selected_day: int
    route_leg_id: str
    predecessor_id: str
    successor_id: str
    travel_mode: str
    operation_intent: str
    target_waypoint_id: str | None
    expires_at: datetime
    consumed: bool = False


class ProductService:
    MAX_STORED_SNAP_PREVIEWS = 128
    def __init__(
        self,
        registry: ProductRunRegistry,
        state_root: Path,
        *,
        copilot: CopilotAdapter | None = None,
        runtime_config: ProductRuntimeConfig | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = WorkspaceStore(state_root)
        self.conversations = ConversationRepository(state_root)
        self.copilot_unavailable_code: str | None = None
        self.copilot_mode = (
            runtime_config.copilot_adapter
            if runtime_config is not None
            else getattr(copilot, "provider_name", "deterministic")
        )
        self.copilot = copilot if copilot is not None else self._configured_copilot(runtime_config)
        self._copilot_locks: dict[str, asyncio.Lock] = {}
        self._bundles: dict[str, Any] = {}
        self._views: dict[str, dict[str, Any]] = {}
        self._geographies: dict[str, dict[str, Any]] = {}
        self._alternative_evidence: dict[str, dict[str, dict[str, Any]]] = {}
        self._evidence_bundles: dict[str, dict[str, Any]] = {}
        self._draft_compilers: dict[str, FrozenDraftCompiler | None] = {}
        self._map_preview_lock = RLock()
        self._map_snap_previews: dict[str, _StoredSnapPreview] = {}

    def load(self, run_id: str) -> tuple[Any, dict[str, Any]]:
        if run_id not in self._bundles:
            run_dir = self.registry.run_dir(run_id)
            if "multi_plan_product_demo" in self.registry.get(run_id).capabilities:
                package = load_product_demo_package(self.registry.repository_root, run_dir)
                bundle = package.primary_bundle
                geography = build_geographic_workspace_v2(
                    bundle,
                    additional_plans=package.additional_plans,
                    route_legs_by_plan=package.route_legs_by_plan,
                )
                view = build_product_dashboard_view_model(bundle)
                view["alternatives"] = [deepcopy(row) for row in package.alternatives]
                view["alternative_evidence"] = deepcopy(package.evidence)
                self._alternative_evidence[run_id] = deepcopy(package.evidence)
                self._evidence_bundles[run_id] = dict(package.evidence_bundles)
                self._draft_compilers[run_id] = FrozenDraftCompiler(
                    bundle.parent_plan,
                    self._evidence_bundles[run_id],
                    repository_root=self.registry.repository_root,
                )
            else:
                bundle = load_product_dashboard_source(run_dir)
                geography = build_geographic_workspace_v2(bundle)
                view = build_product_dashboard_view_model(bundle)
                self._alternative_evidence[run_id] = {}
                self._evidence_bundles[run_id] = {}
                self._draft_compilers[run_id] = None
            self._bundles[run_id] = bundle
            self._views[run_id] = view
            self._geographies[run_id] = geography
        return self._bundles[run_id], deepcopy(self._views[run_id])

    def evidence_bundle(self, run_id: str, plan_id: str | None) -> Any:
        bundle, _ = self.load(run_id)
        alternatives = self._evidence_bundles[run_id]
        if not alternatives:
            return bundle
        if not plan_id:
            raise WorkspaceError("evidence_plan_required")
        selected = alternatives.get(plan_id)
        if selected is None:
            raise WorkspaceError("evidence_plan_not_found")
        return selected

    def workspace_view(self, run_id: str) -> dict[str, Any]:
        _, view = self.load(run_id)
        view["geography"] = deepcopy(self._geographies[run_id])
        compiler = self._draft_compilers[run_id]
        view["draft_capabilities"] = (
            compiler.capabilities()
            if compiler
            else {
                "schema_version": "draft-capabilities-v1",
                "operations": {},
                "candidate_choices": [],
            }
        )
        view["map_edit_capabilities"] = {
            "schema_version": "map-edit-capabilities-v1",
            "operations": {
                operation: {
                    "enabled": True,
                    "feedback_tier": "route_checked",
                    "preview_executable": False,
                    "evaluated_repair": False,
                }
                for operation in (
                    "add_custom_waypoint",
                    "add_route_waypoint",
                    "relocate_custom_waypoint",
                )
            },
        }
        view["typed_edit_capabilities"] = typed_edit_capabilities()
        view["role_constraint_evidence"] = build_role_constraint_evidence(
            self._bundles[run_id].parent_plan
        )
        if self.copilot_mode == "deterministic":
            view["interaction"] = {
                "enabled": True,
                "provider": "deterministic",
                "state": "deterministic_demo",
                "message": "Deterministic demo. Requests stay local and no external provider is called.",
            }
        else:
            view["interaction"] = {
                "enabled": self.copilot is not None,
                "provider": "openai",
                "state": self.copilot_unavailable_code or "openai_configured",
                "message": (
                    "OpenAI Copilot sends the visible trip context, your message, and bounded recent conversation history to OpenAI."
                    if self.copilot is not None
                    else "OpenAI Copilot is unavailable in the current local configuration."
                ),
            }
        return view

    def create_session(self, run_id: str) -> tuple[Any, str, dict[str, Any]]:
        bundle, view = self.load(run_id)
        record = self.registry.get(run_id)
        session, mutation_token = self.workspace.create_session(
            run_id,
            str(bundle.parent_plan.get("plan_id") or ""),
            int(view["trip"]["selected_day"]),
            trip_id=record.trip_id,
        )
        self._conversation_for(session)
        return session, mutation_token, view

    def copilot_context(self, session: ProductSession) -> CopilotContextV1:
        _, view = self.load(session.run_id)
        capabilities = self.workspace_view(session.run_id)["draft_capabilities"]
        candidate_ids = tuple(
            sorted(
                str(row["candidate_id"])
                for row in capabilities.get("candidate_choices") or ()
                if isinstance(row, dict) and row.get("candidate_id")
            )
        )
        proposal = self._decision_safe_proposal(session.proposal)
        return CopilotContextV1(
            run_id=session.run_id,
            trip_id=session.trip_id,
            session_id=session.session_id,
            session_revision=session.revision,
            accepted_plan_id=session.accepted_plan_id,
            selected_day=session.selected_day,
            selected_stop_id=session.selected_stop_id,
            selected_segment_id=session.selected_segment_id,
            selected_candidate_id=session.selected_candidate_id,
            selected_alternative_id=session.selected_alternative_id,
            draft_operations=tuple(
                {
                    "type": operation.type,
                    "target": operation.target,
                    "parameters": deepcopy(operation.parameters),
                }
                for operation in session.draft
            ),
            evaluated_proposal=proposal,
            allowed_stop_ids=tuple(sorted(self.valid_stops(session.run_id))),
            allowed_candidate_ids=candidate_ids,
            allowed_days=tuple(range(1, int(view["trip"]["day_count"]) + 1)),
            allowed_segment_ids=tuple(sorted(self.valid_segments(session.run_id))),
            allowed_alternative_ids=tuple(sorted(self.valid_alternatives(session.run_id))),
        )

    def read_conversation(self, session: ProductSession) -> ConversationV1:
        return self._conversation_for(session)

    async def interpret_copilot_message(
        self,
        *,
        session_id: str,
        mutation_token: str | None,
        expected_revision: int,
        client_message_id: str,
        message: str,
    ) -> dict[str, Any]:
        lock = self._copilot_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self.workspace.authenticate(session_id, mutation_token)
            self.workspace.expect_revision(session, expected_revision)
            conversation = self._conversation_for(session)
            existing = self._existing_turn(conversation.turns, client_message_id, message)
            if existing is not None:
                return self._copilot_response(existing, conversation.revision, session)

            context = self.copilot_context(session)
            created_at = _utc_now()
            if self.copilot is None:
                code = self.copilot_unavailable_code or "copilot_provider_unavailable"
                return self._persist_failed_turn_and_raise(
                    session_id=session_id,
                    mutation_token=mutation_token,
                    expected_revision=expected_revision,
                    conversation_id=conversation.conversation_id,
                    client_message_id=client_message_id,
                    message=message,
                    created_at=created_at,
                    provider="openai",
                    model=None,
                    code=code,
                )

            try:
                interpretation = await self.copilot.interpret(
                    context=context,
                    history=conversation.turns,
                    message=message,
                )
                interpretation = validate_copilot_interpretation(
                    context=context,
                    interpretation=interpretation,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # provider/contract details are normalized below
                code = self._copilot_error_code(exc)
                return self._persist_failed_turn_and_raise(
                    session_id=session_id,
                    mutation_token=mutation_token,
                    expected_revision=expected_revision,
                    conversation_id=conversation.conversation_id,
                    client_message_id=client_message_id,
                    message=message,
                    created_at=created_at,
                    provider=self.copilot.provider_name,
                    model=getattr(self.copilot, "model", None),
                    code=code,
                )

            current = self.workspace.authenticate(session_id, mutation_token)
            self.workspace.expect_revision(current, expected_revision)
            state = "refused" if interpretation.state == "refused" else "completed"
            if interpretation.state == "failed":
                state = "failed"
            turn = self._turn(
                client_message_id=client_message_id,
                context_revision=expected_revision,
                provider=self.copilot.provider_name,
                model=getattr(self.copilot, "model", None),
                state=state,
                user_message=message,
                assistant_message=interpretation.assistant_message,
                interpretation=interpretation,
                error_code=None,
                created_at=created_at,
            )
            saved = self.conversations.append_turn(
                conversation.conversation_id,
                session_id,
                turn,
            )
            persisted = self._existing_turn(saved.turns, client_message_id, message)
            assert persisted is not None
            return self._copilot_response(persisted, saved.revision, current)

    def delete_conversation(
        self,
        session: ProductSession,
        expected_revision: int,
    ) -> bool:
        self.workspace.expect_revision(session, expected_revision)
        conversation = self._conversation_for(session)
        self.workspace.clear_conversation_id(session, expected_revision, conversation.conversation_id)
        try:
            return self.conversations.delete(conversation.conversation_id, session.session_id)
        except ConversationError:
            # Clearing the link first prevents an irreversible transcript loss if
            # session state changes concurrently. Restore that link when file
            # deletion itself fails so the bytes remain reachable and retryable.
            self.workspace.bind_conversation_id(session, conversation.conversation_id)
            raise

    def delete_all_conversations(self, session: ProductSession, expected_revision: int) -> int:
        self.workspace.expect_revision(session, expected_revision)
        return self.conversations.delete_all()

    def _configured_copilot(
        self,
        runtime_config: ProductRuntimeConfig | None,
    ) -> CopilotAdapter | None:
        if runtime_config is None or runtime_config.copilot_adapter == "deterministic":
            return DeterministicCopilotAdapter()
        if not runtime_config.openai_api_key:
            self.copilot_unavailable_code = "openai_not_configured"
            return None
        try:
            from .openai_copilot import OpenAICopilotAdapter

            return OpenAICopilotAdapter(
                model=runtime_config.openai_model,
                api_key=runtime_config.openai_api_key,
                timeout_seconds=runtime_config.openai_timeout_seconds,
                history_messages=runtime_config.openai_history_messages,
                history_characters=runtime_config.openai_history_characters,
            )
        except Exception:  # noqa: BLE001 - construction details remain server-only
            self.copilot_unavailable_code = "openai_transport_unavailable"
            return None

    def _conversation_for(self, session: ProductSession) -> ConversationV1:
        conversation = self.conversations.get_or_create(session)
        self.workspace.bind_conversation_id(session, conversation.conversation_id)
        return conversation

    @staticmethod
    def _existing_turn(
        turns: tuple[CopilotTurnV1, ...],
        client_message_id: str,
        message: str,
    ) -> CopilotTurnV1 | None:
        for turn in turns:
            if turn.client_message_id != client_message_id:
                continue
            if turn.user_message != message:
                raise ConversationError("message_id_conflict")
            return turn
        return None

    def _persist_failed_turn_and_raise(
        self,
        *,
        session_id: str,
        mutation_token: str | None,
        expected_revision: int,
        conversation_id: str,
        client_message_id: str,
        message: str,
        created_at: str,
        provider: str,
        model: str | None,
        code: str,
    ) -> Any:
        current = self.workspace.authenticate(session_id, mutation_token)
        self.workspace.expect_revision(current, expected_revision)
        turn = self._turn(
            client_message_id=client_message_id,
            context_revision=expected_revision,
            provider=provider,
            model=model,
            state="failed",
            user_message=message,
            assistant_message=self._safe_failure_message(code),
            interpretation=None,
            error_code=code,
            created_at=created_at,
        )
        self.conversations.append_turn(conversation_id, session_id, turn)
        raise CopilotServiceError(code, self._copilot_error_status(code))

    @staticmethod
    def _turn(
        *,
        client_message_id: str,
        context_revision: int,
        provider: str,
        model: str | None,
        state: str,
        user_message: str,
        assistant_message: str,
        interpretation: CopilotInterpretationV1 | None,
        error_code: str | None,
        created_at: str,
    ) -> CopilotTurnV1:
        return CopilotTurnV1(
            turn_id=f"copilot_turn_{uuid4().hex}",
            client_message_id=client_message_id,
            context_revision=context_revision,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=prompt_sha256(),
            state=state,
            user_message=user_message,
            assistant_message=assistant_message,
            interpretation=interpretation,
            error_code=error_code,
            created_at=created_at,
            completed_at=_utc_now(),
        )

    def _copilot_response(
        self,
        turn: CopilotTurnV1,
        conversation_revision: int,
        session: ProductSession,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "turn": turn.as_dict(),
            "conversation_revision": conversation_revision,
            "session": session.public_dict(),
        }
        interpretation = turn.interpretation
        if interpretation and any(intent.type == "review_registered_repair" for intent in interpretation.intents):
            response["advisory"] = self._registered_repair_advisory(session.run_id)
        return response

    def _registered_repair_advisory(self, run_id: str) -> dict[str, Any] | None:
        bundle, view = self.load(run_id)
        if not (bundle.child_plan and bundle.certificate and bundle.diff):
            return None
        return {
            "schema_version": "copilot-registered-repair-advisory-v1",
            "state": "available",
            "parent_plan_id": bundle.parent_plan.get("plan_id"),
            "child_plan_id": bundle.child_plan.get("plan_id"),
            "certificate_id": bundle.certificate.get("certificate_id"),
            "diff_id": bundle.diff.get("diff_id"),
            "repair": deepcopy(view.get("repair")),
            "automatic_activation": False,
        }

    @staticmethod
    def _decision_safe_proposal(proposal: dict[str, Any] | None) -> dict[str, Any] | None:
        if not proposal:
            return None
        allowed = {
            "schema_version",
            "state",
            "reason",
            "eligibility",
            "provenance",
            "message",
            "parent_plan_id",
            "child_plan_id",
            "certificate_id",
            "diff_id",
            "evaluation_status",
        }
        return {key: deepcopy(value) for key, value in proposal.items() if key in allowed}

    @staticmethod
    def _copilot_error_code(error: Exception) -> str:
        if isinstance(error, CopilotContractError):
            return error.code
        code = getattr(error, "code", None)
        if isinstance(code, str) and code in {
            "openai_not_configured",
            "openai_transport_unavailable",
            "openai_authentication_failed",
            "openai_timeout",
            "openai_rate_limited",
            "openai_quota_exhausted",
            "openai_provider_unavailable",
            "openai_invalid_structured_output",
            "copilot_interpretation_invalid",
        }:
            return code
        return "copilot_provider_unavailable"

    @staticmethod
    def _copilot_error_status(code: str) -> int:
        return {
            "openai_timeout": 504,
            "openai_rate_limited": 429,
            "openai_quota_exhausted": 429,
            "openai_invalid_structured_output": 502,
            "copilot_interpretation_invalid": 422,
        }.get(code, 503)

    @staticmethod
    def _safe_failure_message(code: str) -> str:
        return {
            "openai_not_configured": "OpenAI Copilot is not configured on this device.",
            "openai_transport_unavailable": "OpenAI Copilot is unavailable in this runtime.",
            "openai_authentication_failed": "OpenAI authentication failed. Check the local provider configuration.",
            "openai_timeout": "OpenAI Copilot did not respond before the local timeout.",
            "openai_rate_limited": "OpenAI Copilot is temporarily rate limited. Try again later.",
            "openai_quota_exhausted": "OpenAI API quota or credits are unavailable for this project.",
            "openai_invalid_structured_output": "OpenAI returned a response that could not be safely interpreted.",
            "copilot_interpretation_invalid": "The Copilot result referenced unsupported itinerary data.",
        }.get(code, "Copilot is temporarily unavailable. No itinerary state was changed.")

    def valid_stops(self, run_id: str) -> set[str]:
        bundle, view = self.load(run_id)
        visible = {str(stop["id"]) for day in view["timeline"] for stop in day["stops"] if stop.get("id")}
        artifact_stops = {
            str(feature.get("properties", {}).get("stop_id") or "")
            for plan in self._geographies[run_id].get("plans", [])
            for feature in plan.get("stops", {}).get("features", [])
        }
        return visible | {stop_id for stop_id in artifact_stops if stop_id}

    def valid_segments(self, run_id: str) -> set[str]:
        self.load(run_id)
        return {
            str(feature.get("properties", {}).get("route_leg_id") or "")
            for plan in self._geographies[run_id].get("plans", [])
            for feature in plan.get("validated_legs", {}).get("features", [])
            if feature.get("properties", {}).get("route_leg_id")
        }

    def valid_segments_for_plan(self, run_id: str, plan_id: str) -> set[str]:
        self.load(run_id)
        matches = [plan for plan in self._geographies[run_id].get("plans", []) if plan.get("plan_id") == plan_id]
        if len(matches) != 1:
            raise WorkspaceError("accepted_route_context_unavailable", 409)
        return {
            str(feature.get("properties", {}).get("route_leg_id") or "")
            for feature in matches[0].get("validated_legs", {}).get("features", [])
            if feature.get("properties", {}).get("route_leg_id")
        }

    def protected_stop_ids(self, run_id: str) -> set[str]:
        """Return artifact-owned hard commitments that direct editing cannot weaken."""

        bundle, _ = self.load(run_id)
        protected: set[str] = set()
        for row in bundle.parent_plan.get("owned_constraints") or ():
            if not isinstance(row, dict):
                continue
            strength = str(row.get("strength") or row.get("commitment_strength") or "").lower()
            target = str(row.get("target_stop_id") or row.get("target_id") or row.get("stop_id") or "")
            if target and strength in {"must_keep", "locked", "booked"}:
                protected.add(target)
        return protected

    def parent_day_by_stop(self, run_id: str) -> dict[str, int]:
        bundle, _ = self.load(run_id)
        result: dict[str, int] = {}
        for row in bundle.parent_plan.get("selected_stops") or ():
            if not isinstance(row, dict):
                continue
            stop_id = str(row.get("stop_id") or row.get("poi_id") or "")
            day = row.get("day")
            if stop_id and isinstance(day, int) and not isinstance(day, bool):
                result[stop_id] = day
        return result

    def parent_order_by_day(self, run_id: str) -> dict[int, tuple[str, ...]]:
        """Return the accepted parent order used to validate same-day edits."""

        bundle, _ = self.load(run_id)
        result: dict[int, tuple[str, ...]] = {}
        for row in bundle.parent_plan.get("ordered_days") or ():
            if not isinstance(row, dict):
                continue
            day = row.get("day")
            stop_ids = row.get("stop_ids")
            if (
                isinstance(day, int)
                and not isinstance(day, bool)
                and isinstance(stop_ids, list)
                and all(isinstance(stop_id, str) and stop_id for stop_id in stop_ids)
            ):
                result[day] = tuple(stop_ids)
        return result

    def parent_duration_by_stop(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Return only explicit parent duration data; evaluator defaults stay implicit."""

        bundle, _ = self.load(run_id)
        result: dict[str, dict[str, Any]] = {}
        for row in bundle.parent_plan.get("selected_stops") or ():
            if not isinstance(row, dict):
                continue
            stop_id = str(row.get("stop_id") or row.get("poi_id") or "")
            if not stop_id:
                continue
            scalar = next(
                (
                    row.get(key)
                    for key in (
                        "visit_duration_minutes",
                        "duration_minutes",
                        "service_minutes",
                        "dwell_minutes",
                    )
                    if row.get(key) is not None
                ),
                None,
            )
            result[stop_id] = {
                "duration_constraint": deepcopy(row.get("duration_constraint")),
                "visit_duration_minutes": scalar,
            }
        return result

    def parent_time_window_by_stop(
        self,
        run_id: str,
    ) -> dict[str, dict[str, Any] | None]:
        """Return only trip-owned required windows, never place opening hours."""

        bundle, _ = self.load(run_id)
        result: dict[str, dict[str, Any] | None] = {}
        for row in bundle.parent_plan.get("selected_stops") or ():
            if not isinstance(row, dict):
                continue
            stop_id = str(row.get("stop_id") or row.get("poi_id") or "")
            if stop_id:
                result[stop_id] = deepcopy(row.get("time_window_constraint"))
        return result

    def parent_role_by_stop(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Return trip-owned itinerary roles without inferring place categories."""

        bundle, _ = self.load(run_id)
        result: dict[str, dict[str, Any]] = {}
        for row in bundle.parent_plan.get("selected_stops") or ():
            if not isinstance(row, dict):
                continue
            stop_id = str(row.get("stop_id") or row.get("poi_id") or "")
            if stop_id:
                result[stop_id] = {
                    "itinerary_role": row.get("itinerary_role"),
                    "itinerary_role_source": row.get("itinerary_role_source"),
                }
        return result

    def role_constraints_by_stop(
        self,
        run_id: str,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return only active STOP/POI constraints explicitly scoped to role."""

        evidence = self.workspace_view(run_id)["role_constraint_evidence"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in evidence["constraints"]:
            target = row["target_stop_id"]
            grouped.setdefault(target, []).append(
                {
                    "constraint_id": row["constraint_id"],
                    "strength": row["strength"],
                    "value": row["required_role"],
                    "permission_semantics": row["permission_semantics"],
                }
            )
        return {target: tuple(rows) for target, rows in grouped.items()}

    def validate_session_draft(self, session: ProductSession) -> None:
        """Fail closed when a restored draft no longer matches its registered artifacts."""

        _, view = self.load(session.run_id)
        capabilities = self.workspace_view(session.run_id)["draft_capabilities"]
        candidate_rows = [
            row
            for row in capabilities.get("candidate_choices") or ()
            if isinstance(row, dict) and row.get("candidate_id")
        ]
        self.workspace.validate_draft_domain(
            session.draft,
            parent_targets=set(capabilities.get("parent_targets") or ()),
            candidate_ids={str(row["candidate_id"]) for row in candidate_rows},
            candidate_replacements={
                str(row["candidate_id"]): str(row["replaces_stop_id"])
                for row in candidate_rows
                if row.get("replaces_stop_id")
            },
            day_count=int(view["trip"]["day_count"]),
            route_leg_ids=self.valid_segments_for_plan(session.run_id, session.accepted_plan_id),
            protected_stop_ids=self.protected_stop_ids(session.run_id),
            parent_day_by_stop=self.parent_day_by_stop(session.run_id),
            parent_order_by_day=self.parent_order_by_day(session.run_id),
            parent_duration_by_stop=self.parent_duration_by_stop(session.run_id),
            parent_time_window_by_stop=self.parent_time_window_by_stop(session.run_id),
            parent_role_by_stop=self.parent_role_by_stop(session.run_id),
            role_constraints_by_stop=self.role_constraints_by_stop(session.run_id),
        )

    def validate_cached_role_proposal(
        self,
        session: ProductSession,
        mutation_token: str | None,
    ) -> None:
        """Recompile and compare a persisted role proposal without mutating it."""

        proposal = session.proposal or {}
        role_draft = bool(session.draft) and all(
            operation.type == DraftOperationTypeV2.SET_STOP_ROLE.value
            for operation in session.draft
        )
        if not proposal or not role_draft:
            return
        if (
            (proposal.get("role_impact") or {}).get("schema_version")
            != "evaluated-role-impact-v1"
        ):
            raise WorkspaceError("draft_role_proposal_invalid", 409)
        if role_draft:
            self.preview(
                session.session_id,
                session.revision,
                mutation_token=mutation_token,
            )

    def draft_impact_preview(self, session: ProductSession, expected_revision: Any) -> dict[str, Any]:
        """Describe persisted draft effects without evaluating or certifying them."""

        self.workspace.expect_revision(session, expected_revision)
        self.validate_session_draft(session)
        bundle, _ = self.load(session.run_id)
        compiler = self._draft_compilers[session.run_id]
        executable_legacy = set(CANONICAL_OPERATIONS) if compiler is not None else set()
        rows: list[dict[str, Any]] = []
        blocking_codes: list[str] = []
        changed_attribute_by_type = {
            DraftOperationTypeV2.SET_STOP_ROLE.value: ("itinerary_role",),
            DraftOperationTypeV2.SET_STOP_DAY.value: ("day",),
            DraftOperationTypeV2.SET_STOP_ORDER.value: ("day", "sequence_order"),
            DraftOperationTypeV2.SET_STOP_DURATION.value: ("duration",),
            DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value: ("time_window",),
            DraftOperationTypeV2.SET_STOP_COMMITMENT.value: ("commitment",),
            DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value: (),
            DraftOperationTypeV2.REPORT_ROUTE_ISSUE.value: ("user_route_report",),
            DraftOperationTypeV2.CHANGE_ROUTE_PREFERENCE.value: ("route_preference",),
        }
        for operation in session.draft:
            kind = operation.type
            if (
                kind == DraftOperationTypeV2.SET_STOP_ROLE.value
                and operation.parameters.get("role") not in EVALUATED_STOP_ROLES
            ):
                tier = "draft_only"
                executable = False
                codes = ["stop_role_evaluation_not_supported"]
            elif (
                kind == DraftOperationTypeV2.SET_STOP_DURATION.value
                and (operation.parameters.get("duration") or {}).get("mode") != "exact"
            ):
                tier = "draft_only"
                executable = False
                codes = ["duration_mode_evaluation_not_supported"]
            elif kind in EVALUATED_TYPED_EDIT_OPERATIONS or kind in executable_legacy:
                tier = "evaluated"
                executable = True
                codes: list[str] = []
            elif kind in DRAFT_ONLY_TYPED_EDIT_OPERATIONS or kind in self.workspace.CONFIRMED_MAP_OPERATIONS:
                tier = "draft_only"
                executable = False
                code = {
                    DraftOperationTypeV2.CHANGE_ROUTE_PREFERENCE.value: (
                        "route_preference_not_supported"
                    ),
                    DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value: (
                        "departure_window_evaluation_not_supported"
                    ),
                }.get(kind, "full_evaluation_not_supported")
                codes = [code]
            else:
                tier = "draft_only"
                executable = False
                codes = ["draft_operation_not_executable"]
            changed_attributes = changed_attribute_by_type.get(kind, ())
            if kind == DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value:
                changed_attributes = (str(operation.parameters.get("attribute") or ""),)
            rows.append(
                {
                    "operation_id": operation.operation_id,
                    "type": kind,
                    "target": operation.target,
                    "feedback_tier": tier,
                    "preview_executable": executable,
                    "evaluated_repair": executable,
                    "changed_attributes": list(changed_attributes),
                    "blocking_codes": codes,
                }
            )
            blocking_codes.extend(codes)
        unique_blocking = list(dict.fromkeys(blocking_codes))
        operation_count = len(rows)
        executable_count = sum(1 for row in rows if row["preview_executable"])
        direct_types = {
            DraftOperationTypeV2.SET_STOP_ORDER.value,
            DraftOperationTypeV2.SET_STOP_ROLE.value,
            DraftOperationTypeV2.SET_STOP_DURATION.value,
            DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
        }
        present_direct_types = {row["type"] for row in rows} & direct_types
        combination_unsupported = bool(present_direct_types) and (
            len(present_direct_types) != 1
            or any(row["type"] not in present_direct_types for row in rows)
        )
        if present_direct_types == {DraftOperationTypeV2.SET_STOP_DURATION.value}:
            duration_modes = {
                (operation.parameters.get("duration") or {}).get("mode")
                for operation in session.draft
            }
            combination_unsupported = combination_unsupported or (
                "exact" in duration_modes and len(duration_modes) > 1
            )
        if combination_unsupported:
            unique_blocking = ["draft_evaluated_operation_combination_unsupported"]
        unique_blocking = list(dict.fromkeys(unique_blocking))
        return {
            "schema_version": "product-draft-impact-preview-v1",
            "session_id": session.session_id,
            "session_revision": session.revision,
            "parent_plan_id": str(bundle.parent_plan.get("plan_id") or ""),
            "parent_plan_content_hash": str(bundle.parent_plan.get("content_hash") or ""),
            "certification_status": "not_certified",
            "is_certified": False,
            "operations": rows,
            "summary": {
                "operation_count": operation_count,
                "evaluated_executable_count": executable_count,
                "draft_only_count": operation_count - executable_count,
                "can_run_evaluated_preview": (
                    bool(operation_count)
                    and executable_count == operation_count
                    and not unique_blocking
                ),
                "blocking_codes": unique_blocking,
            },
        }

    def poi_candidates(
        self,
        session: ProductSession,
        *,
        expected_revision: int,
        day: int,
        route_leg_id: str,
        replacement_target_id: str | None,
        role: str | None,
        maximum_detour_minutes: float,
        limit: int,
    ) -> dict[str, Any]:
        """Discover candidates without mutating session or accepted-plan state."""

        self.workspace.expect_revision(session, expected_revision)
        bundle, view = self.load(session.run_id)
        day_count = int(view["trip"]["day_count"])
        if isinstance(day, bool) or not 1 <= day <= day_count:
            raise WorkspaceError("selected_day_invalid", 422)
        record = self.registry.get(session.run_id)
        return discover_poi_candidates(
            repository_root=self.registry.repository_root,
            session_id=session.session_id,
            session_revision=session.revision,
            trip_id=record.trip_id,
            accepted_plan_id=session.accepted_plan_id,
            day=day,
            route_leg_id=route_leg_id,
            geography=self._geographies[session.run_id],
            route_matrix_record=bundle.route_matrix or {},
            parent_plan=bundle.parent_plan,
            registered_bundles=tuple(self._evidence_bundles[session.run_id].values()),
            replacement_target_id=replacement_target_id,
            role=role,
            maximum_detour_minutes=maximum_detour_minutes,
            limit=limit,
        )

    def resolve_validated_route_context(
        self,
        *,
        run_id: str,
        accepted_plan_id: str,
        route_leg_id: str,
        selected_day: int,
    ) -> tuple[
        tuple[str, tuple[float, float]],
        tuple[str, tuple[float, float]],
    ]:
        """Resolve one exact accepted-plan leg into authoritative insertion endpoints."""

        self.load(run_id)
        plans = [plan for plan in self._geographies[run_id].get("plans", []) if plan.get("plan_id") == accepted_plan_id]
        if len(plans) != 1:
            raise WorkspaceError("accepted_route_context_unavailable", 409)
        matches = [
            feature
            for feature in plans[0].get("validated_legs", {}).get("features", [])
            if feature.get("properties", {}).get("route_leg_id") == route_leg_id
        ]
        if not matches:
            raise WorkspaceError("selected_route_segment_not_found", 422)
        if len(matches) != 1:
            raise WorkspaceError("selected_route_segment_ambiguous", 409)
        feature = matches[0]
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            raise WorkspaceError("selected_route_segment_invalid", 409)
        allowed_days = {
            value
            for value in (
                properties.get("day"),
                properties.get("from_day"),
                properties.get("to_day"),
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if allowed_days and selected_day not in allowed_days:
            raise WorkspaceError("selected_route_segment_day_mismatch", 422)
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "LineString" or not isinstance(coordinates, list) or len(coordinates) < 2:
            raise WorkspaceError("selected_route_segment_invalid", 409)
        origin_id = str(properties.get("origin_id") or "")
        destination_id = str(properties.get("destination_id") or "")
        if not origin_id or not destination_id:
            raise WorkspaceError("selected_route_segment_invalid", 409)
        return (
            (origin_id, _route_endpoint(coordinates[0])),
            (destination_id, _route_endpoint(coordinates[-1])),
        )

    def store_map_snap_preview(
        self,
        *,
        session: ProductSession,
        preview: SnapPreviewV1,
        selected_day: int,
        route_leg_id: str | None,
        predecessor_id: str | None,
        successor_id: str | None,
        travel_mode: str,
        target_waypoint_id: str | None,
    ) -> None:
        """Retain only executable route-checked previews for an explicit confirmation."""

        if preview.operation_intent not in self.workspace.CONFIRMED_MAP_OPERATIONS:
            return
        if not preview.draft_append_allowed or preview.validation_state != "route_checked":
            return
        if not route_leg_id or not predecessor_id or not successor_id:
            raise WorkspaceError("snap_preview_route_context_required", 409)
        if preview.operation_intent == "relocate_custom_waypoint":
            current = self.resolve_custom_waypoint(session, target_waypoint_id)
            insertion = current["insertion"]
            if (
                current["day"] != selected_day
                or insertion["route_leg_id"] != route_leg_id
                or insertion["predecessor_id"] != predecessor_id
                or insertion["successor_id"] != successor_id
                or insertion["travel_mode"] != travel_mode
            ):
                raise WorkspaceError("waypoint_insertion_context_mismatch", 409)
        elif target_waypoint_id is not None:
            raise WorkspaceError("unexpected_target_waypoint", 422)
        try:
            expires_at = datetime.fromisoformat(preview.expires_at)
        except ValueError:
            raise WorkspaceError("snap_preview_invalid", 409) from None
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise WorkspaceError("snap_preview_invalid", 409)
        with self._map_preview_lock:
            self._purge_map_snap_previews_locked()
            while len(self._map_snap_previews) >= self.MAX_STORED_SNAP_PREVIEWS:
                self._map_snap_previews.pop(next(iter(self._map_snap_previews)))
            self._map_snap_previews[preview.snap_preview_id] = _StoredSnapPreview(
                preview=preview,
                session_id=session.session_id,
                session_revision=session.revision,
                accepted_plan_id=session.accepted_plan_id,
                selected_day=selected_day,
                route_leg_id=route_leg_id,
                predecessor_id=predecessor_id,
                successor_id=successor_id,
                travel_mode=travel_mode,
                operation_intent=preview.operation_intent,
                target_waypoint_id=target_waypoint_id,
                expires_at=expires_at,
            )

    def confirm_map_snap_preview(
        self,
        *,
        session: ProductSession,
        snap_preview_id: str,
        payload: dict[str, Any],
    ) -> DraftOperationV1:
        """Resolve one stored preview into exactly one persistent typed operation."""

        with self._map_preview_lock:
            entry = self._map_snap_previews.get(snap_preview_id)
            if entry is None:
                self._purge_map_snap_previews_locked()
                raise WorkspaceError("snap_preview_not_found", 404)
            if entry.session_id != session.session_id:
                raise WorkspaceError("snap_preview_session_mismatch", 403)
            if entry.consumed:
                raise WorkspaceError("snap_preview_already_consumed", 409)
            if datetime.now(UTC) >= entry.expires_at:
                self._map_snap_previews.pop(snap_preview_id, None)
                raise WorkspaceError("snap_preview_expired", 410)
            expected_revision = payload.get("expected_revision")
            self.workspace.expect_revision(session, expected_revision)
            self.validate_session_draft(session)
            if (
                entry.session_revision != session.revision
                or entry.accepted_plan_id != session.accepted_plan_id
                or entry.preview.operation_intent != entry.operation_intent
                or not entry.preview.draft_append_allowed
                or entry.preview.validation_state != "route_checked"
            ):
                raise WorkspaceError("stale_snap_preview", 409)
            if entry.preview.confirmation_required and payload.get("warning_acknowledged") is not True:
                raise WorkspaceError("snap_warning_acknowledgement_required", 409)

            operation_type = entry.operation_intent
            if operation_type == "relocate_custom_waypoint":
                existing = self.resolve_custom_waypoint(session, entry.target_waypoint_id)
                for field in ("name", "role", "duration"):
                    if field in payload and payload[field] != existing[field]:
                        raise WorkspaceError("relocation_metadata_mismatch", 422)
                target = str(entry.target_waypoint_id)
                name = existing["name"]
                role = existing["role"]
                duration = deepcopy(existing["duration"])
            else:
                required_metadata = {"name", "role", "duration"}
                if not required_metadata.issubset(payload):
                    raise WorkspaceError("confirmed_waypoint_metadata_required", 422)
                target = entry.preview.entity_id
                name_value = payload.get("name")
                name = name_value.strip() if isinstance(name_value, str) else name_value
                role = payload.get("role")
                duration = deepcopy(payload.get("duration"))

            preview_payload = entry.preview.as_dict()
            parameters = {
                "schema_version": "confirmed-map-operation-v1",
                "snap_preview_id": snap_preview_id,
                "waypoint_id": target,
                "name": name,
                "day": entry.selected_day,
                "role": role,
                "duration": duration,
                "raw_coordinate": preview_payload["raw_coordinate"],
                "snapped_coordinate": preview_payload["snapped_coordinate"],
                "selected_access_point": preview_payload["selected_access_point"],
                "insertion": {
                    "route_leg_id": entry.route_leg_id,
                    "predecessor_id": entry.predecessor_id,
                    "successor_id": entry.successor_id,
                    "travel_mode": entry.travel_mode,
                },
                "affected_route_legs": preview_payload["affected_route_legs"],
            }
            evidence_refs = tuple(
                f"route_query:{leg.query_hash}" for leg in entry.preview.affected_route_legs
            )
            operation = self.workspace.append_confirmed_map_operation(
                session,
                expected_revision=expected_revision,
                operation_type=operation_type,
                target=target,
                parameters=parameters,
                evidence_refs=evidence_refs,
            )
            entry.consumed = True
            return operation

    def resolve_custom_waypoint(
        self,
        session: ProductSession,
        target_waypoint_id: str | None,
    ) -> dict[str, Any]:
        if not target_waypoint_id:
            raise WorkspaceError("target_waypoint_required", 422)
        resolved = self.workspace.reduce_confirmed_map_draft(session.draft).get(target_waypoint_id)
        if resolved is None:
            raise WorkspaceError("custom_waypoint_not_found", 422)
        return deepcopy(resolved)

    def _purge_map_snap_previews_locked(self) -> None:
        now = datetime.now(UTC)
        expired = [key for key, entry in self._map_snap_previews.items() if now >= entry.expires_at]
        for key in expired:
            self._map_snap_previews.pop(key, None)

    def valid_alternatives(self, run_id: str) -> set[str]:
        self.load(run_id)
        return {
            str(plan.get("plan_id") or "")
            for plan in self._geographies[run_id].get("plans", [])
            if plan.get("role") != "original" and plan.get("plan_id")
        }

    def preview(
        self,
        session_id: str,
        expected_revision: Any,
        *,
        mutation_token: str | None = None,
    ) -> dict[str, Any]:
        session = self.workspace.get(session_id)
        self.validate_session_draft(session)
        draft_content_hash = _draft_content_hash(session.draft)
        existing = session.proposal or {}
        existing_revision = existing.get("expected_revision")
        cache_matches = (
            existing.get("schema_version") == "draft-preview-v1"
            and existing.get("draft_content_hash") == draft_content_hash
            and isinstance(existing_revision, int)
            and existing_revision + 1 == session.revision
            and expected_revision in {existing_revision, session.revision}
        )
        existing_is_role = (
            (existing.get("role_impact") or {}).get("schema_version")
            == "evaluated-role-impact-v1"
        )
        if existing_is_role:
            _verify_role_proposal_integrity(existing, mutation_token)
        existing_should_be_role = bool(existing) and bool(session.draft) and all(
            operation.type == DraftOperationTypeV2.SET_STOP_ROLE.value
            for operation in session.draft
        )
        if existing_should_be_role and not existing_is_role:
            raise WorkspaceError("draft_role_proposal_invalid", 409)
        if existing_is_role and not cache_matches:
            raise WorkspaceError("draft_role_proposal_invalid", 409)
        revalidate_cached_role = existing_is_role and cache_matches
        if cache_matches and not revalidate_cached_role:
            return deepcopy(existing)
        if not revalidate_cached_role:
            self.workspace.expect_revision(session, expected_revision)
        self.load(session.run_id)
        if not session.draft:
            raise WorkspaceError("draft_is_empty")
        compiler = self._draft_compilers[session.run_id]
        if compiler is None:
            raise WorkspaceError("draft_preview_not_supported", 409)
        compiled = compiler.compile(session.draft, accepted_plan_id=session.accepted_plan_id)
        certificate = deepcopy(compiled.certificate or {})
        if (
            compiled.schedule_impact is not None
            and compiled.schedule_impact.get("schema_version")
            == "evaluated-role-impact-v1"
        ):
            certificate = json.loads(
                json.dumps(
                    certificate,
                    allow_nan=False,
                    ensure_ascii=False,
                )
            )
        if revalidate_cached_role:
            certificate = _replay_role_certificate_with_persisted_timestamp(
                certificate,
                existing,
            )
        child = compiled.child_plan or {}
        diff = compiled.diff or {}
        eligible = compiled.state == "eligible"
        repair = None
        geography_plan = None
        if child and diff and certificate:
            runtime_bundle = replace(
                compiled.source_bundle,
                child_plan=child,
                diff=diff,
                certificate=certificate,
                planner_runs=(compiled.planner_run,) if compiled.planner_run else compiled.source_bundle.planner_runs,
            )
            repair = build_product_dashboard_view_model(runtime_bundle)["repair"]
            geography_plan = next(
                (
                    deepcopy(plan)
                    for plan in self._geographies[session.run_id].get("plans") or ()
                    if plan.get("plan_id") == child.get("plan_id")
                    and plan.get("content_hash") == child.get("content_hash")
                ),
                None,
            )
            if geography_plan is None:
                source_child = compiled.source_bundle.child_plan or {}
                source_geography = next(
                    (
                        plan
                        for plan in self._geographies[session.run_id].get("plans") or ()
                        if plan.get("plan_id") == source_child.get("plan_id")
                        and plan.get("content_hash") == source_child.get("content_hash")
                    ),
                    None,
                )
                if source_geography is not None and _route_projection(child) == _route_projection(source_child):
                    geography_plan = _relineage_geography_plan(source_geography, child)
            if geography_plan is None and compiled.route_legs:
                runtime_geography = build_geographic_workspace_v2(
                    runtime_bundle,
                    route_legs_by_plan={
                        compiler.parent_plan_id: compiled.parent_route_legs,
                        str(child.get("plan_id") or ""): compiled.route_legs,
                    },
                )
                geography_plan = next(
                    (
                        deepcopy(plan)
                        for plan in runtime_geography.get("plans") or ()
                        if plan.get("plan_id") == child.get("plan_id")
                        and plan.get("content_hash") == child.get("content_hash")
                    ),
                    None,
                )
                route_validation = certificate.get("route_validation") or {}
                coverage = (geography_plan or {}).get("coverage") or {}
                if (
                    geography_plan is None
                    or coverage.get("complete") is not True
                    or coverage.get("required_leg_count") != route_validation.get("required_leg_count")
                    or coverage.get("road_validated_leg_count")
                    != route_validation.get("road_validated_leg_count")
                    or coverage.get("gap_count") != route_validation.get("missing_leg_count")
                ):
                    raise WorkspaceError("draft_geography_evidence_mismatch", 409)
            if geography_plan:
                geography_plan["role"] = "draft_preview"
                geography_plan["label"] = "Evaluated draft preview"
                for collection_name in ("stops", "route_path", "validated_legs", "gaps"):
                    for feature in geography_plan[collection_name]["features"]:
                        feature["properties"]["role"] = "draft_preview"
        proposal = {
            "schema_version": "draft-preview-v1",
            "state": compiled.state,
            "reason": compiled.reason,
            "eligibility": certificate.get("comparison_eligibility") or "ineligible",
            "provenance": (
                "independent_evaluated_direct_edit"
                if compiled.execution_mode == "typed_direct_edit_independent_evaluation"
                else "deterministic_repair_pipeline"
            ),
            "execution_mode": compiled.execution_mode,
            "expected_revision": (
                existing_revision if revalidate_cached_role else expected_revision
            ),
            "session_revision": (
                session.revision if revalidate_cached_role else session.revision + 1
            ),
            "draft_content_hash": draft_content_hash,
            "draft_operation_ids": [row.get("operation_id") for row in compiled.operations],
            "message": (
                "The typed draft produced an immutable in-memory child that passed independent evaluation."
                if eligible
                else "The typed draft did not produce an independently eligible child."
            ),
            "parent_plan_id": compiler.parent_plan_id,
            "parent_plan_content_hash": compiler.parent_plan.get("content_hash"),
            "child_plan_id": child.get("plan_id"),
            "child_plan_content_hash": child.get("content_hash"),
            "certificate_id": certificate.get("certificate_id"),
            "certificate_content_hash": certificate.get("content_hash"),
            "diff_id": diff.get("diff_id"),
            "diff_content_hash": stable_content_hash(diff) if diff else None,
            "diff_identity": {
                "diff_id": diff.get("diff_id"),
                "parent_plan_id": diff.get("parent_plan_id"),
                "child_plan_id": diff.get("child_plan_id"),
                "content_hash": stable_content_hash(diff) if diff else None,
            },
            "evaluation_status": certificate.get("evaluation_status"),
            "certificate_identity": {
                "certificate_id": certificate.get("certificate_id"),
                "plan_id": certificate.get("plan_id"),
                "plan_content_hash": certificate.get("plan_content_hash"),
                "content_hash": certificate.get("content_hash"),
            },
            "route_validation": deepcopy(certificate.get("route_validation") or {}),
            "route_validation_identity": {
                key: (certificate.get("route_validation") or {}).get(key)
                for key in (
                    "matrix_id",
                    "context_snapshot_id",
                    "source_bundle_id",
                    "source_content_sha256",
                )
            },
            "evidence": {
                "artifact_grounding_status": certificate.get("artifact_grounding_status"),
                "hard_feasibility_status": certificate.get("hard_feasibility_status"),
                "evaluation_status": certificate.get("evaluation_status"),
                "comparison_eligibility": certificate.get("comparison_eligibility"),
                "route_validation": deepcopy(certificate.get("route_validation") or {}),
                "requested_method": certificate.get("requested_method"),
                "executed_method": certificate.get("executed_method"),
            },
            "geography_plan": geography_plan,
            "compiled_request": {
                "schema_version": "typed-repair-request-v1",
                "source_request_id": compiled.source_request_id,
                "operations": [deepcopy(row) for row in compiled.operations],
            },
            "repair": repair,
        }
        if (
            compiled.schedule_impact is not None
            and compiled.schedule_impact.get("schema_version")
            == "evaluated-role-impact-v1"
        ):
            route_matrix, _, _ = compiler._runtime_inputs()
            _attach_role_proposal_evidence(
                proposal,
                role_impact=compiled.schedule_impact,
                diff=diff,
                certificate=certificate,
                parent_plan=compiler.parent_plan,
                child_plan=child,
                route_matrix=route_matrix,
                expected_route_legs=compiled.parent_route_legs,
            )
            _attach_role_proposal_integrity(proposal, mutation_token)
            if revalidate_cached_role:
                if proposal != existing:
                    raise WorkspaceError("draft_role_proposal_invalid", 409)
                return deepcopy(existing)
            self.workspace.set_proposal(session, proposal, expected_revision)
            return proposal
        if (
            compiled.schedule_impact is not None
            and compiled.schedule_impact.get("schema_version")
            == "evaluated-time-window-schedule-impact-v1"
        ):
            route_matrix, _, _ = compiler._runtime_inputs()
            _attach_time_window_proposal_evidence(
                proposal,
                schedule_impact=compiled.schedule_impact,
                diff=diff,
                certificate=certificate,
                parent_plan=compiler.parent_plan,
                child_plan=child,
                route_matrix=route_matrix,
                expected_route_legs=compiled.parent_route_legs,
            )
            self.workspace.set_proposal(session, proposal, expected_revision)
            return proposal
        if compiled.schedule_impact is not None:
            schedule_impact = deepcopy(compiled.schedule_impact)
            schedule_payload = dict(schedule_impact)
            schedule_hash = str(schedule_payload.pop("content_hash", ""))
            parent_schedule = schedule_impact.get("parent") or {}
            child_schedule = schedule_impact.get("child") or {}
            default_visit_minutes = schedule_impact.get("configured_default_visit_minutes")
            duration_cost_policy = schedule_impact.get("duration_cost_policy")
            duration_changes = diff.get("duration_changes")
            if (
                stable_content_hash(schedule_payload) != schedule_hash
                or not isinstance(default_visit_minutes, (int, float))
                or isinstance(default_visit_minutes, bool)
                or not math.isfinite(default_visit_minutes)
                or not 15 <= float(default_visit_minutes) <= 480
                or not _valid_duration_cost_policy(duration_cost_policy)
                or diff.get("schema_version") != "plan-diff-v2"
                or not isinstance(duration_changes, list)
                or not duration_changes
                or diff.get("time_shifts")
                or diff.get("road_changes")
                or any(
                    not _valid_duration_change(
                        change,
                        float(default_visit_minutes),
                        duration_cost_policy,
                    )
                    for change in duration_changes
                )
                or schedule_impact.get("route_matrix_id")
                != (certificate.get("route_validation") or {}).get("matrix_id")
                or parent_schedule.get("plan_id") != compiler.parent_plan_id
                or parent_schedule.get("plan_content_hash")
                != compiler.parent_plan.get("content_hash")
                or child_schedule.get("plan_id") != child.get("plan_id")
                or child_schedule.get("plan_content_hash") != child.get("content_hash")
            ):
                raise WorkspaceError("draft_schedule_evidence_mismatch", 409)
            proposal["schedule_impact"] = schedule_impact
            proposal["plan_diff"] = deepcopy(diff)
            certificate_metrics = certificate.get("metrics") or {}
            schedule_metrics = {
                key: value
                for key, value in certificate_metrics.items()
                if key.startswith("day_")
                or key.startswith("route_")
                or key == "schedule_missing_opening_window_count"
            }
            accounting_metrics = {
                key: value
                for key, value in schedule_metrics.items()
                if key != "schedule_missing_opening_window_count"
            }
            certificate_warning_codes = [
                row.get("code") for row in certificate.get("warnings") or ()
            ]
            certificate_failure_codes = [
                row.get("code") for row in certificate.get("failures") or ()
            ]
            missing_window_ids = child_schedule.get("missing_opening_window_stop_ids") or []
            warning_codes = (
                ["opening_window_evidence_missing"] if missing_window_ids else []
            )
            failure_codes = list(child_schedule.get("blocking_codes") or ())
            expected_schedule_outcome = {
                "failed": ("FAILED", "ineligible"),
                "incomplete_evidence": ("PASSED_WITH_WARNINGS", "eligible"),
                "passed": ("PASSED", "eligible"),
            }.get(child_schedule.get("status"))
            if (
                accounting_metrics != child_schedule.get("metrics")
                or schedule_metrics.get("schedule_missing_opening_window_count")
                != float(len(missing_window_ids))
                or warning_codes
                != certificate_warning_codes
                or failure_codes != certificate_failure_codes
                or expected_schedule_outcome is None
                or (
                    certificate.get("evaluation_status"),
                    certificate.get("comparison_eligibility"),
                )
                != expected_schedule_outcome
            ):
                raise WorkspaceError("draft_schedule_evidence_mismatch", 409)
            certificate_schedule_payload = {
                "schema_version": "evaluated-duration-certificate-schedule-evidence-v1",
                "certificate_id": certificate.get("certificate_id"),
                "certificate_content_hash": certificate.get("content_hash"),
                "plan_id": certificate.get("plan_id"),
                "plan_content_hash": certificate.get("plan_content_hash"),
                "evaluator_version": certificate.get("evaluator_version"),
                "evaluation_status": certificate.get("evaluation_status"),
                "comparison_eligibility": certificate.get("comparison_eligibility"),
                "warning_codes": warning_codes,
                "failure_codes": failure_codes,
                "schedule_metrics": schedule_metrics,
            }
            proposal["certificate_schedule_evidence"] = {
                **certificate_schedule_payload,
                "content_hash": stable_content_hash(certificate_schedule_payload),
            }
            duration_eligible = certificate.get("comparison_eligibility") == "eligible"
            proposal["decision_eligible"] = duration_eligible
            proposal["ranking_eligible"] = False
            proposal["acceptance_eligible"] = False
            proposal["acceptance_blocking_code"] = "acceptance_not_enabled_until_w5"
            proposal["schedule_impact_identity"] = {
                "content_hash": schedule_hash,
                "evaluator_version": schedule_impact.get("evaluator_version"),
                "route_matrix_id": schedule_impact.get("route_matrix_id"),
                "parent_plan_id": parent_schedule.get("plan_id"),
                "parent_plan_content_hash": parent_schedule.get("plan_content_hash"),
                "child_plan_id": child_schedule.get("plan_id"),
                "child_plan_content_hash": child_schedule.get("plan_content_hash"),
                "certificate_id": certificate.get("certificate_id"),
                "certificate_content_hash": certificate.get("content_hash"),
            }
        self.workspace.set_proposal(session, proposal, expected_revision)
        return proposal

    def registered_proposal(self, session_id: str, expected_revision: Any) -> dict[str, Any]:
        session = self.workspace.get(session_id)
        bundle, view = self.load(session.run_id)
        if not (bundle.child_plan and bundle.certificate and bundle.diff):
            raise WorkspaceError("registered_repair_unavailable", 409)
        proposal = {
            "state": "eligible",
            "provenance": "registered_fixture_artifacts",
            "message": "Registered repair ready for comparison and evidence review.",
            "parent_plan_id": bundle.parent_plan["plan_id"],
            "child_plan_id": bundle.child_plan["plan_id"],
            "certificate_id": bundle.certificate.get("certificate_id"),
            "diff_id": bundle.diff.get("diff_id"),
            "repair": view["repair"],
        }
        self.workspace.set_proposal(session, proposal, expected_revision)
        return proposal


def _draft_content_hash(operations: list[Any]) -> str:
    records = []
    for operation in operations:
        raw = operation.as_dict() if hasattr(operation, "as_dict") else dict(operation)
        records.append(
            {
                "operation_id": str(raw.get("operation_id") or ""),
                "type": str(raw.get("type") or ""),
                "target": str(raw.get("target") or ""),
                "parameters": dict(raw.get("parameters") or {}),
                "source": str(raw.get("source") or ""),
                "evidence_refs": list(raw.get("evidence_refs") or ()),
            }
        )
    return stable_content_hash({"operations": records})


def _attach_role_proposal_integrity(
    proposal: dict[str, Any],
    mutation_token: str | None,
) -> None:
    if not isinstance(mutation_token, str) or not mutation_token:
        raise WorkspaceError("invalid_session_token", 403)
    payload = deepcopy(proposal)
    payload.pop("role_proposal_integrity", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    proposal["role_proposal_integrity"] = {
        "schema_version": "evaluated-role-proposal-integrity-v1",
        "algorithm": "hmac-sha256-session-token-v1",
        "digest": hmac.new(
            mutation_token.encode("utf-8"),
            encoded,
            hashlib.sha256,
        ).hexdigest(),
    }


def _verify_role_proposal_integrity(
    proposal: dict[str, Any],
    mutation_token: str | None,
) -> None:
    integrity = proposal.get("role_proposal_integrity")
    if (
        not isinstance(mutation_token, str)
        or not mutation_token
        or not isinstance(integrity, dict)
        or set(integrity) != {"schema_version", "algorithm", "digest"}
        or integrity.get("schema_version")
        != "evaluated-role-proposal-integrity-v1"
        or integrity.get("algorithm") != "hmac-sha256-session-token-v1"
        or not isinstance(integrity.get("digest"), str)
    ):
        raise WorkspaceError("draft_role_proposal_invalid", 409)
    payload = deepcopy(proposal)
    payload.pop("role_proposal_integrity", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected = hmac.new(
        mutation_token.encode("utf-8"),
        encoded,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(integrity["digest"], expected):
        raise WorkspaceError("draft_role_proposal_invalid", 409)


def _replay_role_certificate_with_persisted_timestamp(
    certificate: dict[str, Any],
    existing_proposal: dict[str, Any],
) -> dict[str, Any]:
    """Recreate the exact certificate using its persisted evaluation timestamp."""

    evidence = existing_proposal.get("certificate_role_evidence") or {}
    stored = evidence.get("certificate_record")
    if not isinstance(stored, dict):
        raise WorkspaceError("draft_role_proposal_invalid", 409)
    stored_payload = dict(stored)
    stored_hash = stored_payload.pop("content_hash", None)
    outer_payload = dict(evidence)
    outer_hash = outer_payload.pop("content_hash", None)
    certificate_hashes = {
        existing_proposal.get("certificate_content_hash"),
        (existing_proposal.get("certificate_identity") or {}).get("content_hash"),
        evidence.get("certificate_content_hash"),
        (existing_proposal.get("role_impact_identity") or {}).get(
            "certificate_content_hash"
        ),
        stored_hash,
    }
    if (
        stable_content_hash(stored_payload) != stored_hash
        or stable_content_hash(outer_payload) != outer_hash
        or len(certificate_hashes) != 1
        or not isinstance(stored.get("evaluated_at"), str)
        or not stored["evaluated_at"]
    ):
        raise WorkspaceError("draft_role_proposal_invalid", 409)
    replayed = deepcopy(certificate)
    replayed["evaluated_at"] = stored["evaluated_at"]
    replay_payload = dict(replayed)
    replay_payload.pop("content_hash", None)
    replayed["content_hash"] = stable_content_hash(replay_payload)
    if replayed != stored:
        raise WorkspaceError("draft_role_proposal_invalid", 409)
    return replayed


def _attach_role_proposal_evidence(
    proposal: dict[str, Any],
    *,
    role_impact: dict[str, Any],
    diff: dict[str, Any],
    certificate: dict[str, Any],
    parent_plan: dict[str, Any],
    child_plan: dict[str, Any],
    route_matrix: Any,
    expected_route_legs: tuple[dict[str, Any], ...],
) -> None:
    role_impact = deepcopy(role_impact)
    impact_payload = dict(role_impact)
    impact_hash = str(impact_payload.pop("content_hash", ""))
    default_visit_minutes = role_impact.get("configured_default_visit_minutes")
    day_start_minute = role_impact.get("configured_day_start_minute")
    route_leg_evidence = role_impact.get("route_leg_evidence")
    role_constraint_evidence = role_impact.get("role_constraint_evidence")
    cost_policy = role_impact.get("role_cost_policy")
    changes = diff.get("role_changes")
    parent_schedule = role_impact.get("parent_schedule")
    child_schedule = role_impact.get("child_schedule")
    expected_top_fields = {
        "schema_version",
        "evaluator_version",
        "route_matrix_id",
        "configured_default_visit_minutes",
        "configured_day_start_minute",
        "supported_evaluated_roles",
        "semantic_scope",
        "role_constraint_evidence",
        "route_leg_evidence",
        "role_cost_policy",
        "target_stop_ids",
        "affected_days",
        "parent_role_states",
        "child_role_states",
        "parent_schedule",
        "child_schedule",
        "invariance",
        "content_hash",
    }
    if (
        set(role_impact) != expected_top_fields
        or stable_content_hash(impact_payload) != impact_hash
        or role_impact.get("supported_evaluated_roles")
        != ["attraction", "activity", "meal", "rest_stop", "scenic_stop"]
        or role_impact.get("semantic_scope")
        != {
            "itinerary_role_semantics": "trip_specific_user_declared_use",
            "place_identity_unchanged": True,
            "place_categories_unchanged": True,
            "route_schedule_effect": "none_for_supported_visit_roles",
            "semantic_fit_claim": "unavailable",
            "recommendation_claim": "unavailable",
        }
        or role_constraint_evidence != build_role_constraint_evidence(parent_plan)
        or not _valid_duration_minutes(default_visit_minutes)
        or not _valid_clock_minute_number(day_start_minute)
        or not _valid_time_window_route_evidence(
            route_leg_evidence,
            route_matrix=route_matrix,
            expected_route_legs=expected_route_legs,
        )
        or not _valid_role_cost_policy(cost_policy)
        or diff.get("schema_version") != "plan-diff-v4"
        or not isinstance(changes, list)
        or not changes
        or len({change.get("stop_id") for change in changes}) != len(changes)
        or "duration_changes" in diff
        or "time_window_changes" in diff
        or diff.get("time_shifts")
        or diff.get("road_changes")
        or any(
            not _valid_role_change(
                change,
                cost_policy=cost_policy,
                role_constraint_evidence=role_constraint_evidence,
            )
            for change in changes
        )
        or not _role_states_match_plan(
            role_impact.get("parent_role_states"),
            parent_plan,
        )
        or not _role_states_match_plan(
            role_impact.get("child_role_states"),
            child_plan,
        )
        or not _role_changes_match_states(
            changes,
            role_impact["parent_role_states"],
            role_impact["child_role_states"],
        )
        or role_impact.get("target_stop_ids")
        != sorted(change["stop_id"] for change in changes)
        or role_impact.get("affected_days")
        != sorted({change["day"] for change in changes})
        or not _role_only_selected_stop_change(parent_plan, child_plan, changes)
        or not _valid_time_window_schedule_accounting(
            parent_schedule,
            default_visit_minutes=float(default_visit_minutes),
            day_start_minute=float(day_start_minute),
            route_leg_evidence=route_leg_evidence,
            plan=parent_plan,
        )
        or not _valid_time_window_schedule_accounting(
            child_schedule,
            default_visit_minutes=float(default_visit_minutes),
            day_start_minute=float(day_start_minute),
            route_leg_evidence=route_leg_evidence,
            plan=child_plan,
        )
        or _accounting_evidence_without_identity(parent_schedule)
        != _accounting_evidence_without_identity(child_schedule)
        or role_impact.get("route_matrix_id")
        != (certificate.get("route_validation") or {}).get("matrix_id")
        or len(route_leg_evidence)
        != (certificate.get("route_validation") or {}).get("required_leg_count")
        or role_impact.get("invariance")
        != {
            "sequence_unchanged": True,
            "ordered_days_unchanged": True,
            "route_ids_by_day_unchanged": True,
            "route_legs_unchanged": True,
            "schedule_accounting_unchanged": True,
        }
        or child_plan.get("modeled_metrics") != {}
        or child_plan.get("context_exposure_components") != {}
        or child_plan.get("change_components") != {}
    ):
        raise WorkspaceError("draft_role_evidence_mismatch", 409)

    missing_window_ids = child_schedule["missing_opening_window_stop_ids"]
    expected_warning_codes = _expected_role_constraint_warning_codes(
        child_plan,
        role_constraint_evidence,
    ) + (
        ["opening_window_evidence_missing"] if missing_window_ids else []
    )
    failure_codes = list(child_schedule["blocking_codes"])
    certificate_warning_codes = [
        row.get("code") for row in certificate.get("warnings") or ()
    ]
    certificate_failure_codes = [
        row.get("code") for row in certificate.get("failures") or ()
    ]
    expected_outcome = (
        ("FAILED", "ineligible")
        if failure_codes
        else ("PASSED_WITH_WARNINGS", "eligible")
        if expected_warning_codes
        else ("PASSED", "eligible")
    )
    certificate_metrics = certificate.get("metrics") or {}
    schedule_metrics = {
        key: value
        for key, value in certificate_metrics.items()
        if key.startswith("day_")
        or key.startswith("route_")
        or key == "schedule_missing_opening_window_count"
    }
    accounting_metrics = {
        key: value
        for key, value in schedule_metrics.items()
        if key != "schedule_missing_opening_window_count"
    }
    if (
        accounting_metrics != child_schedule["metrics"]
        or schedule_metrics.get("schedule_missing_opening_window_count")
        != float(len(missing_window_ids))
        or certificate_warning_codes != expected_warning_codes
        or certificate_failure_codes != failure_codes
        or (
            certificate.get("evaluation_status"),
            certificate.get("comparison_eligibility"),
        )
        != expected_outcome
    ):
        raise WorkspaceError("draft_role_evidence_mismatch", 409)

    proposal["role_impact"] = role_impact
    proposal["plan_diff"] = deepcopy(diff)
    certificate_payload = {
        "schema_version": "evaluated-role-certificate-evidence-v1",
        "certificate_id": certificate.get("certificate_id"),
        "certificate_content_hash": certificate.get("content_hash"),
        "plan_id": certificate.get("plan_id"),
        "plan_content_hash": certificate.get("plan_content_hash"),
        "evaluator_version": certificate.get("evaluator_version"),
        "evaluation_status": certificate.get("evaluation_status"),
        "comparison_eligibility": certificate.get("comparison_eligibility"),
        "warning_codes": expected_warning_codes,
        "failure_codes": failure_codes,
        "schedule_metrics": schedule_metrics,
        "semantic_scope": deepcopy(role_impact["semantic_scope"]),
        "role_constraint_evidence_content_hash": role_constraint_evidence[
            "content_hash"
        ],
        "certificate_record": deepcopy(certificate),
    }
    proposal["certificate_role_evidence"] = {
        **certificate_payload,
        "content_hash": stable_content_hash(certificate_payload),
    }
    proposal["decision_eligible"] = (
        certificate.get("comparison_eligibility") == "eligible"
    )
    proposal["ranking_eligible"] = False
    proposal["acceptance_eligible"] = False
    proposal["acceptance_blocking_code"] = "acceptance_not_enabled_until_w5"
    proposal["role_impact_identity"] = {
        "content_hash": impact_hash,
        "evaluator_version": role_impact.get("evaluator_version"),
        "route_matrix_id": role_impact.get("route_matrix_id"),
        "parent_plan_id": parent_schedule.get("plan_id"),
        "parent_plan_content_hash": parent_schedule.get("plan_content_hash"),
        "child_plan_id": child_schedule.get("plan_id"),
        "child_plan_content_hash": child_schedule.get("plan_content_hash"),
        "certificate_id": certificate.get("certificate_id"),
        "certificate_content_hash": certificate.get("content_hash"),
        "role_constraint_evidence_content_hash": role_constraint_evidence[
            "content_hash"
        ],
    }


def _valid_role_cost_policy(value: Any) -> bool:
    return value == {
        "base_change_cost": 0.25,
        "formula": "base_times_role_scoped_strength_weight",
        "constraint_strength_weights": {
            "": 1.0,
            "locked": 1000.0,
            "booked": 500.0,
            "hard": 100.0,
            "strong": 50.0,
            "soft": 10.0,
            "weak": 5.0,
            "preference": 1.0,
            "test_only": 0.0,
        },
    }


def _valid_role_change(
    value: Any,
    *,
    cost_policy: dict[str, Any],
    role_constraint_evidence: dict[str, Any],
) -> bool:
    fields = {
        "stop_id",
        "day",
        "from_role",
        "to_role",
        "from_source",
        "to_source",
        "owner_strength",
        "cost",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    before = value.get("from_role")
    after = value.get("to_role")
    expected_strength = _expected_role_owner_strength(
        role_constraint_evidence,
        str(value.get("stop_id") or ""),
    )
    weights = cost_policy["constraint_strength_weights"]
    return bool(
        isinstance(value.get("stop_id"), str)
        and value["stop_id"]
        and isinstance(value.get("day"), int)
        and not isinstance(value.get("day"), bool)
        and (before is None or before in ALL_ROLE_VALUES)
        and after in EVALUATED_STOP_ROLES
        and value.get("from_source")
        == ("unavailable" if before is None else "user_declared_itinerary_role")
        and value.get("to_source") == "user_declared_itinerary_role"
        and expected_strength is not None
        and value.get("owner_strength") == expected_strength
        and value.get("cost") == 0.25 * weights[expected_strength]
    )


EVALUATED_STOP_ROLES = {
    "attraction",
    "activity",
    "meal",
    "rest_stop",
    "scenic_stop",
}
ALL_ROLE_VALUES = EVALUATED_STOP_ROLES | {
    "lodging",
    "transport_hub",
    "route_waypoint",
    "origin",
    "destination",
}


def _expected_role_owner_strength(
    role_constraint_evidence: dict[str, Any],
    stop_id: str,
) -> str | None:
    weights = {
        "": 1.0,
        "locked": 1000.0,
        "booked": 500.0,
        "hard": 100.0,
        "strong": 50.0,
        "soft": 10.0,
        "weak": 5.0,
        "preference": 1.0,
        "test_only": 0.0,
    }
    result = ""
    for row in role_constraint_evidence.get("constraints") or ():
        if not isinstance(row, dict):
            return None
        if row.get("target_stop_id") != stop_id:
            continue
        strength = row.get("strength")
        if strength not in weights or row.get("required_role") not in ALL_ROLE_VALUES:
            return None
        if weights[strength] >= weights[result]:
            result = strength
    return result


def _role_states_match_plan(value: Any, plan: dict[str, Any]) -> bool:
    if not isinstance(value, list):
        return False
    expected: list[dict[str, Any]] = []
    for stop in plan.get("selected_stops") or ():
        if not isinstance(stop, dict):
            return False
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or "")
        role = stop.get("itinerary_role")
        source = stop.get("itinerary_role_source")
        if not stop_id or (role is not None and role not in ALL_ROLE_VALUES):
            return False
        if role is None:
            if source not in {None, "unavailable"}:
                return False
            source = "unavailable"
        elif source != "user_declared_itinerary_role":
            return False
        expected.append(
            {
                "stop_id": stop_id,
                "day": stop.get("day"),
                "itinerary_role": role,
                "itinerary_role_source": source,
            }
        )
    return value == expected


def _role_changes_match_states(
    changes: list[dict[str, Any]],
    parent_states: list[dict[str, Any]],
    child_states: list[dict[str, Any]],
) -> bool:
    parent_by_stop = {row["stop_id"]: row for row in parent_states}
    child_by_stop = {row["stop_id"]: row for row in child_states}
    return all(
        change["stop_id"] in parent_by_stop
        and change["stop_id"] in child_by_stop
        and parent_by_stop[change["stop_id"]]["itinerary_role"]
        == change["from_role"]
        and parent_by_stop[change["stop_id"]]["itinerary_role_source"]
        == change["from_source"]
        and child_by_stop[change["stop_id"]]["itinerary_role"]
        == change["to_role"]
        and child_by_stop[change["stop_id"]]["itinerary_role_source"]
        == change["to_source"]
        and parent_by_stop[change["stop_id"]]["day"] == change["day"]
        and child_by_stop[change["stop_id"]]["day"] == change["day"]
        for change in changes
    )


def _role_only_selected_stop_change(
    parent_plan: dict[str, Any],
    child_plan: dict[str, Any],
    changes: list[dict[str, Any]],
) -> bool:
    targets = {change["stop_id"] for change in changes}
    parent_stops = parent_plan.get("selected_stops") or ()
    child_stops = child_plan.get("selected_stops") or ()
    if len(parent_stops) != len(child_stops):
        return False
    for parent_stop, child_stop in zip(parent_stops, child_stops, strict=True):
        if not isinstance(parent_stop, dict) or not isinstance(child_stop, dict):
            return False
        stop_id = str(parent_stop.get("stop_id") or parent_stop.get("poi_id") or "")
        if stop_id != str(child_stop.get("stop_id") or child_stop.get("poi_id") or ""):
            return False
        if stop_id not in targets:
            if parent_stop != child_stop:
                return False
            continue
        parent_remainder = dict(parent_stop)
        child_remainder = dict(child_stop)
        for field in ("itinerary_role", "itinerary_role_source"):
            parent_remainder.pop(field, None)
            child_remainder.pop(field, None)
        if parent_remainder != child_remainder:
            return False
    return bool(
        parent_plan.get("sequence") == child_plan.get("sequence")
        and parent_plan.get("ordered_days") == child_plan.get("ordered_days")
        and parent_plan.get("route_ids_by_day") == child_plan.get("route_ids_by_day")
        and parent_plan.get("day_assignments") == child_plan.get("day_assignments")
        and parent_plan.get("lodging_assignments")
        == child_plan.get("lodging_assignments")
    )


def _accounting_evidence_without_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"plan_id", "plan_content_hash", "content_hash"}
    }


def _expected_role_constraint_warning_codes(
    plan: dict[str, Any],
    role_constraint_evidence: dict[str, Any],
) -> list[str]:
    roles = {
        str(stop.get("stop_id") or stop.get("poi_id") or ""): stop.get(
            "itinerary_role"
        )
        for stop in plan.get("selected_stops") or ()
        if isinstance(stop, dict)
    }
    warnings: list[str] = []
    for row in role_constraint_evidence.get("constraints") or ():
        if not isinstance(row, dict):
            continue
        target = str(row.get("target_stop_id") or "")
        if (
            row.get("permission_semantics") == "weighted_mismatch_allowed"
            and roles.get(target) != row.get("required_role")
        ):
            warnings.append("owned_role_constraint_unsatisfied")
    return warnings


def _attach_time_window_proposal_evidence(
    proposal: dict[str, Any],
    *,
    schedule_impact: dict[str, Any],
    diff: dict[str, Any],
    certificate: dict[str, Any],
    parent_plan: dict[str, Any],
    child_plan: dict[str, Any],
    route_matrix: Any,
    expected_route_legs: tuple[dict[str, Any], ...],
) -> None:
    schedule_impact = deepcopy(schedule_impact)
    schedule_payload = dict(schedule_impact)
    schedule_hash = str(schedule_payload.pop("content_hash", ""))
    parent_schedule = schedule_impact.get("parent")
    child_schedule = schedule_impact.get("child")
    default_visit_minutes = schedule_impact.get("configured_default_visit_minutes")
    day_start_minute = schedule_impact.get("configured_day_start_minute")
    route_leg_evidence = schedule_impact.get("route_leg_evidence")
    cost_policy = schedule_impact.get("time_window_cost_policy")
    changes = diff.get("time_window_changes")
    expected_top_fields = {
        "schema_version",
        "evaluator_version",
        "route_matrix_id",
        "configured_default_visit_minutes",
        "configured_day_start_minute",
        "route_leg_evidence",
        "time_window_cost_policy",
        "affected_days",
        "target_stop_ids",
        "parent",
        "child",
        "content_hash",
    }
    if (
        set(schedule_impact) != expected_top_fields
        or stable_content_hash(schedule_payload) != schedule_hash
        or not _valid_duration_minutes(default_visit_minutes)
        or not _valid_clock_minute_number(day_start_minute)
        or not _valid_time_window_route_evidence(
            route_leg_evidence,
            route_matrix=route_matrix,
            expected_route_legs=expected_route_legs,
        )
        or not _valid_time_window_cost_policy(cost_policy)
        or diff.get("schema_version") != "plan-diff-v3"
        or not isinstance(changes, list)
        or not changes
        or len({change.get("stop_id") for change in changes}) != len(changes)
        or "duration_changes" in diff
        or diff.get("time_shifts")
        or diff.get("road_changes")
        or any(
            not _valid_time_window_change(change, cost_policy)
            for change in changes
        )
        or not _valid_time_window_schedule_accounting(
            parent_schedule,
            default_visit_minutes=float(default_visit_minutes),
            day_start_minute=float(day_start_minute),
            route_leg_evidence=route_leg_evidence,
            plan=parent_plan,
        )
        or not _valid_time_window_schedule_accounting(
            child_schedule,
            default_visit_minutes=float(default_visit_minutes),
            day_start_minute=float(day_start_minute),
            route_leg_evidence=route_leg_evidence,
            plan=child_plan,
        )
        or schedule_impact.get("route_matrix_id")
        != (certificate.get("route_validation") or {}).get("matrix_id")
        or len(route_leg_evidence)
        != (certificate.get("route_validation") or {}).get("required_leg_count")
        or parent_schedule.get("plan_id") != parent_plan.get("plan_id")
        or parent_schedule.get("plan_content_hash")
        != parent_plan.get("content_hash")
        or child_schedule.get("plan_id") != child_plan.get("plan_id")
        or child_schedule.get("plan_content_hash")
        != child_plan.get("content_hash")
        or schedule_impact.get("target_stop_ids")
        != sorted(change["stop_id"] for change in changes)
        or schedule_impact.get("affected_days")
        != sorted({change["day"] for change in changes})
        or not _time_window_changes_match_traces(
            changes,
            parent_schedule["stop_traces"],
            child_schedule["stop_traces"],
        )
    ):
        raise WorkspaceError("draft_schedule_evidence_mismatch", 409)

    certificate_metrics = certificate.get("metrics") or {}
    schedule_metrics = {
        key: value
        for key, value in certificate_metrics.items()
        if key.startswith("day_")
        or key.startswith("route_")
        or key == "schedule_missing_opening_window_count"
    }
    accounting_metrics = {
        key: value
        for key, value in schedule_metrics.items()
        if key != "schedule_missing_opening_window_count"
    }
    missing_window_ids = child_schedule["missing_opening_window_stop_ids"]
    warning_codes = (
        ["opening_window_evidence_missing"] if missing_window_ids else []
    )
    failure_codes = list(child_schedule["blocking_codes"])
    certificate_warning_codes = [
        row.get("code") for row in certificate.get("warnings") or ()
    ]
    certificate_failure_codes = [
        row.get("code") for row in certificate.get("failures") or ()
    ]
    expected_schedule_outcome = {
        "failed": ("FAILED", "ineligible"),
        "incomplete_evidence": ("PASSED_WITH_WARNINGS", "eligible"),
        "passed": ("PASSED", "eligible"),
    }.get(child_schedule.get("status"))
    if (
        accounting_metrics != child_schedule["metrics"]
        or schedule_metrics.get("schedule_missing_opening_window_count")
        != float(len(missing_window_ids))
        or warning_codes != certificate_warning_codes
        or failure_codes != certificate_failure_codes
        or expected_schedule_outcome is None
        or (
            certificate.get("evaluation_status"),
            certificate.get("comparison_eligibility"),
        )
        != expected_schedule_outcome
    ):
        raise WorkspaceError("draft_schedule_evidence_mismatch", 409)

    proposal["schedule_impact"] = schedule_impact
    proposal["plan_diff"] = deepcopy(diff)
    certificate_schedule_payload = {
        "schema_version": "evaluated-time-window-certificate-schedule-evidence-v1",
        "certificate_id": certificate.get("certificate_id"),
        "certificate_content_hash": certificate.get("content_hash"),
        "plan_id": certificate.get("plan_id"),
        "plan_content_hash": certificate.get("plan_content_hash"),
        "evaluator_version": certificate.get("evaluator_version"),
        "evaluation_status": certificate.get("evaluation_status"),
        "comparison_eligibility": certificate.get("comparison_eligibility"),
        "warning_codes": warning_codes,
        "failure_codes": failure_codes,
        "schedule_metrics": schedule_metrics,
    }
    proposal["certificate_schedule_evidence"] = {
        **certificate_schedule_payload,
        "content_hash": stable_content_hash(certificate_schedule_payload),
    }
    decision_eligible = certificate.get("comparison_eligibility") == "eligible"
    proposal["decision_eligible"] = decision_eligible
    proposal["ranking_eligible"] = False
    proposal["acceptance_eligible"] = False
    proposal["acceptance_blocking_code"] = "acceptance_not_enabled_until_w5"
    proposal["schedule_impact_identity"] = {
        "content_hash": schedule_hash,
        "evaluator_version": schedule_impact.get("evaluator_version"),
        "route_matrix_id": schedule_impact.get("route_matrix_id"),
        "parent_plan_id": parent_schedule.get("plan_id"),
        "parent_plan_content_hash": parent_schedule.get("plan_content_hash"),
        "child_plan_id": child_schedule.get("plan_id"),
        "child_plan_content_hash": child_schedule.get("plan_content_hash"),
        "certificate_id": certificate.get("certificate_id"),
        "certificate_content_hash": certificate.get("content_hash"),
    }


def _valid_time_window_cost_policy(value: Any) -> bool:
    return value == {
        "base_change_cost": 0.25,
        "formula": "base_times_strength_weight",
        "constraint_strength_weights": {
            "": 1.0,
            "locked": 1000.0,
            "booked": 500.0,
            "hard": 100.0,
            "strong": 50.0,
            "soft": 10.0,
            "weak": 5.0,
            "preference": 1.0,
            "test_only": 0.0,
        },
    }


def _valid_time_window_route_evidence(
    value: Any,
    *,
    route_matrix: Any,
    expected_route_legs: tuple[dict[str, Any], ...],
) -> bool:
    fields = {
        "day",
        "origin_id",
        "destination_id",
        "duration_minutes",
        "query_hash",
        "road_validated",
        "fallback_used",
    }
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(expected_route_legs)
    ):
        return False
    pairs: set[tuple[int, str, str]] = set()
    for row, expected in zip(value, expected_route_legs, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or not isinstance(row.get("day"), int)
            or isinstance(row.get("day"), bool)
            or not isinstance(row.get("origin_id"), str)
            or not row["origin_id"]
            or not isinstance(row.get("destination_id"), str)
            or not row["destination_id"]
            or not _valid_nonnegative_finite(row.get("duration_minutes"))
            or row["duration_minutes"] <= 0
            or not isinstance(row.get("query_hash"), str)
            or not row["query_hash"]
            or row.get("road_validated") is not True
            or row.get("fallback_used") is not False
            or (row["day"], row["origin_id"], row["destination_id"])
            != (
                expected.get("day"),
                expected.get("origin_id"),
                expected.get("destination_id"),
            )
        ):
            return False
        cell = route_matrix.cells.get((row["origin_id"], row["destination_id"]))
        if (
            cell is None
            or row["duration_minutes"] != float(cell.duration_s) / 60.0
            or row["query_hash"] != cell.query_hash
            or row["road_validated"] != cell.road_validated
            or row["fallback_used"] != cell.fallback_used
        ):
            return False
        pair = (row["day"], row["origin_id"], row["destination_id"])
        if pair in pairs:
            return False
        pairs.add(pair)
    return True


def _valid_time_window_change(value: Any, cost_policy: dict[str, Any]) -> bool:
    fields = {
        "stop_id",
        "day",
        "from_constraint",
        "to_constraint",
        "from_earliest_arrival",
        "to_earliest_arrival",
        "from_latest_departure",
        "to_latest_departure",
        "owner_strength",
        "cost",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    before = value.get("from_constraint")
    after = value.get("to_constraint")
    if (
        (before is not None and not _canonical_time_window_record(before))
        or not _canonical_time_window_record(after)
        or value.get("from_earliest_arrival")
        != (before or {}).get("earliest_arrival")
        or value.get("to_earliest_arrival") != after.get("earliest_arrival")
        or value.get("from_latest_departure")
        != (before or {}).get("latest_departure")
        or value.get("to_latest_departure") != after.get("latest_departure")
    ):
        return False
    weights = cost_policy["constraint_strength_weights"]
    strength = value.get("owner_strength")
    return bool(
        isinstance(value.get("stop_id"), str)
        and value["stop_id"]
        and isinstance(value.get("day"), int)
        and not isinstance(value.get("day"), bool)
        and strength in weights
        and value.get("cost") == cost_policy["base_change_cost"] * weights[strength]
    )


def _canonical_time_window_record(value: Any) -> bool:
    fields = {
        "schema_version",
        "earliest_arrival",
        "latest_departure",
        "early_arrival_policy",
        "latest_departure_semantics",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    earliest = value.get("earliest_arrival")
    latest = value.get("latest_departure")
    return bool(
        value.get("schema_version") == "stop-time-window-constraint-v1"
        and value.get("early_arrival_policy") == "wait_until_earliest_arrival"
        and value.get("latest_departure_semantics") == "departure_after_visit"
        and (earliest is not None or latest is not None)
        and _valid_clock(earliest)
        and _valid_clock(latest)
        and (
            earliest is None
            or latest is None
            or _clock_minutes(earliest) <= _clock_minutes(latest)
        )
    )


def _valid_time_window_schedule_accounting(
    value: Any,
    *,
    default_visit_minutes: float,
    day_start_minute: float,
    route_leg_evidence: list[dict[str, Any]],
    plan: dict[str, Any],
) -> bool:
    fields = {
        "schema_version",
        "evaluator_version",
        "plan_id",
        "plan_content_hash",
        "route_matrix_id",
        "status",
        "modeled_components",
        "unavailable_components",
        "blocking_codes",
        "missing_opening_window_stop_ids",
        "metrics",
        "stop_traces",
        "content_hash",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    payload = dict(value)
    content_hash = payload.pop("content_hash")
    traces = value.get("stop_traces")
    valid_shape = bool(
        value.get("schema_version")
        == "evaluated-time-window-schedule-accounting-v1"
        and isinstance(value.get("evaluator_version"), str)
        and value["evaluator_version"]
        and isinstance(value.get("plan_id"), str)
        and value["plan_id"]
        and isinstance(value.get("plan_content_hash"), str)
        and value["plan_content_hash"]
        and isinstance(value.get("route_matrix_id"), str)
        and value["route_matrix_id"]
        and value.get("status") in {"failed", "incomplete_evidence", "passed"}
        and value.get("modeled_components")
        == [
            "road_travel",
            "visit_duration",
            "opening_wait_when_window_available",
            "required_time_window_wait",
            "required_latest_departure",
            "day_limit",
        ]
        and value.get("unavailable_components")
        == [
            "parking_dropoff",
            "walking_transfer",
            "queue_wait",
            "service_buffer",
        ]
        and isinstance(value.get("blocking_codes"), list)
        and all(isinstance(code, str) and code for code in value["blocking_codes"])
        and isinstance(value.get("missing_opening_window_stop_ids"), list)
        and all(
            isinstance(stop_id, str) and stop_id
            for stop_id in value["missing_opening_window_stop_ids"]
        )
        and isinstance(value.get("metrics"), dict)
        and all(_valid_finite_number(metric) for metric in value["metrics"].values())
        and isinstance(traces, list)
        and traces
        and all(
            _valid_time_window_trace(
                trace,
                default_visit_minutes=default_visit_minutes,
                day_start_minute=day_start_minute,
                route_leg_evidence=route_leg_evidence,
            )
            for trace in traces
        )
        and isinstance(content_hash, str)
        and content_hash == stable_content_hash(payload)
    )
    if not valid_shape:
        return False
    missing_window_ids = [
        trace["stop_id"]
        for trace in traces
        if trace["opening_start_minute"] is None
        and trace["opening_end_minute"] is None
    ]
    expected_blocking_codes: list[str] = []
    traces_by_day: dict[int, list[dict[str, Any]]] = {}
    for trace in traces:
        traces_by_day.setdefault(trace["day"], []).append(trace)
    for day, day_traces in sorted(traces_by_day.items()):
        for trace in day_traces:
            expected_blocking_codes.extend(trace["failure_codes"])
        if value["metrics"].get(f"day_{day}_overrun_minutes", 0.0) > 0:
            expected_blocking_codes.append("day_time_exceeded")
    expected_blocking_codes = list(dict.fromkeys(expected_blocking_codes))
    expected_status = (
        "failed"
        if expected_blocking_codes
        else "incomplete_evidence"
        if missing_window_ids
        else "passed"
    )
    metrics = value["metrics"]
    route_required = metrics.get("route_required_leg_count")
    return bool(
        value["missing_opening_window_stop_ids"] == sorted(missing_window_ids)
        and value["blocking_codes"] == expected_blocking_codes
        and value["status"] == expected_status
        and _valid_nonnegative_finite(route_required)
        and route_required > 0
        and metrics.get("route_present_leg_count") == route_required
        and metrics.get("route_road_validated_leg_count") == route_required
        and metrics.get("route_missing_leg_count") == 0.0
        and metrics.get("route_fallback_leg_count") == 0.0
        and _time_window_traces_match_plan(
            traces,
            plan,
            default_visit_minutes=default_visit_minutes,
            day_start_minute=day_start_minute,
            route_leg_evidence=route_leg_evidence,
        )
    )


def _valid_time_window_trace(
    value: Any,
    *,
    default_visit_minutes: float,
    day_start_minute: float,
    route_leg_evidence: list[dict[str, Any]],
) -> bool:
    fields = {
        "stop_id",
        "day",
        "sequence_index",
        "incoming_origin_id",
        "incoming_travel_minutes",
        "incoming_route_query_hash",
        "road_arrival_minute",
        "opening_start_minute",
        "opening_end_minute",
        "opening_wait_minutes",
        "opening_window_source",
        "time_window_constraint",
        "time_window_source",
        "required_window_wait_minutes",
        "service_start_minute",
        "visit_minutes",
        "visit_duration_source",
        "departure_minute",
        "latest_departure_status",
        "latest_departure_overrun_minutes",
        "failure_codes",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    numeric_fields = (
        "incoming_travel_minutes",
        "road_arrival_minute",
        "service_start_minute",
        "visit_minutes",
        "departure_minute",
    )
    optional_numeric_fields = (
        "opening_start_minute",
        "opening_end_minute",
        "opening_wait_minutes",
        "required_window_wait_minutes",
        "latest_departure_overrun_minutes",
    )
    if (
        not isinstance(value.get("stop_id"), str)
        or not value["stop_id"]
        or not isinstance(value.get("day"), int)
        or isinstance(value.get("day"), bool)
        or not isinstance(value.get("sequence_index"), int)
        or isinstance(value.get("sequence_index"), bool)
        or value["sequence_index"] < 0
        or not isinstance(value.get("incoming_origin_id"), str)
        or not value["incoming_origin_id"]
        or not isinstance(value.get("incoming_route_query_hash"), str)
        or not value["incoming_route_query_hash"]
        or not isinstance(value.get("opening_window_source"), str)
        or not value["opening_window_source"]
        or any(not _valid_nonnegative_finite(value.get(key)) for key in numeric_fields)
        or any(
            field_value is not None and not _valid_nonnegative_finite(field_value)
            for field_value in (value.get(key) for key in optional_numeric_fields)
        )
        or not isinstance(value.get("failure_codes"), list)
        or any(not isinstance(code, str) or not code for code in value["failure_codes"])
    ):
        return False
    constraint = value.get("time_window_constraint")
    if constraint is None:
        if (
            value.get("time_window_source") is not None
            or value.get("required_window_wait_minutes") is not None
            or value.get("latest_departure_status") != "not_set"
            or value.get("latest_departure_overrun_minutes") is not None
        ):
            return False
    elif (
        not _canonical_time_window_record(constraint)
        or value.get("time_window_source") != "trip_specific_user_constraint"
    ):
        return False
    arrival = float(value["road_arrival_minute"])
    opening_start = value.get("opening_start_minute")
    opening_end = value.get("opening_end_minute")
    opening_wait = value.get("opening_wait_minutes")
    expected_opening_wait = (
        None
        if opening_start is None and opening_end is None
        else max(0.0, float(opening_start) - arrival)
        if opening_start is not None
        else 0.0
    )
    if opening_wait != expected_opening_wait:
        return False
    opening_ready = arrival + (float(opening_wait) if opening_wait is not None else 0.0)
    earliest = (constraint or {}).get("earliest_arrival")
    expected_required_wait = (
        None
        if earliest is None
        else max(0.0, float(_clock_minutes(earliest)) - opening_ready)
    )
    if value.get("required_window_wait_minutes") != expected_required_wait:
        return False
    expected_service_start = opening_ready + (
        float(expected_required_wait) if expected_required_wait is not None else 0.0
    )
    if value.get("service_start_minute") != expected_service_start:
        return False
    duration = float(value["visit_minutes"])
    duration_source = value.get("visit_duration_source")
    if duration_source == "configured_evaluator_default":
        if duration != default_visit_minutes:
            return False
    elif duration_source not in {
        "visit_duration_minutes",
        "duration_minutes",
        "service_minutes",
        "dwell_minutes",
    }:
        return False
    departure = expected_service_start + duration
    if value.get("departure_minute") != departure:
        return False
    latest = (constraint or {}).get("latest_departure")
    expected_overrun = (
        None
        if latest is None
        else max(0.0, departure - float(_clock_minutes(latest)))
    )
    expected_latest_status = (
        "not_set"
        if latest is None
        else "violated"
        if expected_overrun and expected_overrun > 0
        else "satisfied"
    )
    expected_failures: list[str] = []
    if opening_end is not None and expected_service_start > float(opening_end):
        expected_failures.append("opening_window_missed")
    if expected_latest_status == "violated":
        expected_failures.append("stop_time_window_latest_departure_exceeded")
    return bool(
        value.get("latest_departure_status") == expected_latest_status
        and value.get("latest_departure_overrun_minutes") == expected_overrun
        and value.get("failure_codes") == expected_failures
    )


def _time_window_changes_match_traces(
    changes: list[dict[str, Any]],
    parent_traces: list[dict[str, Any]],
    child_traces: list[dict[str, Any]],
) -> bool:
    parent_by_stop = {trace["stop_id"]: trace for trace in parent_traces}
    child_by_stop = {trace["stop_id"]: trace for trace in child_traces}
    return all(
        change["stop_id"] in parent_by_stop
        and change["stop_id"] in child_by_stop
        and parent_by_stop[change["stop_id"]]["time_window_constraint"]
        == change["from_constraint"]
        and parent_by_stop[change["stop_id"]]["day"] == change["day"]
        and child_by_stop[change["stop_id"]]["time_window_constraint"]
        == change["to_constraint"]
        and child_by_stop[change["stop_id"]]["day"] == change["day"]
        for change in changes
    )


def _time_window_traces_match_plan(
    traces: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    default_visit_minutes: float,
    day_start_minute: float,
    route_leg_evidence: list[dict[str, Any]],
) -> bool:
    selected: dict[str, dict[str, Any]] = {}
    for row in plan.get("selected_stops") or ():
        if not isinstance(row, dict):
            return False
        stop_id = str(row.get("stop_id") or row.get("poi_id") or "")
        if not stop_id or stop_id in selected:
            return False
        selected[stop_id] = row
    ordered_day: dict[str, int] = {}
    for row in plan.get("ordered_days") or ():
        if not isinstance(row, dict):
            return False
        day = row.get("day")
        stop_ids = row.get("stop_ids")
        if (
            not isinstance(day, int)
            or isinstance(day, bool)
            or not isinstance(stop_ids, list)
        ):
            return False
        for stop_id in stop_ids:
            ordered_day[str(stop_id)] = day
    grouped: dict[int, list[str]] = {}
    assignments = plan.get("day_assignments") or {}
    for stop_id in plan.get("sequence") or ():
        stop_id = str(stop_id)
        day = assignments.get(stop_id)
        if not isinstance(day, int) or isinstance(day, bool):
            day = ordered_day.get(stop_id)
        if not isinstance(day, int) or isinstance(day, bool):
            return False
        grouped.setdefault(day, []).append(stop_id)
    expected_order = [
        (stop_id, day, index)
        for day, stop_ids in sorted(grouped.items())
        for index, stop_id in enumerate(stop_ids)
    ]
    actual_order = [
        (trace["stop_id"], trace["day"], trace["sequence_index"])
        for trace in traces
    ]
    if actual_order != expected_order or set(selected) != {row[0] for row in expected_order}:
        return False
    incoming_by_stop: dict[tuple[int, str], dict[str, Any]] = {}
    for leg in route_leg_evidence:
        key = (leg["day"], leg["destination_id"])
        if key in incoming_by_stop:
            return False
        incoming_by_stop[key] = leg
    prior_departure_by_day: dict[int, float] = {}
    for trace in traces:
        stop = selected[trace["stop_id"]]
        incoming = incoming_by_stop.get((trace["day"], trace["stop_id"]))
        if incoming is None:
            return False
        expected_current = prior_departure_by_day.get(trace["day"], day_start_minute)
        expected_arrival = expected_current + float(incoming["duration_minutes"])
        if (
            trace["incoming_origin_id"] != incoming["origin_id"]
            or trace["incoming_travel_minutes"] != incoming["duration_minutes"]
            or trace["incoming_route_query_hash"] != incoming["query_hash"]
            or trace["road_arrival_minute"] != expected_arrival
        ):
            return False
        prior_departure_by_day[trace["day"]] = float(trace["departure_minute"])
        if trace["time_window_constraint"] != stop.get("time_window_constraint"):
            return False
        expected_opening_start, expected_opening_end, expected_opening_source = (
            _plan_opening_window(stop)
        )
        if (
            trace["opening_start_minute"] != expected_opening_start
            or trace["opening_end_minute"] != expected_opening_end
            or trace["opening_window_source"] != expected_opening_source
        ):
            return False
        duration_values = [
            (key, stop.get(key))
            for key in (
                "visit_duration_minutes",
                "duration_minutes",
                "service_minutes",
                "dwell_minutes",
            )
            if stop.get(key) is not None
        ]
        if duration_values:
            duration_source, duration_value = duration_values[0]
            if (
                not _valid_duration_minutes(duration_value)
                or trace["visit_duration_source"] != duration_source
                or trace["visit_minutes"] != float(duration_value)
            ):
                return False
        elif (
            trace["visit_duration_source"] != "configured_evaluator_default"
            or trace["visit_minutes"] != default_visit_minutes
        ):
            return False
    return True


def _plan_opening_window(
    stop: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    start_key, start_value = _first_plan_field(
        stop,
        "opening_start",
        "window_start",
        "start_time",
        "earliest_start",
        "open_time",
    )
    end_key, end_value = _first_plan_field(
        stop,
        "opening_end",
        "window_end",
        "latest_start",
        "close_time",
    )
    start = _plan_clock_minute(start_value)
    end = _plan_clock_minute(end_value)
    if start_key is None and end_key is None:
        return None, None, "unavailable"
    return (
        start,
        end,
        f"plan_stop_fields:{start_key or 'none'}:{end_key or 'none'}",
    )


def _first_plan_field(
    stop: dict[str, Any],
    *keys: str,
) -> tuple[str | None, Any]:
    for key in keys:
        value = stop.get(key)
        if value is not None and str(value).strip():
            return key, value
    return None, None


def _plan_clock_minute(value: Any) -> float | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            hours, minutes = text.split(":", 1)
            result = int(hours) * 60 + int(minutes[:2])
        else:
            result = int(float(text))
    except (TypeError, ValueError):
        return None
    return float(result)


def _valid_clock(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is not None
    )


def _clock_minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _valid_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_nonnegative_finite(value: Any) -> bool:
    return _valid_finite_number(value) and float(value) >= 0


def _valid_clock_minute_number(value: Any) -> bool:
    return _valid_nonnegative_finite(value) and float(value) < 1440


def _valid_duration_change(
    value: Any,
    default_visit_minutes: float,
    cost_policy: dict[str, Any],
) -> bool:
    fields = {
        "stop_id",
        "day",
        "from_constraint",
        "to_constraint",
        "from_minutes",
        "to_minutes",
        "from_source",
        "to_source",
        "accounting_from_minutes",
        "accounting_to_minutes",
        "accounting_from_source",
        "accounting_to_source",
        "delta_minutes",
        "owner_strength",
        "cost",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    to_constraint = value.get("to_constraint")
    if not _canonical_exact_duration_record(to_constraint):
        return False
    to_minutes = float(to_constraint["preferred_minutes"])
    if (
        value.get("to_minutes") != to_minutes
        or value.get("to_source") != "visit_duration_minutes"
        or value.get("accounting_to_minutes") != to_minutes
        or value.get("accounting_to_source") != "visit_duration_minutes"
    ):
        return False
    from_constraint = value.get("from_constraint")
    from_minutes = value.get("from_minutes")
    from_source = value.get("from_source")
    accounting_from = value.get("accounting_from_minutes")
    accounting_source = value.get("accounting_from_source")
    if from_constraint is None and from_minutes is None:
        if (
            from_source != "unavailable"
            or accounting_from != default_visit_minutes
            or accounting_source != "configured_evaluator_default"
        ):
            return False
    else:
        if from_constraint is not None and not _canonical_exact_duration_record(from_constraint):
            return False
        if (
            not _valid_duration_minutes(from_minutes)
            or from_source
            not in {
                "visit_duration_minutes",
                "duration_minutes",
                "service_minutes",
                "dwell_minutes",
            }
            or accounting_from != from_minutes
            or accounting_source != from_source
        ):
            return False
        if (
            from_constraint is not None
            and from_constraint["preferred_minutes"] != from_minutes
        ):
            return False
    expected_delta = to_minutes - float(accounting_from)
    strength_weights = cost_policy["constraint_strength_weights"]
    owner_strength = value.get("owner_strength")
    if owner_strength not in strength_weights:
        return False
    expected_cost = (
        cost_policy["base_change_cost"]
        * strength_weights[owner_strength]
        * max(1.0, abs(expected_delta) / 60.0)
    )
    return (
        value.get("delta_minutes") == expected_delta
        and isinstance(value.get("stop_id"), str)
        and bool(value["stop_id"])
        and isinstance(value.get("day"), int)
        and not isinstance(value.get("day"), bool)
        and value.get("cost") == expected_cost
    )


def _valid_duration_cost_policy(value: Any) -> bool:
    return value == {
        "base_change_cost": 0.25,
        "formula": "base_times_strength_weight_times_max_1_abs_delta_minutes_over_60",
        "constraint_strength_weights": {
            "": 1.0,
            "locked": 1000.0,
            "booked": 500.0,
            "hard": 100.0,
            "strong": 50.0,
            "soft": 10.0,
            "weak": 5.0,
            "preference": 1.0,
            "test_only": 0.0,
        },
    }


def _canonical_exact_duration_record(value: Any) -> bool:
    fields = {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
    if not isinstance(value, dict) or set(value) != fields or value.get("mode") != "exact":
        return False
    preferred = value.get("preferred_minutes")
    return (
        isinstance(preferred, int)
        and not isinstance(preferred, bool)
        and 15 <= preferred <= 480
        and value.get("minimum_minutes") == preferred
        and value.get("maximum_minutes") == preferred
    )


def _valid_duration_minutes(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 15 <= float(value) <= 480
    )


def _route_projection(plan: dict[str, Any]) -> dict[str, Any]:
    stops = []
    for stop in plan.get("selected_stops") or ():
        if not isinstance(stop, dict):
            return {}
        stops.append(
            {
                "stop_id": str(stop.get("stop_id") or stop.get("poi_id") or ""),
                "day": stop.get("day"),
                "order": stop.get("order"),
                "latitude": stop.get("latitude"),
                "longitude": stop.get("longitude"),
            }
        )
    return {
        "sequence": list(plan.get("sequence") or ()),
        "selected_stops": stops,
        "lodging_assignments": dict(plan.get("lodging_assignments") or {}),
    }


def _relineage_geography_plan(
    source: dict[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    plan = deepcopy(source)
    source_plan_id = str(plan.get("plan_id") or "")
    child_plan_id = str(child.get("plan_id") or "")
    child_hash = str(child.get("content_hash") or "")
    plan["plan_id"] = child_plan_id
    plan["content_hash"] = child_hash
    role_by_stop: dict[str, tuple[str | None, str]] = {}
    for stop in child.get("selected_stops") or ():
        if not isinstance(stop, dict):
            raise WorkspaceError("draft_geography_evidence_mismatch", 409)
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or "")
        role = stop.get("itinerary_role")
        source = stop.get("itinerary_role_source")
        if role is None:
            if source not in {None, "unavailable"}:
                raise WorkspaceError("draft_geography_evidence_mismatch", 409)
            source = "unavailable"
        elif role not in ALL_ROLE_VALUES or source != "user_declared_itinerary_role":
            raise WorkspaceError("draft_geography_evidence_mismatch", 409)
        role_by_stop[stop_id] = (role, source)
    for collection_name in ("stops", "route_path", "validated_legs", "gaps"):
        for feature in plan[collection_name]["features"]:
            feature_id = str(feature.get("id") or "")
            if source_plan_id and feature_id.startswith(f"{source_plan_id}:"):
                feature["id"] = f"{child_plan_id}:{feature_id.removeprefix(f'{source_plan_id}:')}"
            feature["properties"]["plan_id"] = child_plan_id
            feature["properties"]["content_hash"] = child_hash
            if collection_name == "stops":
                stop_id = str(feature["properties"].get("stop_id") or "")
                role, source = role_by_stop.get(stop_id, (None, "unavailable"))
                feature["properties"]["itinerary_role"] = role
                feature["properties"]["itinerary_role_source"] = source
    return plan


def _route_endpoint(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise WorkspaceError("selected_route_segment_invalid", 409)
    longitude, latitude = value
    if (
        isinstance(longitude, bool)
        or isinstance(latitude, bool)
        or not isinstance(longitude, (int, float))
        or not isinstance(latitude, (int, float))
        or not math.isfinite(float(longitude))
        or not math.isfinite(float(latitude))
        or not -180 <= float(longitude) <= 180
        or not -90 <= float(latitude) <= 90
    ):
        raise WorkspaceError("selected_route_segment_invalid", 409)
    return float(latitude), float(longitude)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
