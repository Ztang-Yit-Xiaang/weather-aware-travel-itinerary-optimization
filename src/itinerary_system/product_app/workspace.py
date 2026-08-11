"""Persistent server-authoritative sessions with W5 decisions disabled."""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .interaction_contracts import (
    TYPED_EDIT_OPERATION_NAMES,
    AttributeConstraintNameV1,
    CommitmentStrengthV1,
    ConstraintScopeLifetimeV1,
    DraftOperationTypeV2,
    DurationModeV1,
    RouteIssueTypeV1,
    StopRoleV1,
)
from .models import DraftOperationV1, ProductSession, utc_now
from .persistence import SessionSnapshotError, SessionSnapshotStore, StateLockTimeoutError


class WorkspaceError(ValueError):
    def __init__(self, code: str, status_code: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class WorkspaceStore:
    CONFIRMED_MAP_OPERATIONS = {
        "add_custom_waypoint",
        "add_route_waypoint",
        "relocate_custom_waypoint",
    }
    KNOWN_OPERATIONS = {
        "keep_stop",
        "lock_stop",
        "mark_flexible",
        "move_day",
        "route_feedback",
        "replace_nearby",
        "add_candidate",
        *TYPED_EDIT_OPERATION_NAMES,
        *CONFIRMED_MAP_OPERATIONS,
    }
    ALLOWED_OPERATIONS = KNOWN_OPERATIONS
    ROUTE_FEEDBACK_PREFERENCES = {
        "reduce_contextual_risk",
        "reduce_driving",
        "preserve_experience",
    }
    MAX_SESSIONS = 128
    MAX_DRAFT_OPERATIONS = 32
    MAX_PERMISSION_DECISIONS = 32
    SESSION_TTL = timedelta(hours=24)
    _CONVERSATION_ID = re.compile(r"conversation_[0-9a-f]{32}\Z")

    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self._sessions: dict[str, ProductSession] = {}
        self._lock = threading.RLock()
        self._snapshots = SessionSnapshotStore(state_root)
        self._restoration_errors: dict[str, str] = {}
        self._restore()

    @property
    def restoration_errors(self) -> dict[str, str]:
        return dict(self._restoration_errors)

    def create_session(
        self,
        run_id: str,
        parent_plan_id: str,
        selected_day: int | None,
        trip_id: str | None = None,
    ) -> tuple[ProductSession, str]:
        with self._lock:
            try:
                with self._snapshots.locked():
                    self._refresh_locked()
                    self._purge_expired_locked()
                    if len(self._sessions) >= self.MAX_SESSIONS:
                        raise WorkspaceError("session_capacity_reached", 429)
                    raw_token = secrets.token_urlsafe(24)
                    salt = secrets.token_hex(16)
                    session = ProductSession(
                        session_id=f"session_{uuid4().hex}",
                        mutation_token_salt=salt,
                        mutation_token_verifier=self._token_digest(raw_token, salt),
                        trip_id=trip_id or run_id,
                        run_id=run_id,
                        revision=0,
                        accepted_plan_id=parent_plan_id,
                        selected_day=selected_day,
                    )
                    self._snapshots.save(session)
                    self._sessions[session.session_id] = session
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise WorkspaceError(exc.code, 503) from None
        return session, raw_token

    def get(self, session_id: str) -> ProductSession:
        with self._lock:
            if session_id in self._restoration_errors:
                raise WorkspaceError(self._restoration_errors[session_id], 409)
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise WorkspaceError("unknown_session", 404) from exc
            if self._is_expired(session):
                try:
                    with self._snapshots.locked():
                        current = self._load_current_locked(session_id)
                        if self._is_expired(current):
                            self._snapshots.delete(session_id)
                            self._sessions.pop(session_id, None)
                            raise WorkspaceError("session_expired", 404)
                        return self._install_locked(current)
                except StateLockTimeoutError:
                    raise WorkspaceError("state_lock_timeout", 503) from None
                except SessionSnapshotError as exc:
                    raise self._snapshot_workspace_error(exc) from None
            self.reduce_confirmed_map_draft(session.draft)
            return session

    def authenticate(self, session_id: str, token: str | None) -> ProductSession:
        if not token:
            raise WorkspaceError("invalid_session_token", 403)
        with self._lock:
            try:
                with self._snapshots.locked():
                    session = self._load_current_locked(session_id)
                    if self._is_expired(session):
                        self._snapshots.delete(session_id)
                        self._sessions.pop(session_id, None)
                        raise WorkspaceError("session_expired", 404)
                    candidate = self._token_digest(token, session.mutation_token_salt)
                    if not secrets.compare_digest(session.mutation_token_verifier, candidate):
                        raise WorkspaceError("invalid_session_token", 403)
                    session.last_accessed_at = utc_now()
                    self._snapshots.save(session)
                    return self._install_locked(session)
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise self._snapshot_workspace_error(exc) from None

    def authenticate_readonly(
        self,
        session_id: str,
        token: str | None,
    ) -> ProductSession:
        """Authenticate without touching persisted bytes before semantic validation."""

        if not token:
            raise WorkspaceError("invalid_session_token", 403)
        with self._lock:
            try:
                with self._snapshots.locked():
                    session = self._load_current_locked(session_id)
                    if self._is_expired(session):
                        self._snapshots.delete(session_id)
                        self._sessions.pop(session_id, None)
                        raise WorkspaceError("session_expired", 404)
                    candidate = self._token_digest(
                        token,
                        session.mutation_token_salt,
                    )
                    if not secrets.compare_digest(
                        session.mutation_token_verifier,
                        candidate,
                    ):
                        raise WorkspaceError("invalid_session_token", 403)
                    return self._install_locked(session)
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise self._snapshot_workspace_error(exc) from None

    def bind_conversation_id(self, session: ProductSession, conversation_id: str) -> None:
        """Persist a conversation owner without advancing itinerary revision."""

        if not self._CONVERSATION_ID.fullmatch(conversation_id):
            raise WorkspaceError("invalid_conversation_id", 422)
        with self._lock:
            try:
                with self._snapshots.locked():
                    current = self._load_current_locked(session.session_id)
                    if self._is_expired(current):
                        raise WorkspaceError("session_expired", 404)
                    if current.conversation_id not in {None, conversation_id}:
                        raise WorkspaceError("conversation_session_mismatch", 409)
                    current.conversation_id = conversation_id
                    current.last_accessed_at = utc_now()
                    self._snapshots.save(current)
                    self._sync_session(session, current)
                    self._sessions[current.session_id] = session
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise self._snapshot_workspace_error(exc) from None

    def clear_conversation_id(
        self,
        session: ProductSession,
        expected_revision: Any,
        conversation_id: str,
    ) -> None:
        """Detach only the authenticated session's deleted conversation."""

        with self._lock:
            try:
                with self._snapshots.locked():
                    current = self._load_current_locked(session.session_id)
                    self.expect_revision(current, expected_revision)
                    if current.conversation_id not in {None, conversation_id}:
                        raise WorkspaceError("conversation_session_mismatch", 409)
                    current.conversation_id = None
                    current.last_accessed_at = utc_now()
                    self._snapshots.save(current)
                    self._sync_session(session, current)
                    self._sessions[current.session_id] = session
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise self._snapshot_workspace_error(exc) from None

    @staticmethod
    def expect_revision(session: ProductSession, expected: Any) -> None:
        if not isinstance(expected, int) or expected != session.revision:
            raise WorkspaceError("stale_session_revision", 409)

    def select(self, session: ProductSession, payload: dict[str, Any]) -> ProductSession:
        expected = payload.get("expected_revision")

        def mutate(current: ProductSession) -> None:
            current.selected_day = self._optional_int(payload.get("selected_day"))
            current.selected_stop_id = self._optional_text(payload.get("selected_stop_id"))
            current.selected_segment_id = self._optional_text(payload.get("selected_segment_id"))
            current.selected_candidate_id = self._optional_text(payload.get("selected_candidate_id"))
            current.selected_alternative_id = self._optional_text(payload.get("selected_alternative_id"))

        self._mutate(session, expected, mutate)
        return session

    def add_operation(
        self,
        session: ProductSession,
        payload: dict[str, Any],
        *,
        valid_stop_ids: set[str],
        day_count: int,
        parent_stop_ids: set[str] | None = None,
        candidate_ids: set[str] | None = None,
        candidate_replacements: dict[str, str] | None = None,
        route_leg_ids: set[str] | None = None,
        protected_stop_ids: set[str] | None = None,
        parent_day_by_stop: dict[str, int] | None = None,
        parent_order_by_day: dict[int, tuple[str, ...]] | None = None,
        parent_duration_by_stop: dict[str, dict[str, Any]] | None = None,
        parent_time_window_by_stop: dict[str, dict[str, Any] | None] | None = None,
        parent_role_by_stop: dict[str, dict[str, Any]] | None = None,
        role_constraints_by_stop: dict[str, tuple[dict[str, Any], ...]] | None = None,
    ) -> DraftOperationV1:
        operation_type = str(payload.get("type") or "")
        target = str(payload.get("target") or "")
        parameters = payload.get("parameters")
        if operation_type not in self.KNOWN_OPERATIONS:
            raise WorkspaceError("unsupported_draft_operation", 422)
        if operation_type in self.CONFIRMED_MAP_OPERATIONS:
            raise WorkspaceError("map_operation_confirmation_required", 409)
        if not isinstance(parameters, dict):
            raise WorkspaceError("invalid_draft_parameters", 422)
        parent_targets = parent_stop_ids if parent_stop_ids is not None else valid_stop_ids
        candidates = candidate_ids if candidate_ids is not None else valid_stop_ids
        self._validate_operation_domain(
            operation_type,
            target,
            parameters,
            parent_targets=parent_targets,
            candidate_ids=candidates,
            candidate_replacements=candidate_replacements,
            day_count=day_count,
            route_leg_ids=route_leg_ids or set(),
            protected_stop_ids=protected_stop_ids or set(),
            parent_day_by_stop=parent_day_by_stop or {},
            parent_order_by_day=parent_order_by_day or {},
            parent_duration_by_stop=parent_duration_by_stop or {},
            parent_time_window_by_stop=parent_time_window_by_stop or {},
            parent_role_by_stop=parent_role_by_stop or {},
            role_constraints_by_stop=role_constraints_by_stop or {},
        )
        evidence_refs = self._evidence_refs(payload.get("evidence_refs"))
        operation: DraftOperationV1 | None = None

        def mutate(current: ProductSession) -> None:
            nonlocal operation
            if len(current.draft) >= self.MAX_DRAFT_OPERATIONS:
                raise WorkspaceError("draft_operation_limit_reached", 409)
            self.validate_draft_domain(
                current.draft,
                parent_targets=parent_targets,
                candidate_ids=candidates,
                candidate_replacements=candidate_replacements,
                day_count=day_count,
                route_leg_ids=route_leg_ids or set(),
                protected_stop_ids=protected_stop_ids or set(),
                parent_day_by_stop=parent_day_by_stop or {},
                parent_order_by_day=parent_order_by_day or {},
                parent_duration_by_stop=parent_duration_by_stop or {},
                parent_time_window_by_stop=parent_time_window_by_stop or {},
                parent_role_by_stop=parent_role_by_stop or {},
                role_constraints_by_stop=role_constraints_by_stop or {},
            )
            operation = DraftOperationV1(
                operation_id=f"operation_{uuid4().hex}",
                type=operation_type,
                target=target,
                parameters=parameters,
                source=str(payload.get("source") or "map"),
                evidence_refs=evidence_refs,
            )
            self._validate_draft_conflicts((*current.draft, operation))
            current.draft.append(operation)
            current.proposal = None

        self._mutate(session, payload.get("expected_revision"), mutate)
        assert operation is not None
        return operation

    def append_confirmed_map_operation(
        self,
        session: ProductSession,
        *,
        expected_revision: Any,
        operation_type: str,
        target: str,
        parameters: dict[str, Any],
        evidence_refs: tuple[str, ...],
    ) -> DraftOperationV1:
        """Append one server-resolved map operation through the session CAS path."""

        if operation_type not in self.CONFIRMED_MAP_OPERATIONS:
            raise WorkspaceError("unsupported_confirmed_map_operation", 422)
        self._validate_confirmed_map_parameters(operation_type, target, parameters, evidence_refs)
        operation: DraftOperationV1 | None = None

        def mutate(current: ProductSession) -> None:
            nonlocal operation
            if len(current.draft) >= self.MAX_DRAFT_OPERATIONS:
                raise WorkspaceError("draft_operation_limit_reached", 409)
            operation = DraftOperationV1(
                operation_id=f"operation_{uuid4().hex}",
                type=operation_type,
                target=target,
                parameters=parameters,
                source="confirmed_map_interaction",
                evidence_refs=evidence_refs,
            )
            current.draft.append(operation)
            self.reduce_confirmed_map_draft(current.draft)
            current.proposal = None

        self._mutate(session, expected_revision, mutate)
        assert operation is not None
        return operation

    @classmethod
    def validate_draft_domain(
        cls,
        operations: list[DraftOperationV1] | tuple[DraftOperationV1, ...],
        *,
        parent_targets: set[str],
        candidate_ids: set[str],
        candidate_replacements: dict[str, str] | None,
        day_count: int,
        route_leg_ids: set[str],
        protected_stop_ids: set[str],
        parent_day_by_stop: dict[str, int],
        parent_order_by_day: dict[int, tuple[str, ...]],
        parent_duration_by_stop: dict[str, dict[str, Any]] | None = None,
        parent_time_window_by_stop: dict[str, dict[str, Any] | None] | None = None,
        parent_role_by_stop: dict[str, dict[str, Any]] | None = None,
        role_constraints_by_stop: dict[str, tuple[dict[str, Any], ...]] | None = None,
    ) -> None:
        """Revalidate persisted operations against the current artifact universe."""

        cls.reduce_confirmed_map_draft(operations)
        for operation in operations:
            if operation.type in cls.CONFIRMED_MAP_OPERATIONS:
                continue
            if operation.type not in cls.KNOWN_OPERATIONS:
                raise WorkspaceError("unsupported_draft_operation", 422)
            if not isinstance(operation.parameters, dict):
                raise WorkspaceError("invalid_draft_parameters", 422)
            cls._validate_operation_domain(
                operation.type,
                operation.target,
                operation.parameters,
                parent_targets=parent_targets,
                candidate_ids=candidate_ids,
                candidate_replacements=candidate_replacements,
                day_count=day_count,
                route_leg_ids=route_leg_ids,
                protected_stop_ids=protected_stop_ids,
                parent_day_by_stop=parent_day_by_stop,
                parent_order_by_day=parent_order_by_day,
                parent_duration_by_stop=parent_duration_by_stop or {},
                parent_time_window_by_stop=parent_time_window_by_stop or {},
                parent_role_by_stop=parent_role_by_stop or {},
                role_constraints_by_stop=role_constraints_by_stop or {},
            )
        cls._validate_draft_conflicts(tuple(operations))

    @classmethod
    def reduce_confirmed_map_draft(
        cls,
        operations: list[DraftOperationV1] | tuple[DraftOperationV1, ...],
    ) -> dict[str, dict[str, Any]]:
        """Validate and reduce server-confirmed map operations at every trust boundary."""

        resolved: dict[str, dict[str, Any]] = {}
        try:
            for operation in operations:
                if operation.type not in cls.CONFIRMED_MAP_OPERATIONS:
                    continue
                if operation.source != "confirmed_map_interaction":
                    raise WorkspaceError("invalid_confirmed_map_operation", 422)
                cls._validate_confirmed_map_parameters(
                    operation.type,
                    operation.target,
                    operation.parameters,
                    operation.evidence_refs,
                )
                parameters = operation.parameters
                if operation.type in {"add_custom_waypoint", "add_route_waypoint"}:
                    if operation.target in resolved:
                        raise WorkspaceError("invalid_confirmed_map_operation", 422)
                    resolved[operation.target] = parameters
                    continue
                previous = resolved.get(operation.target)
                if previous is None or any(
                    parameters[field] != previous[field]
                    for field in ("name", "role", "duration", "day", "insertion")
                ):
                    raise WorkspaceError("invalid_confirmed_map_operation", 422)
                resolved[operation.target] = parameters
        except (KeyError, TypeError, WorkspaceError):
            raise WorkspaceError("confirmed_map_draft_invalid", 409) from None
        return resolved

    @classmethod
    def _validate_confirmed_map_parameters(
        cls,
        operation_type: str,
        target: str,
        parameters: dict[str, Any],
        evidence_refs: tuple[str, ...],
    ) -> None:
        required = {
            "schema_version",
            "snap_preview_id",
            "waypoint_id",
            "name",
            "day",
            "role",
            "duration",
            "raw_coordinate",
            "snapped_coordinate",
            "selected_access_point",
            "insertion",
            "affected_route_legs",
        }
        if set(parameters) != required or parameters.get("schema_version") != "confirmed-map-operation-v1":
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        waypoint_id = parameters.get("waypoint_id")
        if (
            not isinstance(target, str)
            or target != waypoint_id
            or re.fullmatch(r"waypoint_[0-9a-f]{32}", target) is None
            or re.fullmatch(r"snap_[0-9a-f]{32}", str(parameters.get("snap_preview_id") or "")) is None
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        name = parameters.get("name")
        day = parameters.get("day")
        if (
            not isinstance(name, str)
            or name != name.strip()
            or not 1 <= len(name) <= 120
            or isinstance(day, bool)
            or not isinstance(day, int)
            or day < 1
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        role = parameters.get("role")
        duration = parameters.get("duration")
        if operation_type == "add_route_waypoint" or (
            operation_type == "relocate_custom_waypoint" and role == "route_waypoint"
        ):
            if role != "route_waypoint" or duration is not None:
                raise WorkspaceError("invalid_confirmed_map_operation", 422)
        else:
            if role not in {"attraction", "activity", "meal", "rest_stop", "scenic_stop"}:
                raise WorkspaceError("invalid_confirmed_map_operation", 422)
            cls._validate_duration(duration)
        raw_coordinate = parameters.get("raw_coordinate")
        snapped_coordinate = parameters.get("snapped_coordinate")
        if not cls._valid_coordinate(raw_coordinate) or not cls._valid_coordinate(snapped_coordinate):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        access = parameters.get("selected_access_point")
        if (
            not isinstance(access, dict)
            or set(access)
            != {
                "access_point_id",
                "access_type",
                "coordinate",
                "source",
                "road_validated",
                "access_confidence",
                "evidence_refs",
            }
            or re.fullmatch(r"access_[0-9a-f]{32}", str(access.get("access_point_id") or "")) is None
            or access.get("access_type") != "road_snap"
            or access.get("source") != "runtime_osrm"
            or access.get("road_validated") is not True
            or access.get("access_confidence") != "road_snap_only"
            or access.get("coordinate") != snapped_coordinate
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        insertion = parameters.get("insertion")
        if (
            not isinstance(insertion, dict)
            or set(insertion) != {"route_leg_id", "predecessor_id", "successor_id", "travel_mode"}
            or insertion.get("travel_mode") != "driving"
            or any(
                not isinstance(insertion.get(field), str) or not insertion[field]
                for field in ("route_leg_id", "predecessor_id", "successor_id")
            )
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        legs = parameters.get("affected_route_legs")
        if not isinstance(legs, list) or len(legs) != 2:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        query_refs: list[str] = []
        for leg in legs:
            query_refs.append(cls._validate_confirmed_route_leg(leg))
        if legs[0]["origin_id"] != insertion["predecessor_id"] or legs[1]["destination_id"] != insertion["successor_id"]:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        if legs[0]["destination_id"] != target or legs[1]["origin_id"] != target:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        snapped_point = [snapped_coordinate["longitude"], snapped_coordinate["latitude"]]
        if (
            legs[0]["geometry"]["coordinates"][-1] != snapped_point
            or legs[1]["geometry"]["coordinates"][0] != snapped_point
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        if tuple(query_refs) != tuple(evidence_refs):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        access_refs = access.get("evidence_refs")
        if not isinstance(access_refs, (list, tuple)) or tuple(access_refs) != tuple(evidence_refs):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)

    @staticmethod
    def _validate_duration(value: Any) -> None:
        fields = {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
        if not isinstance(value, dict) or set(value) != fields:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        mode = value.get("mode")
        preferred = value.get("preferred_minutes")
        minimum = value.get("minimum_minutes")
        maximum = value.get("maximum_minutes")

        def valid_minutes(item: Any) -> bool:
            return isinstance(item, int) and not isinstance(item, bool) and 15 <= item <= 480

        valid = (
            mode == "exact"
            and valid_minutes(preferred)
            and preferred == minimum == maximum
        ) or (
            mode == "preferred"
            and valid_minutes(preferred)
            and minimum is None
            and maximum is None
        ) or (
            mode == "minimum"
            and preferred is None
            and valid_minutes(minimum)
            and maximum is None
        ) or (
            mode == "maximum"
            and preferred is None
            and minimum is None
            and valid_minutes(maximum)
        ) or (
            mode == "range"
            and preferred is None
            and valid_minutes(minimum)
            and valid_minutes(maximum)
            and minimum <= maximum
        )
        if not valid:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)

    @staticmethod
    def _valid_coordinate(value: Any) -> bool:
        if not isinstance(value, dict) or set(value) != {"latitude", "longitude"}:
            return False
        latitude = value.get("latitude")
        longitude = value.get("longitude")
        return bool(
            isinstance(latitude, (int, float))
            and not isinstance(latitude, bool)
            and math.isfinite(latitude)
            and -90 <= latitude <= 90
            and isinstance(longitude, (int, float))
            and not isinstance(longitude, bool)
            and math.isfinite(longitude)
            and -180 <= longitude <= 180
        )

    @staticmethod
    def _validate_confirmed_route_leg(value: Any) -> str:
        fields = {
            "route_leg_id",
            "origin_id",
            "destination_id",
            "travel_mode",
            "validation_status",
            "geometry",
            "distance_m",
            "duration_s",
            "provider",
            "routing_status",
            "geometry_source",
            "distance_source",
            "duration_source",
            "road_validated",
            "fallback_used",
            "query_hash",
            "evidence_refs",
            "retrieved_at",
            "snap_distance_origin_m",
            "snap_distance_destination_m",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        query_hash = value.get("query_hash")
        geometry = value.get("geometry")
        if (
            not isinstance(value.get("origin_id"), str)
            or not value["origin_id"]
            or len(value["origin_id"]) > 128
            or not isinstance(value.get("destination_id"), str)
            or not value["destination_id"]
            or len(value["destination_id"]) > 128
            or value.get("travel_mode") != "driving"
            or value.get("validation_status") != "road_validated"
            or value.get("provider") != "runtime_osrm"
            or value.get("routing_status") != "osrm_route_validated"
            or value.get("geometry_source") != "runtime_osrm_geojson"
            or value.get("distance_source") != "runtime_osrm_route"
            or value.get("duration_source") != "runtime_osrm_route"
            or value.get("road_validated") is not True
            or value.get("fallback_used") is not False
            or not isinstance(query_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", query_hash) is None
            or value.get("route_leg_id") != f"preview_leg_{query_hash[:16]}"
            or value.get("evidence_refs") != [f"route_query:{query_hash}"]
            or not isinstance(geometry, dict)
            or geometry.get("type") != "LineString"
            or not isinstance(geometry.get("coordinates"), list)
            or len(geometry["coordinates"]) < 2
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or not all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in point
                )
                or not -180 <= point[0] <= 180
                or not -90 <= point[1] <= 90
                for point in geometry["coordinates"]
            )
            or not isinstance(value.get("distance_m"), (int, float))
            or isinstance(value.get("distance_m"), bool)
            or not math.isfinite(value["distance_m"])
            or value["distance_m"] <= 0
            or not isinstance(value.get("duration_s"), (int, float))
            or isinstance(value.get("duration_s"), bool)
            or not math.isfinite(value["duration_s"])
            or value["duration_s"] <= 0
            or not WorkspaceStore._valid_retrieved_at(value.get("retrieved_at"))
            or not WorkspaceStore._valid_nonnegative_number(value.get("snap_distance_origin_m"))
            or not WorkspaceStore._valid_nonnegative_number(value.get("snap_distance_destination_m"))
        ):
            raise WorkspaceError("invalid_confirmed_map_operation", 422)
        return f"route_query:{query_hash}"

    @staticmethod
    def _valid_retrieved_at(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.tzinfo is not None and parsed.utcoffset() is not None
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _valid_nonnegative_number(value: Any) -> bool:
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    @classmethod
    def _validate_operation_domain(
        cls,
        operation_type: str,
        target: str,
        parameters: dict[str, Any],
        *,
        parent_targets: set[str],
        candidate_ids: set[str],
        candidate_replacements: dict[str, str] | None,
        day_count: int,
        route_leg_ids: set[str],
        protected_stop_ids: set[str],
        parent_day_by_stop: dict[str, int],
        parent_order_by_day: dict[int, tuple[str, ...]],
        parent_duration_by_stop: dict[str, dict[str, Any]],
        parent_time_window_by_stop: dict[str, dict[str, Any] | None],
        parent_role_by_stop: dict[str, dict[str, Any]],
        role_constraints_by_stop: dict[str, tuple[dict[str, Any], ...]],
    ) -> None:
        parent_operations = {"keep_stop", "lock_stop", "mark_flexible", "move_day", "replace_nearby"}
        if operation_type in parent_operations and target not in parent_targets:
            raise WorkspaceError("invalid_draft_target", 422)
        if operation_type in {"keep_stop", "lock_stop", "mark_flexible"}:
            if parameters:
                raise WorkspaceError("invalid_draft_parameters", 422)
            return
        if operation_type == "move_day":
            day = parameters.get("day")
            if set(parameters) != {"day"} or isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= day_count:
                raise WorkspaceError("invalid_destination_day", 422)
            return
        if operation_type == "route_feedback":
            preference = parameters.get("preference")
            weight = parameters.get("weight")
            if (
                target != "selected_route"
                or set(parameters) - {"preference", "weight"}
                or preference not in cls.ROUTE_FEEDBACK_PREFERENCES
                or (weight is not None and (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or not 0 <= float(weight) <= 1
                ))
            ):
                raise WorkspaceError("invalid_route_feedback", 422)
            return
        if operation_type == "replace_nearby":
            candidate = parameters.get("candidate_id")
            if set(parameters) != {"candidate_id"} or candidate not in candidate_ids:
                raise WorkspaceError("invalid_draft_candidate", 422)
            if (
                candidate_replacements is not None
                and candidate_replacements.get(str(candidate)) != target
            ):
                raise WorkspaceError("draft_candidate_target_mismatch", 422)
            return
        if operation_type == "add_candidate":
            day = parameters.get("day")
            if (
                target not in candidate_ids
                or set(parameters) != {"day"}
                or isinstance(day, bool)
                or not isinstance(day, int)
                or not 1 <= day <= day_count
            ):
                raise WorkspaceError("invalid_draft_candidate", 422)
            return
        if operation_type == DraftOperationTypeV2.CHANGE_ROUTE_PREFERENCE.value:
            raise WorkspaceError("route_preference_not_supported", 409)
        stop_edit_operations = {
            DraftOperationTypeV2.SET_STOP_ROLE.value,
            DraftOperationTypeV2.SET_STOP_DAY.value,
            DraftOperationTypeV2.SET_STOP_ORDER.value,
            DraftOperationTypeV2.SET_STOP_DURATION.value,
            DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
            DraftOperationTypeV2.SET_STOP_COMMITMENT.value,
            DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value,
        }
        if operation_type in stop_edit_operations:
            if target not in parent_targets:
                raise WorkspaceError("invalid_draft_target", 422)
            cls._validate_typed_stop_edit(
                operation_type,
                target,
                parameters,
                day_count=day_count,
                protected_stop_ids=protected_stop_ids,
                parent_day_by_stop=parent_day_by_stop,
                parent_order_by_day=parent_order_by_day,
                parent_duration_by_stop=parent_duration_by_stop,
                parent_time_window_by_stop=parent_time_window_by_stop,
                parent_role_by_stop=parent_role_by_stop,
                role_constraints_by_stop=role_constraints_by_stop,
            )
            return
        if operation_type == DraftOperationTypeV2.REPORT_ROUTE_ISSUE.value:
            cls._validate_route_issue(target, parameters, route_leg_ids)
            return

    @classmethod
    def _validate_typed_stop_edit(
        cls,
        operation_type: str,
        target: str,
        parameters: dict[str, Any],
        *,
        day_count: int,
        protected_stop_ids: set[str],
        parent_day_by_stop: dict[str, int],
        parent_order_by_day: dict[int, tuple[str, ...]],
        parent_duration_by_stop: dict[str, dict[str, Any]],
        parent_time_window_by_stop: dict[str, dict[str, Any] | None],
        parent_role_by_stop: dict[str, dict[str, Any]],
        role_constraints_by_stop: dict[str, tuple[dict[str, Any], ...]],
    ) -> None:
        if operation_type == DraftOperationTypeV2.SET_STOP_ROLE.value:
            role = parameters.get("role")
            if set(parameters) != {"role"} or role not in {
                item.value for item in StopRoleV1
            }:
                raise WorkspaceError("invalid_stop_role", 422)
            parent_role = parent_role_by_stop.get(target) or {}
            if (
                parent_role.get("itinerary_role") == role
                and parent_role.get("itinerary_role_source")
                == "user_declared_itinerary_role"
            ):
                raise WorkspaceError("draft_no_effect", 409)
            for constraint in role_constraints_by_stop.get(target, ()):
                value = constraint.get("value")
                strength = constraint.get("strength")
                permission_semantics = constraint.get("permission_semantics")
                if (
                    not isinstance(value, str)
                    or value not in {item.value for item in StopRoleV1}
                    or strength
                    not in {
                        "locked",
                        "booked",
                        "hard",
                        "strong",
                        "soft",
                        "weak",
                        "preference",
                        "test_only",
                    }
                    or permission_semantics
                    not in {
                        "explicit_permission_required_for_mismatch",
                        "weighted_mismatch_allowed",
                    }
                ):
                    raise WorkspaceError(
                        "draft_parent_role_constraint_invalid",
                        409,
                    )
                if (
                    value != role
                    and permission_semantics
                    == "explicit_permission_required_for_mismatch"
                ):
                    raise WorkspaceError("role_edit_permission_required", 409)
            return
        if operation_type == DraftOperationTypeV2.SET_STOP_DAY.value:
            cls._validate_day(parameters, day_count)
            if parent_day_by_stop.get(target) == parameters["day"]:
                raise WorkspaceError("draft_no_effect", 409)
            return
        if operation_type == DraftOperationTypeV2.SET_STOP_ORDER.value:
            if set(parameters) != {"day", "sequence_index"}:
                raise WorkspaceError("invalid_draft_parameters", 422)
            cls._validate_day({"day": parameters.get("day")}, day_count)
            sequence_index = parameters.get("sequence_index")
            day = parameters["day"]
            day_stops = parent_order_by_day.get(day, ())
            if parent_day_by_stop.get(target) != day or target not in day_stops:
                raise WorkspaceError("draft_order_day_mismatch", 409)
            if (
                isinstance(sequence_index, bool)
                or not isinstance(sequence_index, int)
                or not 0 <= sequence_index < len(day_stops)
            ):
                raise WorkspaceError("draft_order_index_invalid", 409)
            if day_stops.index(target) == sequence_index:
                raise WorkspaceError("draft_no_effect", 409)
            return
        if operation_type == DraftOperationTypeV2.SET_STOP_DURATION.value:
            if set(parameters) != {"duration"}:
                raise WorkspaceError("invalid_draft_parameters", 422)
            cls._validate_typed_duration(parameters.get("duration"))
            duration = parameters["duration"]
            parent_duration = parent_duration_by_stop.get(target) or {}
            if (
                duration.get("mode") == DurationModeV1.EXACT.value
                and parent_duration.get("duration_constraint") == duration
                and parent_duration.get("visit_duration_minutes")
                == duration.get("preferred_minutes")
            ):
                raise WorkspaceError("draft_no_effect", 409)
            return
        if operation_type == DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value:
            cls._validate_time_window(parameters)
            parent_window = parent_time_window_by_stop.get(target)
            if parent_window is not None:
                canonical = {
                    "schema_version": "stop-time-window-constraint-v1",
                    "earliest_arrival": parameters.get("earliest_arrival"),
                    "latest_departure": parameters.get("latest_departure"),
                    "early_arrival_policy": "wait_until_earliest_arrival",
                    "latest_departure_semantics": "departure_after_visit",
                }
                if parent_window == canonical:
                    raise WorkspaceError("draft_no_effect", 409)
            return
        if operation_type == DraftOperationTypeV2.SET_STOP_COMMITMENT.value:
            if set(parameters) != {"strength", "scope_lifetime"}:
                raise WorkspaceError("invalid_draft_parameters", 422)
            cls._validate_commitment(
                target,
                parameters.get("strength"),
                parameters.get("scope_lifetime"),
                protected_stop_ids,
            )
            return
        if operation_type == DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value:
            if set(parameters) != {"attribute", "strength", "value", "scope_lifetime"}:
                raise WorkspaceError("invalid_draft_parameters", 422)
            attribute = parameters.get("attribute")
            if attribute not in {item.value for item in AttributeConstraintNameV1}:
                raise WorkspaceError("invalid_attribute_constraint", 422)
            cls._validate_commitment(
                target,
                parameters.get("strength"),
                parameters.get("scope_lifetime"),
                protected_stop_ids,
            )
            cls._validate_attribute_value(attribute, parameters.get("value"), day_count)
            return
        raise WorkspaceError("unsupported_draft_operation", 422)

    @staticmethod
    def _validate_day(parameters: dict[str, Any], day_count: int) -> None:
        day = parameters.get("day")
        if set(parameters) != {"day"} or isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= day_count:
            raise WorkspaceError("invalid_destination_day", 422)

    @staticmethod
    def _validate_typed_duration(value: Any) -> None:
        fields = {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
        if not isinstance(value, dict) or set(value) != fields:
            raise WorkspaceError("invalid_stop_duration", 422)
        mode = value.get("mode")
        preferred = value.get("preferred_minutes")
        minimum = value.get("minimum_minutes")
        maximum = value.get("maximum_minutes")

        def valid_minutes(item: Any) -> bool:
            return isinstance(item, int) and not isinstance(item, bool) and 15 <= item <= 480

        valid = (
            mode == DurationModeV1.EXACT.value
            and valid_minutes(preferred)
            and preferred == minimum == maximum
        ) or (
            mode == DurationModeV1.PREFERRED.value
            and valid_minutes(preferred)
            and minimum is None
            and maximum is None
        ) or (
            mode == DurationModeV1.MINIMUM.value
            and preferred is None
            and valid_minutes(minimum)
            and maximum is None
        ) or (
            mode == DurationModeV1.MAXIMUM.value
            and preferred is None
            and minimum is None
            and valid_minutes(maximum)
        ) or (
            mode == DurationModeV1.RANGE.value
            and preferred is None
            and valid_minutes(minimum)
            and valid_minutes(maximum)
            and minimum <= maximum
        )
        if not valid:
            raise WorkspaceError("invalid_stop_duration", 422)

    @staticmethod
    def _validate_time_window(parameters: dict[str, Any]) -> None:
        if set(parameters) != {"earliest_arrival", "latest_departure"}:
            raise WorkspaceError("invalid_stop_time_window", 422)
        earliest = parameters.get("earliest_arrival")
        latest = parameters.get("latest_departure")
        if earliest is None and latest is None:
            raise WorkspaceError("invalid_stop_time_window", 422)

        def minute(value: Any) -> int | None:
            if value is None:
                return None
            if not isinstance(value, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is None:
                raise WorkspaceError("invalid_stop_time_window", 422)
            hours, minutes = value.split(":")
            return int(hours) * 60 + int(minutes)

        earliest_minute = minute(earliest)
        latest_minute = minute(latest)
        if earliest_minute is not None and latest_minute is not None and earliest_minute > latest_minute:
            raise WorkspaceError("invalid_stop_time_window", 422)

    @staticmethod
    def _validate_commitment(
        target: str,
        strength: Any,
        scope_lifetime: Any,
        protected_stop_ids: set[str],
    ) -> None:
        if strength not in {item.value for item in CommitmentStrengthV1}:
            raise WorkspaceError("invalid_stop_commitment", 422)
        if scope_lifetime not in {item.value for item in ConstraintScopeLifetimeV1}:
            raise WorkspaceError("invalid_stop_commitment", 422)
        if target in protected_stop_ids or strength in {
            CommitmentStrengthV1.MUST_KEEP.value,
            CommitmentStrengthV1.BOOKED.value,
        }:
            raise WorkspaceError("commitment_permission_required", 409)

    @classmethod
    def _validate_attribute_value(cls, attribute: str, value: Any, day_count: int) -> None:
        if attribute == AttributeConstraintNameV1.EXISTENCE.value:
            if not isinstance(value, bool):
                raise WorkspaceError("invalid_attribute_constraint", 422)
            return
        if attribute == AttributeConstraintNameV1.DAY.value:
            cls._validate_day({"day": value}, day_count)
            return
        if attribute == AttributeConstraintNameV1.DURATION.value:
            cls._validate_typed_duration(value)
            return
        if attribute == AttributeConstraintNameV1.SEQUENCE_ORDER.value:
            if not isinstance(value, dict) or set(value) != {"day", "sequence_index"}:
                raise WorkspaceError("invalid_attribute_constraint", 422)
            cls._validate_day({"day": value.get("day")}, day_count)
            sequence_index = value.get("sequence_index")
            if (
                isinstance(sequence_index, bool)
                or not isinstance(sequence_index, int)
                or not 0 <= sequence_index <= 63
            ):
                raise WorkspaceError("invalid_attribute_constraint", 422)
            return
        if attribute == AttributeConstraintNameV1.ROLE.value:
            if value not in {item.value for item in StopRoleV1}:
                raise WorkspaceError("invalid_attribute_constraint", 422)
            return
        raise WorkspaceError("invalid_attribute_constraint", 422)

    @staticmethod
    def _validate_route_issue(target: str, parameters: dict[str, Any], route_leg_ids: set[str]) -> None:
        if target not in route_leg_ids:
            raise WorkspaceError("invalid_route_issue", 422)
        if set(parameters) != {"route_leg_id", "issue_type", "note"}:
            raise WorkspaceError("invalid_route_issue", 422)
        note = parameters.get("note")
        if (
            parameters.get("route_leg_id") != target
            or parameters.get("issue_type") not in {item.value for item in RouteIssueTypeV1}
            or (note is not None and (not isinstance(note, str) or not note.strip() or len(note) > 500))
        ):
            raise WorkspaceError("invalid_route_issue", 422)

    @staticmethod
    def _validate_draft_conflicts(operations: tuple[DraftOperationV1, ...]) -> None:
        signatures = [(row.type, row.target, row.parameters) for row in operations]
        if any(
            left[0] == right[0]
            and left[0] != DraftOperationTypeV2.SET_STOP_ROLE.value
            and left[1] == right[1]
            and left[2] == right[2]
            for index, left in enumerate(signatures)
            for right in signatures[index + 1 :]
        ):
            raise WorkspaceError("draft_duplicate_operation", 409)
        policies: dict[str, set[str]] = {}
        moves: dict[str, set[int]] = {}
        replacements: dict[str, set[str]] = {}
        preferences: set[str] = set()
        requested_candidates: set[str] = set()
        typed_values: dict[tuple[str, str], set[str]] = {}
        role_edit_counts: dict[str, int] = {}
        day_edits: dict[str, set[int]] = {}
        order_edits: dict[str, set[tuple[int, int]]] = {}
        day_edit_sources: dict[str, set[str]] = {}
        order_edit_sources: dict[str, set[str]] = {}
        for operation in operations:
            if operation.type in {"keep_stop", "lock_stop", "mark_flexible"}:
                policies.setdefault(operation.target, set()).add(operation.type)
            elif operation.type == "move_day":
                day = int(operation.parameters["day"])
                moves.setdefault(operation.target, set()).add(day)
                day_edits.setdefault(operation.target, set()).add(day)
                day_edit_sources.setdefault(operation.target, set()).add("move_day")
            elif operation.type == "replace_nearby":
                candidate = str(operation.parameters["candidate_id"])
                replacements.setdefault(operation.target, set()).add(candidate)
                requested_candidates.add(candidate)
            elif operation.type == "add_candidate":
                requested_candidates.add(operation.target)
            elif operation.type == "route_feedback":
                preferences.add(str(operation.parameters["preference"]))
            elif operation.type == DraftOperationTypeV2.SET_STOP_DAY.value:
                day_edits.setdefault(operation.target, set()).add(int(operation.parameters["day"]))
                day_edit_sources.setdefault(operation.target, set()).add("set_stop_day")
            elif operation.type == DraftOperationTypeV2.SET_STOP_ORDER.value:
                order_edits.setdefault(operation.target, set()).add(
                    (int(operation.parameters["day"]), int(operation.parameters["sequence_index"]))
                )
                order_edit_sources.setdefault(operation.target, set()).add("set_stop_order")
            elif operation.type in {
                DraftOperationTypeV2.SET_STOP_ROLE.value,
                DraftOperationTypeV2.SET_STOP_DURATION.value,
                DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
                DraftOperationTypeV2.SET_STOP_COMMITMENT.value,
                DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value,
            }:
                attribute = {
                    DraftOperationTypeV2.SET_STOP_ROLE.value: "role",
                    DraftOperationTypeV2.SET_STOP_DURATION.value: "duration",
                    DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value: "time_window",
                    DraftOperationTypeV2.SET_STOP_COMMITMENT.value: "commitment",
                }.get(operation.type, str(operation.parameters.get("attribute") or ""))
                signature = repr(sorted(operation.parameters.items()))
                typed_values.setdefault((operation.target, attribute), set()).add(signature)
                if operation.type == DraftOperationTypeV2.SET_STOP_ROLE.value:
                    role_edit_counts[operation.target] = (
                        role_edit_counts.get(operation.target, 0) + 1
                    )
                if (
                    operation.type == DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value
                    and attribute == AttributeConstraintNameV1.DAY.value
                ):
                    day_edits.setdefault(operation.target, set()).add(int(operation.parameters["value"]))
                    day_edit_sources.setdefault(operation.target, set()).add("attribute_day")
                elif (
                    operation.type == DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value
                    and attribute == AttributeConstraintNameV1.SEQUENCE_ORDER.value
                ):
                    value = operation.parameters["value"]
                    order_edits.setdefault(operation.target, set()).add(
                        (int(value["day"]), int(value["sequence_index"]))
                    )
                    order_edit_sources.setdefault(operation.target, set()).add("attribute_sequence_order")
        if any("mark_flexible" in values and values & {"keep_stop", "lock_stop"} for values in policies.values()):
            raise WorkspaceError("draft_conflicting_stop_policy", 409)
        if any(target in replacements and values & {"keep_stop", "lock_stop"} for target, values in policies.items()):
            raise WorkspaceError("draft_conflicts_with_replacement", 409)
        if any(len(values) > 1 for values in moves.values()):
            raise WorkspaceError("draft_conflicting_day_moves", 409)
        if any(len(values) > 1 for values in replacements.values()):
            raise WorkspaceError("draft_conflicting_replacements", 409)
        if len(preferences) > 1:
            raise WorkspaceError("draft_conflicting_route_feedback", 409)
        if len(requested_candidates) > 1:
            raise WorkspaceError("draft_conflicting_candidates", 409)
        if any(len(values) > 1 for values in typed_values.values()):
            raise WorkspaceError("draft_conflicting_attribute_edits", 409)
        if any(count > 1 for count in role_edit_counts.values()):
            raise WorkspaceError("draft_conflicting_attribute_edits", 409)
        if any(len(values) > 1 for values in day_edits.values()):
            raise WorkspaceError("draft_conflicting_day_moves", 409)
        if any(len(values) > 1 for values in order_edits.values()):
            raise WorkspaceError("draft_conflicting_stop_order", 409)
        if any(len(values) > 1 for values in day_edit_sources.values()):
            raise WorkspaceError("draft_duplicate_day_edit", 409)
        if any(len(values) > 1 for values in order_edit_sources.values()):
            raise WorkspaceError("draft_duplicate_stop_order", 409)
        for target, orders in order_edits.items():
            if target in day_edits and {day for day, _ in orders} != day_edits[target]:
                raise WorkspaceError("draft_conflicting_day_order", 409)

    def undo(self, session: ProductSession, expected_revision: Any) -> DraftOperationV1:
        operation: DraftOperationV1 | None = None

        def mutate(current: ProductSession) -> None:
            nonlocal operation
            if not current.draft:
                raise WorkspaceError("draft_is_empty")
            operation = current.draft.pop()
            current.proposal = None

        self._mutate(session, expected_revision, mutate)
        assert operation is not None
        return operation

    def set_proposal(self, session: ProductSession, proposal: dict[str, Any], expected_revision: Any) -> None:
        self._mutate(session, expected_revision, lambda current: setattr(current, "proposal", proposal))

    def clear_draft(self, session: ProductSession) -> None:
        def mutate(current: ProductSession) -> None:
            current.draft.clear()
            current.proposal = None

        self._mutate(session, session.revision, mutate)

    def append_permission(self, session: ProductSession, payload: dict[str, Any]) -> dict[str, Any]:
        decision = {
            "permission": str(payload.get("permission") or ""),
            "decision": str(payload.get("decision") or ""),
            "proposal_id": self._optional_text(payload.get("proposal_id")),
            "scope": self._optional_text(payload.get("scope")),
            "expires_at": self._optional_text(payload.get("expires_at")),
            "created_at": utc_now(),
        }
        if decision["decision"] not in {"granted", "denied"}:
            raise WorkspaceError("invalid_permission_decision")
        def mutate(current: ProductSession) -> None:
            if len(current.permission_decisions) >= self.MAX_PERMISSION_DECISIONS:
                raise WorkspaceError("permission_decision_limit_reached", 409)
            current.permission_decisions.append(decision)

        self._mutate(session, payload.get("expected_revision"), mutate)
        return decision

    def accept(
        self,
        session: ProductSession,
        *,
        expected_revision: Any,
        parent_plan_id: str,
        child_plan_id: str,
        certificate_id: str,
        diff_id: str,
    ) -> dict[str, Any]:
        del parent_plan_id, child_plan_id, certificate_id, diff_id
        self.expect_revision(session, expected_revision)
        raise WorkspaceError("acceptance_not_enabled_until_w5", 409)

    def keep_original(self, session: ProductSession, expected_revision: Any) -> dict[str, Any]:
        self.expect_revision(session, expected_revision)
        raise WorkspaceError("acceptance_not_enabled_until_w5", 409)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _is_expired(session: ProductSession) -> bool:
        try:
            last_accessed = datetime.fromisoformat(session.last_accessed_at)
        except ValueError:
            return True
        return datetime.now(UTC) - last_accessed > WorkspaceStore.SESSION_TTL

    def _purge_expired_locked(self) -> None:
        expired = [session_id for session_id, session in self._sessions.items() if self._is_expired(session)]
        for session_id in expired:
            self._snapshots.delete(session_id)
            self._sessions.pop(session_id, None)

    def _restore(self) -> None:
        try:
            self._snapshots.ensure_directories()
            with self._snapshots.locked():
                self._refresh_locked()
                self._purge_expired_locked()
        except StateLockTimeoutError:
            self._restoration_errors["__store__"] = "state_lock_timeout"
        except SessionSnapshotError as exc:
            self._restoration_errors["__store__"] = exc.code

    def _refresh_locked(self) -> None:
        for session_id in self._snapshots.session_ids():
            if session_id in self._sessions or session_id in self._restoration_errors:
                continue
            try:
                session = self._snapshots.load(session_id)
                self.reduce_confirmed_map_draft(session.draft)
                self._sessions[session_id] = session
            except SessionSnapshotError as exc:
                self._restoration_errors[session_id] = exc.code
            except WorkspaceError as exc:
                self._restoration_errors[session_id] = exc.code

    def _load_current_locked(self, session_id: str) -> ProductSession:
        if session_id in self._restoration_errors:
            raise WorkspaceError(self._restoration_errors[session_id], 409)
        try:
            current = self._snapshots.load(session_id)
        except SessionSnapshotError as exc:
            if exc.code == "unknown_session":
                raise WorkspaceError("unknown_session", 404) from None
            self._restoration_errors[session_id] = exc.code
            raise
        self.reduce_confirmed_map_draft(current.draft)
        return current

    def _mutate(
        self,
        session: ProductSession,
        expected_revision: Any,
        mutation: Any,
    ) -> None:
        with self._lock:
            try:
                with self._snapshots.locked():
                    current = self._load_current_locked(session.session_id)
                    if self._is_expired(current):
                        self._snapshots.delete(current.session_id)
                        self._sessions.pop(current.session_id, None)
                        raise WorkspaceError("session_expired", 404)
                    self.expect_revision(current, expected_revision)
                    mutation(current)
                    current.revision += 1
                    current.last_accessed_at = utc_now()
                    self._snapshots.save(current)
                    self._sync_session(session, current)
                    self._sessions[current.session_id] = session
            except StateLockTimeoutError:
                raise WorkspaceError("state_lock_timeout", 503) from None
            except SessionSnapshotError as exc:
                raise self._snapshot_workspace_error(exc) from None

    def _install_locked(self, current: ProductSession) -> ProductSession:
        existing = self._sessions.get(current.session_id)
        if existing is None:
            self._sessions[current.session_id] = current
            return current
        self._sync_session(existing, current)
        return existing

    @staticmethod
    def _sync_session(target: ProductSession, source: ProductSession) -> None:
        for field_name in ProductSession.__dataclass_fields__:
            setattr(target, field_name, getattr(source, field_name))

    @staticmethod
    def _token_digest(token: str, salt: str) -> str:
        return hashlib.sha256(bytes.fromhex(salt) + token.encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot_workspace_error(error: SessionSnapshotError) -> WorkspaceError:
        status = 404 if error.code == "unknown_session" else 409
        if error.code in {
            "session_delete_failed",
            "session_write_failed",
            "session_store_unavailable",
        }:
            status = 503
        return WorkspaceError(error.code, status)

    @staticmethod
    def _evidence_refs(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or len(value) > 16:
            raise WorkspaceError("invalid_evidence_refs")
        refs = tuple(str(item) for item in value)
        if any(not ref or len(ref) > 256 for ref in refs):
            raise WorkspaceError("invalid_evidence_refs")
        return refs
