"""Typed contracts for local product state.

These contracts describe mutable application state. Canonical plan, diff, route,
evaluation, and certificate artifacts remain owned by the research run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ProductRunRecord:
    run_id: str
    trip_id: str
    label: str
    relative_path: str
    manifest_hash: str
    capabilities: tuple[str, ...]
    default: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DraftOperationV1:
    operation_id: str
    type: str
    target: str
    parameters: dict[str, Any]
    source: str
    evidence_refs: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotContextV1:
    run_id: str
    trip_id: str
    session_id: str
    session_revision: int
    accepted_plan_id: str
    selected_day: int | None
    selected_stop_id: str | None
    selected_segment_id: str | None
    selected_candidate_id: str | None
    selected_alternative_id: str | None
    draft_operations: tuple[dict[str, Any], ...]
    evaluated_proposal: dict[str, Any] | None
    allowed_stop_ids: tuple[str, ...]
    allowed_candidate_ids: tuple[str, ...]
    allowed_days: tuple[int, ...]
    allowed_segment_ids: tuple[str, ...] = ()
    allowed_alternative_ids: tuple[str, ...] = ()
    schema_version: str = "copilot-context-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotIntentV1:
    type: str
    target_stop_id: str | None = None
    destination_day: int | None = None
    candidate_id: str | None = None
    preference: str | None = None
    clarification_question: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotHighlightsV1:
    day_ids: tuple[int, ...] = ()
    stop_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotInterpretationV1:
    state: str
    assistant_message: str
    intents: tuple[CopilotIntentV1, ...] = ()
    permission: str | None = None
    highlights: CopilotHighlightsV1 = field(default_factory=CopilotHighlightsV1)
    schema_version: str = "copilot-interpretation-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotTurnV1:
    turn_id: str
    client_message_id: str
    context_revision: int
    provider: str
    model: str | None
    prompt_version: str
    prompt_sha256: str
    state: str
    user_message: str
    assistant_message: str
    interpretation: CopilotInterpretationV1 | None
    error_code: str | None
    created_at: str
    completed_at: str
    schema_version: str = "copilot-turn-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationV1:
    conversation_id: str
    session_id: str
    trip_id: str
    run_id: str
    revision: int
    turns: tuple[CopilotTurnV1, ...]
    created_at: str
    updated_at: str
    expires_at: str
    schema_version: str = "product-conversation-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductSession:
    session_id: str
    mutation_token_salt: str = field(repr=False)
    mutation_token_verifier: str = field(repr=False)
    trip_id: str
    run_id: str
    revision: int
    accepted_plan_id: str
    selected_day: int | None = None
    selected_stop_id: str | None = None
    selected_segment_id: str | None = None
    selected_candidate_id: str | None = None
    selected_alternative_id: str | None = None
    draft: list[DraftOperationV1] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    permission_decisions: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    last_accessed_at: str = field(default_factory=utc_now)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "workspace-context-v1",
            "session_id": self.session_id,
            "trip_id": self.trip_id,
            "run_id": self.run_id,
            "revision": self.revision,
            "accepted_plan_id": self.accepted_plan_id,
            "selected_day": self.selected_day,
            "selected_stop_id": self.selected_stop_id,
            "selected_segment_id": self.selected_segment_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_alternative_id": self.selected_alternative_id,
            "draft": [operation.as_dict() for operation in self.draft],
            "proposal": self.proposal,
            "permission_decisions": list(self.permission_decisions),
            "conversation_id": self.conversation_id,
        }


@dataclass(frozen=True)
class ComponentHealthV1:
    name: str
    status: str
    required_for_core: bool
    code: str
    checked_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductHealthV2:
    product_id: str
    product_version: str
    build_id: str
    status: str
    core_ready: bool
    default_run: str | None
    legacy_enabled: bool
    components: dict[str, ComponentHealthV1]
    schema_version: str = "product-health-v2"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready"] = self.core_ready
        return payload


@dataclass(frozen=True)
class MapConfigurationV2:
    status: str
    base_url: str
    style_url: str
    script_url: str
    stylesheet_url: str
    protocol_script_url: str | None
    provenance_url: str | None
    attribution: str
    attribution_url: str | None
    runtime_license: str | None
    provider: str
    range_requests_required: bool = True
    schema_version: str = "product-map-configuration-v2"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
