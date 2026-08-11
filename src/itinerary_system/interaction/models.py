"""Typed records for permission-aware counterfactual clarification.

The records in this module are evidence containers only.  They do not grant
permission, determine solver feasibility, or make a hypothetical plan
execution-eligible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from ..repair.neighborhood import RepairRadius

JsonValue = Any


class ClarificationMode(StrEnum):
    DISABLED = "disabled"
    RULE_BASED_CONSEQUENCE = "rule_based_consequence"
    FROZEN_FIXTURE = "frozen_fixture"


class ProbeStatus(StrEnum):
    FEASIBLE_BOUNDED = "feasible_bounded"
    INFEASIBLE = "infeasible"
    TIME_LIMIT_WITH_INCUMBENT = "time_limit_with_incumbent"
    TIME_LIMIT_NO_INCUMBENT = "time_limit_no_incumbent"
    INVALID_PATCH = "invalid_patch"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    FAILED = "failed"


class ClarificationAction(StrEnum):
    COMMIT = "commit"
    ASK_SEMANTIC = "ask_semantic"
    ASK_PERMISSION = "ask_permission"
    PROBE_MORE = "probe_more"
    DEFER = "defer"


class PermissionDecisionAction(StrEnum):
    GRANT = "grant"
    DENY = "deny"
    GRANT_ONCE = "grant_once"


@dataclass(frozen=True)
class InteractionOptions:
    clarification_mode: ClarificationMode = ClarificationMode.DISABLED
    max_questions: int = 2
    probe_time_limit_seconds: float = 2.0
    allow_live_llm: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "clarification_mode", ClarificationMode(self.clarification_mode))
        if isinstance(self.max_questions, bool) or not isinstance(self.max_questions, int) or self.max_questions < 0:
            raise ValueError("max_questions must be a nonnegative integer")
        if (
            isinstance(self.probe_time_limit_seconds, bool)
            or not isinstance(self.probe_time_limit_seconds, (int, float))
            or not isfinite(float(self.probe_time_limit_seconds))
            or self.probe_time_limit_seconds <= 0
        ):
            raise ValueError("probe_time_limit_seconds must be finite and positive")
        if self.allow_live_llm:
            raise ValueError("live LLM providers are not supported by the rule-based interaction extension")


@dataclass(frozen=True)
class InteractionRequest:
    repair_session_id: str
    parent_plan_id: str
    user_edit: str
    evidence_refs: tuple[str, ...] = ()
    selected_interpretation_id: str | None = None
    question_count: int = 0
    continuation_of_session_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("repair_session_id", "parent_plan_id", "user_edit"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonempty")
        if isinstance(self.question_count, bool) or not isinstance(self.question_count, int) or self.question_count < 0:
            raise ValueError("question_count must be a nonnegative integer")


@dataclass(frozen=True)
class SemanticInterpretationCandidate:
    interpretation_id: str
    repair_session_id: str
    user_text_hash: str
    target_ids: tuple[str, ...]
    interpretation_type: str
    normalized_parameters: dict[str, JsonValue]
    support_score: float | None
    evidence_refs: tuple[str, ...]
    confirmed: bool = False
    redacted_excerpt: str | None = None
    schema_version: str = "semantic-interpretation-v1"

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class ModelPatch:
    patch_id: str
    interpretation_id: str
    patch_type: str
    target_ids: tuple[str, ...]
    parameters: dict[str, JsonValue]
    affected_constraint_ids: tuple[str, ...]
    validation_status: str
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    schema_version: str = "model-patch-v1"

    @property
    def is_valid(self) -> bool:
        return self.validation_status == "valid"

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class CounterfactualProbeRequest:
    probe_request_id: str
    repair_session_id: str
    parent_plan_id: str
    interpretation_id: str
    model_patch_id: str
    allowed_probe_constraint_ids: tuple[str, ...]
    repair_radius: RepairRadius
    time_limit_seconds: float
    test_only: bool = True
    schema_version: str = "counterfactual-probe-request-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_radius", RepairRadius(self.repair_radius))
        for field_name in (
            "probe_request_id",
            "repair_session_id",
            "parent_plan_id",
            "interpretation_id",
            "model_patch_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonempty")
        if not self.test_only:
            raise ValueError("counterfactual probes must be test_only")
        if (
            isinstance(self.time_limit_seconds, bool)
            or not isinstance(self.time_limit_seconds, (int, float))
            or not isfinite(float(self.time_limit_seconds))
            or self.time_limit_seconds <= 0
        ):
            raise ValueError("counterfactual probe time limit must be finite and positive")

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class CounterfactualProbeResult:
    probe_result_id: str
    probe_request_id: str
    parent_plan_id: str
    hypothetical_plan_id: str | None
    status: ProbeStatus
    diff_id: str | None
    solver_run_ids: tuple[str, ...]
    requires_user_permission: bool
    permission_constraint_ids: tuple[str, ...]
    eligible_for_execution: bool
    evidence_refs: tuple[str, ...]
    interpretation_id: str | None = None
    affected_constraint_ids: tuple[str, ...] = ()
    runtime_seconds: float | None = None
    solver_bound: float | None = None
    solver_gap: float | None = None
    accepted_repair_radius: str | None = None
    failure_reasons: tuple[str, ...] = ()
    diff_record: dict[str, Any] | None = None
    hypothetical_plan_record: dict[str, Any] | None = None
    diagnostic_evaluation: dict[str, Any] | None = None
    schema_version: str = "counterfactual-probe-result-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProbeStatus(self.status))
        if self.eligible_for_execution:
            raise ValueError("counterfactual probe results can never be execution-eligible")
        for field_name, minimum in (("runtime_seconds", 0.0), ("solver_gap", 0.0)):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < minimum
            ):
                raise ValueError(f"{field_name} must be finite and nonnegative when provided")
        if self.solver_bound is not None and (
            isinstance(self.solver_bound, bool)
            or not isinstance(self.solver_bound, (int, float))
            or not isfinite(float(self.solver_bound))
        ):
            raise ValueError("solver_bound must be finite when provided")

    def to_record(self) -> dict[str, Any]:
        record = _jsonish(asdict(self))
        record.pop("hypothetical_plan_record", None)
        return record


@dataclass(frozen=True)
class ConsequenceVector:
    consequence_id: str
    interpretation_id: str
    probe_result_id: str
    hard_feasible: bool
    permission_change_count: int
    locked_change_count: int
    booked_change_count: int
    strong_change_count: int
    flexible_change_count: int
    core_commitment_changes: tuple[str, ...]
    weighted_edit_cost: float | None
    affected_day_count: int
    lodging_change_count: int
    road_change_count: int
    travel_minutes_delta: float | None
    monetary_cost_delta: float | None
    walking_burden_delta: float | None
    contextual_risk_delta: float | None
    utility_delta: float | None
    accepted_repair_radius: str | None
    evidence_refs: tuple[str, ...]
    schema_version: str = "consequence-vector-v1"

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class ConsequenceThresholds:
    max_low_consequence_edit_cost: float = 10.0
    max_low_consequence_affected_days: int = 1
    max_low_consequence_travel_minutes: float = 30.0
    max_low_consequence_cost_delta: float = 25.0
    permission_change_is_material: bool = True
    feasibility_difference_is_material: bool = True

    def __post_init__(self) -> None:
        numeric_fields = (
            "max_low_consequence_edit_cost",
            "max_low_consequence_travel_minutes",
            "max_low_consequence_cost_delta",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if (
            isinstance(self.max_low_consequence_affected_days, bool)
            or not isinstance(self.max_low_consequence_affected_days, int)
            or self.max_low_consequence_affected_days < 0
        ):
            raise ValueError("max_low_consequence_affected_days must be a nonnegative integer")


@dataclass(frozen=True)
class CriticalTradeoff:
    tradeoff_id: str
    repair_session_id: str
    left_interpretation_id: str
    right_interpretation_id: str
    primary_dimension: str
    left_summary: str
    right_summary: str
    numerical_deltas: dict[str, float]
    permission_required: bool
    evidence_refs: tuple[str, ...]
    schema_version: str = "critical-tradeoff-v1"

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class ClarificationDecision:
    decision_id: str
    repair_session_id: str
    action: ClarificationAction
    selected_interpretation_id: str | None
    selected_tradeoff_id: str | None
    question_text: str | None
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    schema_version: str = "clarification-decision-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ClarificationAction(self.action))
        asks = self.action in {ClarificationAction.ASK_PERMISSION, ClarificationAction.ASK_SEMANTIC}
        if asks and not str(self.question_text or "").strip():
            raise ValueError("question_text is required for clarification actions")

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class UserPermissionDecision:
    permission_decision_id: str
    repair_session_id: str
    constraint_ids: tuple[str, ...]
    action: PermissionDecisionAction
    selected_interpretation_id: str | None
    created_at: str
    evidence_refs: tuple[str, ...]
    schema_version: str = "user-permission-decision-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", PermissionDecisionAction(self.action))
        if not self.repair_session_id:
            raise ValueError("permission decisions must be scoped to a repair session")

    def to_record(self) -> dict[str, Any]:
        return _jsonish(asdict(self))


@dataclass(frozen=True)
class InteractionArtifacts:
    semantic_candidates: tuple[SemanticInterpretationCandidate, ...] = ()
    model_patches: tuple[ModelPatch, ...] = ()
    probe_requests: tuple[CounterfactualProbeRequest, ...] = ()
    probe_results: tuple[CounterfactualProbeResult, ...] = ()
    consequence_vectors: tuple[ConsequenceVector, ...] = ()
    critical_tradeoffs: tuple[CriticalTradeoff, ...] = ()
    clarification_decisions: tuple[ClarificationDecision, ...] = ()
    permission_decisions: tuple[UserPermissionDecision, ...] = ()
    selected_authorized_repair_request_id: str | None = None
    selected_interpretation_id: str | None = None
    semantic_candidate_source: str = ""
    continuation_of_session_id: str | None = None


def _jsonish(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonish(item) for item in value]
    return value
