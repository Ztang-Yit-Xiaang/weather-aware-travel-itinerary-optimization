"""Compile typed drafts into deterministic repair and evaluation requests."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_trip_config
from ..pipeline_runner import (
    PipelineRunContext,
    RefreshPolicy,
    run_deterministic_context_aware_heuristic_executor,
)
from ..repair.day_route_solver import DayRouteSolverConfig
from ..repair_planner import ParsedRepairIntent, RepairRequest
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixCell
from .evaluated_stop_edits import (
    EvaluatedStopDurationCompiler,
    EvaluatedStopOrderCompiler,
    EvaluatedStopRoleCompiler,
    EvaluatedStopTimeWindowCompiler,
)
from .interaction_contracts import DRAFT_ONLY_TYPED_EDIT_OPERATIONS, DraftOperationTypeV2
from .workspace import WorkspaceError

CANONICAL_OPERATIONS = frozenset(
    {
        "keep_stop",
        "lock_stop",
        "mark_flexible",
        "move_day",
        "route_feedback",
        "replace_nearby",
        "add_candidate",
        DraftOperationTypeV2.SET_STOP_DAY.value,
        DraftOperationTypeV2.SET_STOP_ORDER.value,
        DraftOperationTypeV2.SET_STOP_ROLE.value,
        DraftOperationTypeV2.SET_STOP_DURATION.value,
        DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
    }
)
LEGACY_CAPABILITY_OPERATIONS = CANONICAL_OPERATIONS - {
    DraftOperationTypeV2.SET_STOP_DAY.value,
    DraftOperationTypeV2.SET_STOP_ORDER.value,
    DraftOperationTypeV2.SET_STOP_ROLE.value,
    DraftOperationTypeV2.SET_STOP_DURATION.value,
    DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
}

_PREFERENCES = frozenset({"reduce_contextual_risk", "reduce_driving", "preserve_experience"})
_DAY_ANCHORS = {
    1: ("los_angeles_international_airport", "the_line_la"),
    2: ("the_line_la", "the_line_la"),
    3: ("the_line_la", "the_line_la"),
    4: ("the_line_la", "hotel_milo_santa_barbara"),
    5: ("hotel_milo_santa_barbara", "hotel_milo_santa_barbara"),
    6: ("hotel_milo_santa_barbara", "hotel_milo_santa_barbara"),
    7: ("hotel_milo_santa_barbara", "san_francisco_international_airport"),
}


@dataclass(frozen=True)
class ExecutableCandidate:
    candidate_id: str
    label: str
    replaced_stop_id: str
    child_plan_id: str
    source_request_id: str
    record: dict[str, Any]
    bundle: Any


@dataclass(frozen=True)
class CompiledDraftPreview:
    """In-memory pipeline result; W5 remains the only acceptance authority."""

    state: str
    reason: str | None
    source_request_id: str
    operations: tuple[dict[str, Any], ...]
    source_bundle: Any
    child_plan: dict[str, Any] | None = None
    diff: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None
    planner_run: dict[str, Any] | None = None
    parent_route_legs: tuple[dict[str, Any], ...] = ()
    route_legs: tuple[dict[str, Any], ...] = ()
    schedule_impact: dict[str, Any] | None = None
    execution_mode: str = "deterministic_repair_pipeline"


class FrozenDraftCompiler:
    """Compile the canonical vocabulary against a closed artifact-backed universe."""

    def __init__(
        self,
        parent_plan: dict[str, Any],
        bundles_by_plan: dict[str, Any],
        *,
        repository_root: Path,
    ) -> None:
        self.parent_plan = parent_plan
        self.parent_plan_id = str(parent_plan.get("plan_id") or "")
        self.parent_stop_ids = frozenset(_plan_stop_ids(parent_plan))
        self._parent_day_by_stop = {
            _stop_id(row): int(row["day"])
            for row in parent_plan.get("selected_stops") or ()
            if isinstance(row, dict)
            and _stop_id(row)
            and isinstance(row.get("day"), int)
            and not isinstance(row.get("day"), bool)
        }
        self.repository_root = repository_root.resolve()
        self._parent_artifact = _plan_artifact(parent_plan)
        self._bundles_by_plan = dict(bundles_by_plan)
        self._candidates = self._build_candidates(bundles_by_plan)
        if not self._candidates:
            raise WorkspaceError("draft_candidate_evidence_missing", 409)
        self._base_bundle = next(iter(bundles_by_plan.values()))
        self._runtime_cache: tuple[RouteMatrix, DayRouteSolverConfig, Any] | None = None
        self._runtime_cache_lock = threading.RLock()

    def capabilities(self) -> dict[str, Any]:
        choices = [
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "replaces_stop_id": candidate.replaced_stop_id,
                "child_plan_id": candidate.child_plan_id,
            }
            for candidate in sorted(self._candidates.values(), key=lambda item: item.candidate_id)
        ]
        return {
            "schema_version": "draft-capabilities-v1",
            "operations": {
                name: {
                    "enabled": True,
                    "preview_executable": True,
                    "status": "deterministic_pipeline",
                }
                for name in sorted(LEGACY_CAPABILITY_OPERATIONS)
            },
            "parent_targets": sorted(self.parent_stop_ids),
            "candidate_choices": choices,
            "executable_combinations": [
                {
                    "operations": sorted(LEGACY_CAPABILITY_OPERATIONS),
                    "cardinality": 32,
                    "description": (
                        "Ordered canonical drafts run through deterministic repair and independent evaluation; "
                        "conflicts fail closed."
                    ),
                }
            ],
        }

    def compile(
        self,
        operations: Iterable[Any],
        *,
        accepted_plan_id: str,
    ) -> CompiledDraftPreview:
        if accepted_plan_id != self.parent_plan_id:
            raise WorkspaceError("draft_parent_not_supported", 409)
        normalized = tuple(_operation_record(operation) for operation in operations)
        if not normalized:
            raise WorkspaceError("draft_is_empty")
        evaluated_direct_types = {
            DraftOperationTypeV2.SET_STOP_ORDER.value,
            DraftOperationTypeV2.SET_STOP_ROLE.value,
            DraftOperationTypeV2.SET_STOP_DURATION.value,
            DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
        }
        present_direct_types = {row["type"] for row in normalized} & evaluated_direct_types
        if present_direct_types and (
            len(present_direct_types) != 1
            or any(row["type"] not in present_direct_types for row in normalized)
        ):
            raise WorkspaceError("draft_evaluated_operation_combination_unsupported", 409)
        if present_direct_types == {DraftOperationTypeV2.SET_STOP_DURATION.value}:
            duration_modes = {
                (row.get("parameters", {}).get("duration") or {}).get("mode")
                for row in normalized
            }
            if "exact" in duration_modes and len(duration_modes) > 1:
                raise WorkspaceError("draft_evaluated_operation_combination_unsupported", 409)
        self._validate_operations(normalized)
        self._reject_conflicts(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_ORDER.value for row in normalized):
            return self._compile_stop_order(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_ROLE.value for row in normalized):
            return self._compile_stop_role(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_DURATION.value for row in normalized):
            return self._compile_stop_duration(normalized)
        if any(
            row["type"] == DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value
            for row in normalized
        ):
            return self._compile_stop_time_window(normalized)
        request, exact_candidate = self._compile_request(normalized)
        result = self._execute(request)
        if len(result.output_plans) == 0:
            return CompiledDraftPreview(
                state="ineligible",
                reason="no_feasible_evaluated_child",
                source_request_id=request.request_id,
                operations=normalized,
                source_bundle=self._base_bundle,
            )
        if len(result.output_plans) != 1 or len(result.diff_records) != 1 or len(result.evaluations) != 1:
            raise WorkspaceError("draft_pipeline_artifact_invalid", 409)
        child = _record(result.output_plans[0])
        diff = _record(result.diff_records[0])
        certificate = _record(result.evaluations[0])
        self._verify_pipeline_artifacts(child, diff, certificate)
        if exact_candidate is not None:
            self._cross_check_frozen_candidate(exact_candidate, child, diff, certificate)
        semantic_failures = self._semantic_failures(normalized, child)
        eligible = (
            not semantic_failures
            and certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status") in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else (semantic_failures[0] if semantic_failures else "independent_evaluation_failed"),
            source_request_id=request.request_id,
            operations=normalized,
            source_bundle=exact_candidate.bundle if exact_candidate else self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
        )

    def _build_candidates(self, bundles_by_plan: dict[str, Any]) -> dict[str, ExecutableCandidate]:
        candidates: dict[str, ExecutableCandidate] = {}
        for plan_id, bundle in bundles_by_plan.items():
            request = bundle.request or {}
            child = bundle.child_plan or {}
            diff = bundle.diff or {}
            candidate_rows = request.get("candidate_pois") or []
            added = diff.get("added_stops") or []
            deleted = diff.get("deleted_stops") or []
            if len(candidate_rows) != 1 or len(added) != 1 or len(deleted) != 1:
                raise WorkspaceError("draft_candidate_evidence_missing", 409)
            candidate_record = dict(candidate_rows[0])
            candidate_id = _stop_id(candidate_record)
            replaced_stop_id = _stop_id(deleted[0])
            target_ids = tuple(
                str(value)
                for value in (request.get("confirmed_constraints") or {}).get("target_stop_ids") or ()
            )
            if (
                not candidate_id
                or candidate_id != _stop_id(added[0])
                or replaced_stop_id not in self.parent_stop_ids
                or str(child.get("plan_id") or "") != plan_id
                or candidate_id not in _plan_stop_ids(child)
                or target_ids != (replaced_stop_id,)
                or not request.get("evidence_records")
            ):
                raise WorkspaceError("draft_candidate_lineage_mismatch", 409)
            if candidate_id in candidates:
                raise WorkspaceError("draft_candidate_not_distinct", 409)
            candidates[candidate_id] = ExecutableCandidate(
                candidate_id=candidate_id,
                label=str(
                    candidate_record.get("name")
                    or candidate_record.get("attraction_name")
                    or candidate_id
                ),
                replaced_stop_id=replaced_stop_id,
                child_plan_id=plan_id,
                source_request_id=str(request.get("request_id") or ""),
                record=candidate_record,
                bundle=bundle,
            )
        return candidates

    def _validate_operations(self, operations: tuple[dict[str, Any], ...]) -> None:
        for row in operations:
            operation_type = row["type"]
            target = row["target"]
            parameters = row["parameters"]
            if operation_type in DRAFT_ONLY_TYPED_EDIT_OPERATIONS:
                raise WorkspaceError("draft_contains_non_executable_operation", 409)
            if operation_type not in CANONICAL_OPERATIONS:
                raise WorkspaceError("draft_operation_unsupported", 409)
            if operation_type in {
                "keep_stop",
                "lock_stop",
                "mark_flexible",
                "move_day",
                DraftOperationTypeV2.SET_STOP_DAY.value,
                DraftOperationTypeV2.SET_STOP_ROLE.value,
                "replace_nearby",
            }:
                if target not in self.parent_stop_ids:
                    raise WorkspaceError("draft_target_not_in_parent", 409)
            if operation_type in {"keep_stop", "lock_stop", "mark_flexible"}:
                if parameters:
                    raise WorkspaceError("draft_operation_parameters_invalid", 409)
            elif operation_type in {"move_day", DraftOperationTypeV2.SET_STOP_DAY.value}:
                if set(parameters) != {"day"} or not _valid_day(parameters.get("day")):
                    raise WorkspaceError("draft_destination_day_invalid", 409)
                if (
                    operation_type == DraftOperationTypeV2.SET_STOP_DAY.value
                    and self._parent_day_by_stop.get(target) == parameters["day"]
                ):
                    raise WorkspaceError("draft_no_effect", 409)
            elif operation_type == DraftOperationTypeV2.SET_STOP_ORDER.value:
                if set(parameters) != {"day", "sequence_index"} or not _valid_day(
                    parameters.get("day")
                ):
                    raise WorkspaceError("draft_operation_parameters_invalid", 409)
                sequence_index = parameters.get("sequence_index")
                if (
                    isinstance(sequence_index, bool)
                    or not isinstance(sequence_index, int)
                    or not 0 <= sequence_index <= 63
                ):
                    raise WorkspaceError("draft_order_index_invalid", 409)
            elif operation_type == DraftOperationTypeV2.SET_STOP_ROLE.value:
                if set(parameters) != {"role"}:
                    raise WorkspaceError("invalid_stop_role", 422)
            elif operation_type == DraftOperationTypeV2.SET_STOP_DURATION.value:
                if set(parameters) != {"duration"}:
                    raise WorkspaceError("draft_operation_parameters_invalid", 409)
                duration = parameters.get("duration")
                if not isinstance(duration, dict):
                    raise WorkspaceError("invalid_stop_duration", 422)
                if duration.get("mode") != "exact":
                    raise WorkspaceError("duration_mode_evaluation_not_supported", 409)
            elif operation_type == "replace_nearby":
                if set(parameters) != {"candidate_id"}:
                    raise WorkspaceError("draft_operation_parameters_invalid", 409)
                if _candidate_parameter(parameters) not in self._candidates:
                    raise WorkspaceError("draft_candidate_not_registered", 409)
            elif operation_type == "add_candidate":
                if set(parameters) != {"day"} or not _valid_day(parameters.get("day")):
                    raise WorkspaceError("draft_destination_day_invalid", 409)
                if target not in self._candidates:
                    raise WorkspaceError("draft_candidate_not_registered", 409)
            elif operation_type == "route_feedback":
                if target != "selected_route" or set(parameters) - {"preference", "weight"}:
                    raise WorkspaceError("draft_route_feedback_invalid", 409)
                if str(parameters.get("preference") or "") not in _PREFERENCES:
                    raise WorkspaceError("draft_route_feedback_invalid", 409)
                weight = parameters.get("weight")
                if weight is not None and not _finite_unit_interval(weight):
                    raise WorkspaceError("draft_route_feedback_invalid", 409)

    @staticmethod
    def _reject_conflicts(operations: tuple[dict[str, Any], ...]) -> None:
        signatures = [
            stable_content_hash({"type": row["type"], "target": row["target"], "parameters": row["parameters"]})
            for row in operations
        ]
        if len(signatures) != len(set(signatures)):
            raise WorkspaceError("draft_duplicate_operation", 409)
        policies: dict[str, set[str]] = {}
        moves: dict[str, set[int]] = {}
        replacements: dict[str, set[str]] = {}
        orders: dict[str, set[tuple[int, int]]] = {}
        preferences: set[str] = set()
        for row in operations:
            kind = row["type"]
            if kind in {"keep_stop", "lock_stop", "mark_flexible"}:
                policies.setdefault(row["target"], set()).add(kind)
            elif kind in {"move_day", DraftOperationTypeV2.SET_STOP_DAY.value}:
                moves.setdefault(row["target"], set()).add(int(row["parameters"]["day"]))
            elif kind == "replace_nearby":
                replacements.setdefault(row["target"], set()).add(_candidate_parameter(row["parameters"]))
            elif kind == DraftOperationTypeV2.SET_STOP_ORDER.value:
                orders.setdefault(row["target"], set()).add(
                    (int(row["parameters"]["day"]), int(row["parameters"]["sequence_index"]))
                )
            elif kind == "route_feedback":
                preferences.add(str(row["parameters"]["preference"]))
        if any(
            "mark_flexible" in values and values.intersection({"keep_stop", "lock_stop"})
            for values in policies.values()
        ):
            raise WorkspaceError("draft_conflicting_stop_policy", 409)
        if any(
            target in replacements and values.intersection({"keep_stop", "lock_stop"})
            for target, values in policies.items()
        ):
            raise WorkspaceError("draft_conflicts_with_replacement", 409)
        if any(len(days) > 1 for days in moves.values()):
            raise WorkspaceError("draft_conflicting_day_moves", 409)
        if any(len(candidates) > 1 for candidates in replacements.values()):
            raise WorkspaceError("draft_conflicting_replacements", 409)
        if any(len(values) > 1 for values in orders.values()):
            raise WorkspaceError("draft_conflicting_stop_order", 409)
        if len(preferences) > 1:
            raise WorkspaceError("draft_conflicting_route_feedback", 409)
        requested_candidates = {
            candidate
            for values in replacements.values()
            for candidate in values
        } | {row["target"] for row in operations if row["type"] == "add_candidate"}
        if len(requested_candidates) > 1:
            raise WorkspaceError("draft_conflicting_candidates", 409)
        if preferences == {"reduce_driving"} and requested_candidates and not requested_candidates.issubset(
            {"santa_barbara_museum_of_natural_history_sea_center"}
        ):
            raise WorkspaceError("draft_conflicting_preferences", 409)
        if preferences == {"preserve_experience"} and requested_candidates and not requested_candidates.issubset(
            {"bixby_creek_bridge_viewpoint"}
        ):
            raise WorkspaceError("draft_conflicting_preferences", 409)

    def _compile_request(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> tuple[RepairRequest, ExecutableCandidate | None]:
        exact_candidate = self._exact_replacement(operations)
        source = exact_candidate.bundle.request if exact_candidate else self._base_bundle.request
        constraints = dict(source.get("confirmed_constraints") or {})
        must_include = set(str(value) for value in constraints.get("must_include") or ())
        must_delete = set(str(value) for value in constraints.get("must_delete") or ())
        must_keep: set[str] = set()
        requested_candidates: set[str] = set()
        affected_days = set(int(value) for value in constraints.get("affected_days") or ())
        preferences: list[str] = []
        day_moves: dict[str, int] = {}
        flexible: set[str] = set()
        locked: set[str] = set()
        for row in operations:
            kind = row["type"]
            target = row["target"]
            if kind == "keep_stop":
                must_keep.add(target)
                must_include.add(target)
            elif kind == "lock_stop":
                must_keep.add(target)
                must_include.add(target)
                locked.add(target)
            elif kind == "mark_flexible":
                flexible.add(target)
            elif kind in {"move_day", DraftOperationTypeV2.SET_STOP_DAY.value}:
                day_moves[target] = int(row["parameters"]["day"])
                affected_days.add(int(row["parameters"]["day"]))
            elif kind == "route_feedback":
                preferences.append(str(row["parameters"]["preference"]))
            elif kind == "replace_nearby":
                must_delete.add(target)
                candidate_id = _candidate_parameter(row["parameters"])
                must_include.add(candidate_id)
                requested_candidates.add(candidate_id)
            elif kind == "add_candidate":
                must_include.add(target)
                requested_candidates.add(target)
                affected_days.add(int(row["parameters"]["day"]))
        if not requested_candidates and preferences == ["reduce_driving"]:
            requested_candidates.add("santa_barbara_museum_of_natural_history_sea_center")
        elif not requested_candidates and preferences == ["preserve_experience"]:
            requested_candidates.add("bixby_creek_bridge_viewpoint")
        candidate_ids = requested_candidates or set(self._candidates)
        candidate_rows = tuple(dict(self._candidates[candidate_id].record) for candidate_id in sorted(candidate_ids))
        constraints.update(
            {
                "affected_days": sorted(affected_days),
                "must_include": sorted(must_include),
                "must_delete": sorted(must_delete),
                "locked_stop_ids": sorted(locked),
                "flexible_stop_ids": sorted(flexible),
                "move_stops": dict(sorted(day_moves.items())),
                "route_feedback": preferences,
            }
        )

        if exact_candidate:
            request_id = exact_candidate.source_request_id
            user_intent = str(source.get("user_intent") or "")
        else:
            semantic_operations = [
                {
                    "type": row["type"],
                    "target": row["target"],
                    "parameters": row["parameters"],
                }
                for row in operations
            ]
            request_id = f"w3_preview_{stable_content_hash({'parent': self.parent_plan_id, 'operations': semantic_operations})}"
            user_intent = "Execute the confirmed typed itinerary draft through deterministic repair."
        parsed_intent = ParsedRepairIntent(
            raw_text=user_intent,
            confirmed=True,
            confidence=1.0,
            must_keep=tuple(sorted(must_keep)),
            must_delete=tuple(sorted(must_delete)),
            must_include=tuple(sorted(must_include)),
            parser_name="typed_draft_compiler_v1",
        )
        return (
            RepairRequest(
                request_id=request_id,
                baseline_route=tuple(dict(row) for row in source.get("baseline_route") or ()),
                user_intent=user_intent,
                evidence_records=tuple(dict(row) for row in source.get("evidence_records") or ()),
                tolerance_profile=dict(source.get("tolerance_profile") or {}),
                confirmed_constraints=constraints,
                candidate_pois=candidate_rows,
                travel_graph=dict(source.get("travel_graph") or {}),
                parsed_intent=parsed_intent,
            ),
            exact_candidate,
        )

    def _compile_stop_order(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> CompiledDraftPreview:
        route_matrix, day_config, _ = self._runtime_inputs()
        artifacts = EvaluatedStopOrderCompiler(
            parent=self._parent_artifact,
            route_matrix=route_matrix,
            start_anchor_by_day=day_config.start_anchor_by_day,
            end_anchor_by_day=day_config.end_anchor_by_day,
            max_day_minutes=day_config.max_day_minutes,
            default_visit_minutes=day_config.default_visit_minutes,
        ).compile(operations)
        child = artifacts.child_plan.to_record()
        diff = artifacts.diff.to_record()
        certificate = artifacts.certificate.to_record()
        self._verify_pipeline_artifacts(child, diff, certificate)
        eligible = (
            certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status") in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else "independent_evaluation_failed",
            source_request_id=artifacts.source_request_id,
            operations=operations,
            source_bundle=self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=artifacts.planner_run.to_record(),
            parent_route_legs=artifacts.parent_route_legs,
            route_legs=artifacts.route_legs,
            schedule_impact=artifacts.schedule_impact,
            execution_mode="typed_direct_edit_independent_evaluation",
        )

    def _compile_stop_duration(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> CompiledDraftPreview:
        route_matrix, day_config, _ = self._runtime_inputs()
        artifacts = EvaluatedStopDurationCompiler(
            parent=self._parent_artifact,
            route_matrix=route_matrix,
            start_anchor_by_day=day_config.start_anchor_by_day,
            end_anchor_by_day=day_config.end_anchor_by_day,
            max_day_minutes=day_config.max_day_minutes,
            default_visit_minutes=day_config.default_visit_minutes,
        ).compile(operations)
        child = artifacts.child_plan.to_record()
        diff = artifacts.diff.to_record()
        certificate = artifacts.certificate.to_record()
        self._verify_pipeline_artifacts(child, diff, certificate)
        eligible = (
            certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status") in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else "independent_evaluation_failed",
            source_request_id=artifacts.source_request_id,
            operations=operations,
            source_bundle=self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=artifacts.planner_run.to_record(),
            parent_route_legs=artifacts.parent_route_legs,
            route_legs=artifacts.route_legs,
            schedule_impact=artifacts.schedule_impact,
            execution_mode="typed_direct_edit_independent_evaluation",
        )

    def _compile_stop_role(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> CompiledDraftPreview:
        route_matrix, day_config, _ = self._runtime_inputs()
        artifacts = EvaluatedStopRoleCompiler(
            parent=self._parent_artifact,
            route_matrix=route_matrix,
            start_anchor_by_day=day_config.start_anchor_by_day,
            end_anchor_by_day=day_config.end_anchor_by_day,
            max_day_minutes=day_config.max_day_minutes,
            default_visit_minutes=day_config.default_visit_minutes,
            day_start_time=day_config.day_start_time,
        ).compile(operations)
        child = artifacts.child_plan.to_record()
        diff = artifacts.diff.to_record()
        certificate = artifacts.certificate.to_record()
        self._verify_pipeline_artifacts(child, diff, certificate)
        eligible = (
            certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status")
            in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else "independent_evaluation_failed",
            source_request_id=artifacts.source_request_id,
            operations=operations,
            source_bundle=self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=artifacts.planner_run.to_record(),
            parent_route_legs=artifacts.parent_route_legs,
            route_legs=artifacts.route_legs,
            schedule_impact=artifacts.schedule_impact,
            execution_mode="typed_direct_edit_independent_evaluation",
        )

    def _compile_stop_time_window(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> CompiledDraftPreview:
        route_matrix, day_config, _ = self._runtime_inputs()
        artifacts = EvaluatedStopTimeWindowCompiler(
            parent=self._parent_artifact,
            route_matrix=route_matrix,
            start_anchor_by_day=day_config.start_anchor_by_day,
            end_anchor_by_day=day_config.end_anchor_by_day,
            max_day_minutes=day_config.max_day_minutes,
            default_visit_minutes=day_config.default_visit_minutes,
            day_start_time=day_config.day_start_time,
        ).compile(operations)
        child = artifacts.child_plan.to_record()
        diff = artifacts.diff.to_record()
        certificate = artifacts.certificate.to_record()
        self._verify_pipeline_artifacts(child, diff, certificate)
        eligible = (
            certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status")
            in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else "independent_evaluation_failed",
            source_request_id=artifacts.source_request_id,
            operations=operations,
            source_bundle=self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=artifacts.planner_run.to_record(),
            parent_route_legs=artifacts.parent_route_legs,
            route_legs=artifacts.route_legs,
            schedule_impact=artifacts.schedule_impact,
            execution_mode="typed_direct_edit_independent_evaluation",
        )

    def _exact_replacement(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> ExecutableCandidate | None:
        if len(operations) != 1 or operations[0]["type"] != "replace_nearby":
            return None
        row = operations[0]
        candidate = self._candidates[_candidate_parameter(row["parameters"])]
        if row["target"] != candidate.replaced_stop_id:
            raise WorkspaceError("draft_candidate_target_mismatch", 409)
        return candidate

    def _execute(self, request: RepairRequest) -> Any:
        try:
            route_matrix, day_config, trip_config = self._runtime_inputs()
            context = PipelineRunContext(
                run_id=request.request_id,
                mode="repair",
                config=trip_config,
                catalog_snapshot_id=self._parent_artifact.catalog_snapshot_id,
                context_snapshot_id=self._parent_artifact.context_snapshot_id,
                output_dir=Path("<in-memory-w3-preview>"),
                refresh_policy=RefreshPolicy.NEVER,
                parent_plan_id=self.parent_plan_id,
                repair_request_id=request.request_id,
            )
            return run_deterministic_context_aware_heuristic_executor(
                context,
                parent_plan=self._parent_artifact,
                repair_request=request,
                route_matrix=route_matrix,
                day_route_config=day_config,
                publication_mode=True,
            )
        except Exception as exc:
            raise WorkspaceError("draft_pipeline_execution_failed", 409) from exc

    def _runtime_inputs(self) -> tuple[RouteMatrix, DayRouteSolverConfig, Any]:
        with self._runtime_cache_lock:
            if self._runtime_cache is None:
                route_matrix = _route_matrix(self._base_bundle.route_matrix or {})
                day_config = DayRouteSolverConfig(
                    max_day_minutes=720.0,
                    default_visit_minutes=45.0,
                    strict_route_matrix=True,
                    start_anchor_by_day={day: anchors[0] for day, anchors in _DAY_ANCHORS.items()},
                    end_anchor_by_day={day: anchors[1] for day, anchors in _DAY_ANCHORS.items()},
                )
                trip_config = load_trip_config(
                    self.repository_root / "configs" / "default_trip_config.yaml"
                )
                self._runtime_cache = (route_matrix, day_config, trip_config)
            return self._runtime_cache

    def _verify_pipeline_artifacts(
        self,
        child: dict[str, Any],
        diff: dict[str, Any],
        certificate: dict[str, Any],
    ) -> None:
        child_payload = dict(child)
        child_hash = str(child_payload.pop("content_hash", ""))
        certificate_payload = dict(certificate)
        certificate_hash = str(certificate_payload.pop("content_hash", ""))
        if (
            stable_content_hash(child_payload) != child_hash
            or child.get("parent_plan_id") != self.parent_plan_id
            or diff.get("parent_plan_id") != self.parent_plan_id
            or diff.get("child_plan_id") != child.get("plan_id")
            or certificate.get("plan_id") != child.get("plan_id")
            or certificate.get("plan_content_hash") != child_hash
            or stable_content_hash(certificate_payload) != certificate_hash
        ):
            raise WorkspaceError("draft_pipeline_lineage_mismatch", 409)
        _verify_diff(diff)
        _verify_certificate_id(certificate)
        route_validation = certificate.get("route_validation") or {}
        matrix = self._base_bundle.route_matrix or {}
        if (
            route_validation.get("publication_ready") is not True
            or route_validation.get("fallback_leg_count") != 0
            or route_validation.get("missing_leg_count") != 0
            or route_validation.get("road_validated_leg_count")
            != route_validation.get("required_leg_count")
            or route_validation.get("matrix_id") != matrix.get("matrix_id")
            or route_validation.get("source_bundle_id") != matrix.get("source_bundle_id")
            or route_validation.get("source_content_sha256") != matrix.get("source_content_sha256")
        ):
            raise WorkspaceError("draft_candidate_route_evidence_missing", 409)

    @staticmethod
    def _cross_check_frozen_candidate(
        candidate: ExecutableCandidate,
        child: dict[str, Any],
        diff: dict[str, Any],
        certificate: dict[str, Any],
    ) -> None:
        fields = (
            "certificate_id",
            "plan_id",
            "plan_content_hash",
            "source_run_id",
            "evaluation_status",
            "comparison_eligibility",
            "failures",
            "warnings",
            "route_validation",
        )
        if (
            child != candidate.bundle.child_plan
            or diff != candidate.bundle.diff
            or any(certificate.get(field) != candidate.bundle.certificate.get(field) for field in fields)
        ):
            raise WorkspaceError("draft_pipeline_artifact_mismatch", 409)

    @staticmethod
    def _semantic_failures(
        operations: tuple[dict[str, Any], ...],
        child: dict[str, Any],
    ) -> tuple[str, ...]:
        day_by_stop = {
            str(stop_id): int(day)
            for day, stop_ids in (
                (row.get("day"), row.get("stop_ids") or ()) for row in child.get("ordered_days") or ()
            )
            for stop_id in stop_ids
        }
        failures: list[str] = []
        for row in operations:
            kind = row["type"]
            target = row["target"]
            if kind in {"keep_stop", "lock_stop"} and target not in day_by_stop:
                failures.append("required_stop_not_preserved")
            elif kind in {"move_day", DraftOperationTypeV2.SET_STOP_DAY.value} and (
                day_by_stop.get(target) != row["parameters"]["day"]
            ):
                failures.append("requested_day_move_not_satisfied")
            elif kind == "replace_nearby":
                candidate_id = _candidate_parameter(row["parameters"])
                if target in day_by_stop or candidate_id not in day_by_stop:
                    failures.append("requested_replacement_not_satisfied")
            elif kind == "add_candidate":
                if day_by_stop.get(target) != row["parameters"]["day"]:
                    failures.append("requested_candidate_addition_not_satisfied")
        return tuple(dict.fromkeys(failures))


def _operation_record(operation: Any) -> dict[str, Any]:
    if hasattr(operation, "as_dict"):
        raw = operation.as_dict()
    elif isinstance(operation, dict):
        raw = operation
    else:
        raise WorkspaceError("draft_operation_invalid", 409)
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise WorkspaceError("draft_operation_invalid", 409)
    return {
        "operation_id": str(raw.get("operation_id") or ""),
        "type": str(raw.get("type") or ""),
        "target": str(raw.get("target") or ""),
        "parameters": dict(parameters),
        "source": str(raw.get("source") or ""),
        "evidence_refs": list(raw.get("evidence_refs") or ()),
    }


def _candidate_parameter(parameters: dict[str, Any]) -> str:
    candidate_id = str(parameters.get("candidate_id") or "").strip()
    if not candidate_id:
        raise WorkspaceError("draft_candidate_required", 409)
    return candidate_id


def _stop_id(record: dict[str, Any]) -> str:
    return str(record.get("stop_id") or record.get("poi_id") or "").strip()


def _plan_stop_ids(plan: dict[str, Any]) -> set[str]:
    return {_stop_id(row) for row in plan.get("selected_stops") or [] if _stop_id(row)}


def _valid_day(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 7


def _finite_unit_interval(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= float(value) <= 1.0
    )


def _plan_artifact(record: dict[str, Any]) -> PlanArtifactV2:
    plan = PlanArtifactV2(
        plan_id=str(record.get("plan_id") or ""),
        parent_plan_id=record.get("parent_plan_id"),
        source_run_id=str(record.get("source_run_id") or ""),
        planning_request_id=str(record.get("planning_request_id") or ""),
        catalog_snapshot_id=str(record.get("catalog_snapshot_id") or ""),
        context_snapshot_id=str(record.get("context_snapshot_id") or ""),
        selected_stops=tuple(dict(row) for row in record.get("selected_stops") or ()),
        day_assignments={str(key): int(value) for key, value in (record.get("day_assignments") or {}).items()},
        sequence=tuple(str(value) for value in record.get("sequence") or ()),
        lodging_assignments={
            str(key): str(value) for key, value in (record.get("lodging_assignments") or {}).items()
        },
        ordered_days=tuple(dict(row) for row in record.get("ordered_days") or ()),
        route_ids_by_day={
            int(key): str(value) for key, value in (record.get("route_ids_by_day") or {}).items()
        },
        owned_constraints=tuple(dict(row) for row in record.get("owned_constraints") or ()),
        modeled_metrics=dict(record.get("modeled_metrics") or {}),
        context_exposure_components=dict(record.get("context_exposure_components") or {}),
        change_components=dict(record.get("change_components") or {}),
        certificate_id=record.get("certificate_id"),
        created_at=str(record.get("created_at") or ""),
    )
    if plan.content_hash != record.get("content_hash"):
        raise WorkspaceError("draft_parent_hash_mismatch", 409)
    return plan


def _route_matrix(record: dict[str, Any]) -> RouteMatrix:
    cells: dict[tuple[str, str], RouteMatrixCell] = {}
    for raw_cell in record.get("cells") or ():
        cell_record = dict(raw_cell)
        cell_record["geometry"] = tuple(tuple(point) for point in cell_record.get("geometry") or ())
        cell = RouteMatrixCell(**cell_record)
        cells[(cell.origin_id, cell.destination_id)] = cell
    return RouteMatrix(
        matrix_id=str(record.get("matrix_id") or ""),
        context_snapshot_id=str(record.get("context_snapshot_id") or ""),
        entity_ids=tuple(str(value) for value in record.get("entity_ids") or ()),
        cells=cells,
        source_bundle_id=str(record.get("source_bundle_id") or ""),
        source_content_sha256=str(record.get("source_content_sha256") or ""),
    )


def _record(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        record = value.to_record()
    elif isinstance(value, dict):
        record = dict(value)
    else:
        raise WorkspaceError("draft_pipeline_artifact_invalid", 409)
    if not isinstance(record, dict):
        raise WorkspaceError("draft_pipeline_artifact_invalid", 409)
    return record


def _verify_diff(diff: dict[str, Any]) -> None:
    groups = [
        ("added_stops", "added"),
        ("deleted_stops", "deleted"),
        ("day_moves", "day_moves"),
        ("time_shifts", "time_shifts"),
        ("reorder_changes", "reorder"),
        ("lodging_changes", "lodging"),
        ("road_changes", "road"),
    ]
    schema_version = diff.get("schema_version")
    if schema_version == "plan-diff-v2":
        if not isinstance(diff.get("duration_changes"), list) or not diff["duration_changes"]:
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        groups.append(("duration_changes", "duration_changes"))
    elif schema_version == "plan-diff-v3":
        if (
            not isinstance(diff.get("time_window_changes"), list)
            or not diff["time_window_changes"]
            or "duration_changes" in diff
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        groups.append(("time_window_changes", "time_window_changes"))
    elif schema_version == "plan-diff-v4":
        if (
            not isinstance(diff.get("role_changes"), list)
            or not diff["role_changes"]
            or "duration_changes" in diff
            or "time_window_changes" in diff
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        groups.append(("role_changes", "role_changes"))
    elif (
        schema_version != "plan-diff-v1"
        or "duration_changes" in diff
        or "time_window_changes" in diff
        or "role_changes" in diff
    ):
        raise WorkspaceError("draft_pipeline_diff_invalid", 409)
    weighted = diff.get("weighted_edit_cost")
    if not isinstance(weighted, (int, float)) or isinstance(weighted, bool) or not math.isfinite(weighted):
        raise WorkspaceError("draft_pipeline_diff_invalid", 409)
    seed = {
        "parent_plan_id": diff.get("parent_plan_id"),
        "child_plan_id": diff.get("child_plan_id"),
        **{seed_key: diff.get(record_key) or [] for record_key, seed_key in groups},
        "unchanged_days": diff.get("unchanged_days") or [],
        "weighted_edit_cost": float(weighted),
    }
    if diff.get("diff_id") != f"diff_{stable_content_hash(seed)}":
        raise WorkspaceError("draft_pipeline_diff_invalid", 409)


def _verify_certificate_id(certificate: dict[str, Any]) -> None:
    seed = {
        "plan_id": certificate.get("plan_id"),
        "plan_content_hash": certificate.get("plan_content_hash"),
        "source_run_id": certificate.get("source_run_id"),
        "failures": [row.get("code") for row in certificate.get("failures") or []],
        "warnings": [row.get("code") for row in certificate.get("warnings") or []],
    }
    if certificate.get("certificate_id") != f"cert_{stable_content_hash(seed)}":
        raise WorkspaceError("draft_pipeline_certificate_invalid", 409)
