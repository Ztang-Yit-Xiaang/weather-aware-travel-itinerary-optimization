"""Compile typed drafts into deterministic repair and evaluation requests."""

from __future__ import annotations

import math
import re
import threading
from ast import literal_eval
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import load_trip_config
from ..evaluation import PlanEvaluator, PlanEvaluatorConfig
from ..pipeline_runner import (
    PipelineRunContext,
    RefreshPolicy,
    run_deterministic_context_aware_heuristic_executor,
)
from ..plans.attribute_ownership import attribute_ownership_rows
from ..plans.diff import PlanDiffBuilder
from ..repair.context import context_evaluation_requirements
from ..repair.day_route_solver import DayRouteSolverConfig
from ..repair_planner import ParsedRepairIntent, RepairRequest
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixCell, RouteMatrixCellMissing, RouteMatrixNotPublicationEligible
from .evaluated_stop_edits import (
    EvaluatedStopDurationCompiler,
    EvaluatedStopOrderCompiler,
    EvaluatedStopRoleCompiler,
    EvaluatedStopTimeWindowCompiler,
    _route_legs,
)
from .interaction_contracts import (
    ATOMIC_COMPOSABLE_OPERATIONS,
    ATOMIC_DIRECT_EDIT_OPERATIONS,
    DRAFT_ONLY_TYPED_EDIT_OPERATIONS,
    DraftOperationTypeV2,
)
from .workspace import WorkspaceError

CANONICAL_OPERATIONS = frozenset(
    {
        "replan_trip",
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
    "replan_trip",
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
    runtime_execution: dict[str, Any] | None = None
    blocking_findings: tuple[dict[str, Any], ...] = ()


class FrozenDraftCompiler:
    """Compile the canonical vocabulary against a closed artifact-backed universe."""

    def __init__(
        self,
        parent_plan: dict[str, Any],
        bundles_by_plan: dict[str, Any],
        *,
        repository_root: Path,
        route_matrix: dict[str, Any] | None = None,
        planning_state_root: Path | None = None,
        planning_routing_url: str | None = None,
    ) -> None:
        self.planning_state_root = planning_state_root
        self.planning_routing_url = planning_routing_url
        self._planning_baseline_bundle = None
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
        source = next(iter(bundles_by_plan.values()))
        self._base_bundle = replace(source, parent_plan=parent_plan, route_matrix=route_matrix or source.route_matrix)
        self._coherent_product = (self._base_bundle.run_manifest or {}).get(
            "schema_version"
        ) == "product-demo-package-v2"
        self._runtime_cache: tuple[RouteMatrix, DayRouteSolverConfig, Any] | None = None
        self._runtime_cache_lock = threading.RLock()

    def for_parent(
        self, parent_plan: dict[str, Any], *, planning_bundle: dict[str, Any] | None = None
    ) -> FrozenDraftCompiler:
        """Bind verified current plan data to the same immutable evidence inputs.

        The service validates acceptance authority. Candidate source lineage
        remains its registered parent; it is not rewritten as a new repair.
        """
        parent = _plan_artifact(parent_plan)
        if planning_bundle is None and (
            parent.context_snapshot_id != self._parent_artifact.context_snapshot_id
            or parent.catalog_snapshot_id != self._parent_artifact.catalog_snapshot_id
        ):
            raise WorkspaceError("draft_baseline_context_mismatch", 409)
        compiler = FrozenDraftCompiler(
            parent_plan,
            self._bundles_by_plan,
            repository_root=self.repository_root,
            route_matrix=planning_bundle["route_matrix"] if planning_bundle else self._base_bundle.route_matrix,
            planning_state_root=self.planning_state_root,
            planning_routing_url=self.planning_routing_url,
        )
        if planning_bundle:
            if (
                parent.context_snapshot_id != planning_bundle["plan"]["context_snapshot_id"]
                or parent.catalog_snapshot_id != planning_bundle["plan"]["catalog_snapshot_id"]
            ):
                raise WorkspaceError("draft_baseline_context_mismatch", 409)
            compiler._planning_baseline_bundle = planning_bundle
            compiler._candidates = {
                key: value
                for key, value in compiler._candidates.items()
                if key not in set(planning_bundle["excluded_stop_ids"])
            }
            compiler._base_bundle = replace(
                compiler._base_bundle,
                run_manifest={
                    **compiler._base_bundle.run_manifest,
                    "scenario": {
                        **compiler._base_bundle.run_manifest["scenario"],
                        "days": planning_bundle["days"],
                        "place_sources": planning_bundle["place_sources"],
                    },
                },
            )
        else:
            compiler._runtime_cache = self._runtime_cache
        return compiler

    def capabilities(self) -> dict[str, Any]:
        choices = [
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "replaces_stop_id": candidate.replaced_stop_id,
                "child_plan_id": candidate.child_plan_id,
            }
            for candidate in sorted(self._candidates.values(), key=lambda item: item.candidate_id)
            if candidate.candidate_id not in self.parent_stop_ids
        ]
        capabilities = {
            "schema_version": "draft-capabilities-v1",
            "operations": {
                name: {
                    "enabled": not self._unsupported_product_operation(name),
                    "preview_executable": not self._unsupported_product_operation(name),
                    "status": (
                        "unsupported_optimizer_mapping"
                        if self._unsupported_product_operation(name)
                        else "deterministic_pipeline"
                    ),
                }
                for name in sorted(LEGACY_CAPABILITY_OPERATIONS)
            },
            "parent_targets": sorted(self.parent_stop_ids),
            "candidate_choices": choices,
            "executable_combinations": [
                {
                    "operations": sorted(
                        name for name in LEGACY_CAPABILITY_OPERATIONS if not self._unsupported_product_operation(name)
                    ),
                    "cardinality": 32,
                    "description": (
                        "Ordered canonical drafts run through deterministic repair and independent evaluation; "
                        "conflicts fail closed."
                    ),
                }
            ],
        }
        if self._coherent_product:
            capabilities["executable_combinations"][0]["operations"] = sorted(ATOMIC_COMPOSABLE_OPERATIONS)
            readiness = [
                self.candidate_addition_readiness(candidate_id, day)
                for candidate_id in sorted(self._candidates.keys() - self.parent_stop_ids)
                for day in sorted(row["day"] for row in self.parent_plan["ordered_days"])
            ]
            ready = any(row["preview_executable"] for row in readiness)
            capabilities["operations"]["add_candidate"] = {
                "enabled": ready,
                "preview_executable": ready,
                "status": "route_evidence_ready"
                if ready
                else (
                    "blocked_missing_route_evidence"
                    if all(row["status"] == "blocked_missing_route_evidence" for row in readiness)
                    else "blocked_invalid_route_evidence"
                ),
                "candidate_day_readiness": readiness,
            }
            if not ready:
                capabilities["executable_combinations"][0]["operations"].remove("add_candidate")
            if not any(candidate.replaced_stop_id in self.parent_stop_ids for candidate in self._candidates.values()):
                capabilities["operations"]["replace_nearby"] = {
                    "enabled": False,
                    "preview_executable": False,
                    "status": "no_applicable_replacement_candidates",
                }
                capabilities["executable_combinations"][0]["operations"].remove("replace_nearby")
        if self._coherent_product and self.planning_state_root is not None:
            capabilities["operations"]["replan_trip"] = {
                "enabled": True,
                "preview_executable": True,
                "status": "fetch_road_and_official_evidence_then_solve",
            }
            capabilities["executable_combinations"].append(
                {
                    "operations": ["replan_trip"],
                    "cardinality": 1,
                    "description": "Whole-trip planning runs as one separate draft, with fresh road evidence and independent evaluation.",
                }
            )
        return capabilities

    def candidate_addition_readiness(self, candidate_id: str, day: int) -> dict[str, Any]:
        """Check the exact append route; readiness is not independent eligibility.

        Replacement evidence is insufficient for addition: existing stops remain
        in their original order and the candidate precedes the day's end anchor.
        This check is deliberately not cached separately from canonical inputs.
        """
        if candidate_id not in self._candidates:
            raise WorkspaceError("draft_candidate_not_registered", 409)
        if candidate_id in self.parent_stop_ids:
            raise WorkspaceError("draft_candidate_already_in_plan", 409)
        if not _valid_day(day) or not any(row["day"] == day for row in self.parent_plan["ordered_days"]):
            raise WorkspaceError("draft_destination_day_invalid", 409)
        matrix, config, _ = self._runtime_inputs()
        stops = next(row["stop_ids"] for row in self.parent_plan["ordered_days"] if row["day"] == day)
        sequence = [config.start_anchor_by_day[day], *stops, candidate_id, config.end_anchor_by_day[day]]
        required_legs: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        for origin, destination in zip(sequence[:-1], sequence[1:], strict=True):
            leg: dict[str, Any] = {"origin_id": origin, "destination_id": destination}
            code = None
            try:
                cell = matrix.cell(origin, destination)
                cell.require_publication_eligible()
                if (
                    cell.context_snapshot_id != matrix.context_snapshot_id
                    or matrix.context_snapshot_id != self._parent_artifact.context_snapshot_id
                ):
                    code = "route_context_mismatch"
                elif origin != destination and len(cell.geometry) < 2:
                    code = "route_geometry_missing"
                elif any(
                    len(point) != 2
                    or any(
                        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                        for value in point
                    )
                    or not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180)
                    for point in cell.geometry
                ):
                    code = "route_geometry_invalid"
                leg["route_leg_id"] = cell.route_leg_id
            except RouteMatrixCellMissing:
                code = "missing_route_matrix_cell"
            except RouteMatrixNotPublicationEligible:
                code = "route_leg_not_road_validated"
            leg["ready"] = code is None
            required_legs.append(leg)
            if code:
                findings.append(
                    {
                        "code": code,
                        "origin_id": origin,
                        "destination_id": destination,
                        "matrix_id": matrix.matrix_id,
                        "context_snapshot_id": matrix.context_snapshot_id,
                        "corrective_action": (
                            "Provide road-validated geometry and travel evidence for this directed leg "
                            "in the current trip context, then preview again."
                        ),
                    }
                )
        status = "route_evidence_ready"
        if findings:
            status = (
                "blocked_missing_route_evidence"
                if all(row["code"] == "missing_route_matrix_cell" for row in findings)
                else "blocked_invalid_route_evidence"
            )
        return {
            "candidate_id": candidate_id,
            "day": day,
            "parent_plan_id": self.parent_plan_id,
            "parent_plan_content_hash": self.parent_plan["content_hash"],
            "context_snapshot_id": self._parent_artifact.context_snapshot_id,
            "matrix_id": matrix.matrix_id,
            "insertion_mode": "append_after_day_stops",
            "route_sequence": sequence,
            "required_legs": required_legs,
            "preview_executable": not findings,
            "status": status,
            "comparison_eligibility": "not_evaluated",
            "blocking_findings": findings,
        }

    def compile(
        self,
        operations: Iterable[Any],
        *,
        accepted_plan_id: str,
        read_only: bool = False,
        refresh_expired: bool = False,
    ) -> CompiledDraftPreview:
        if accepted_plan_id != self.parent_plan_id:
            raise WorkspaceError("draft_parent_not_supported", 409)
        normalized = tuple(_operation_record(operation) for operation in operations)
        if not normalized:
            raise WorkspaceError("draft_is_empty")
        if any(row["type"] == "replan_trip" for row in normalized):
            from .trip_replanner import compile_trip_replan

            self._validate_operations(normalized)
            return compile_trip_replan(self, normalized, read_only=read_only, refresh_expired=refresh_expired)
        if refresh_expired:
            raise WorkspaceError("trip_planning_refresh_requires_replan", 409)
        evaluated_direct_types = ATOMIC_DIRECT_EDIT_OPERATIONS
        present_direct_types = {row["type"] for row in normalized} & evaluated_direct_types
        combined = bool(present_direct_types) and (
            len(present_direct_types) != 1 or any(row["type"] not in present_direct_types for row in normalized)
        )
        # Owned schedule edits need the complete child/rule receipt even when
        # the draft contains only one family. Reuse the canonical atomic path.
        schedule_attributes = {"set_stop_duration": "duration", "set_stop_time_window": "time_window"}
        if self._coherent_product and not combined and present_direct_types <= schedule_attributes.keys():
            changed_keys = frozenset(
                (row["target"], schedule_attributes[row["type"]])
                for row in normalized
                if row["type"] in schedule_attributes
            )
            combined = bool(attribute_ownership_rows(self._parent_artifact.owned_constraints, changed_keys))
        if combined and (
            not self._coherent_product or any(row["type"] not in ATOMIC_COMPOSABLE_OPERATIONS for row in normalized)
        ):
            raise WorkspaceError("draft_evaluated_operation_combination_unsupported", 409)
        if present_direct_types == {DraftOperationTypeV2.SET_STOP_DURATION.value}:
            duration_modes = {
                (row.get("parameters", {}).get("duration") or {}).get("mode")
                for row in normalized
                if row["type"] == DraftOperationTypeV2.SET_STOP_DURATION.value
            }
            if "exact" in duration_modes and len(duration_modes) > 1:
                raise WorkspaceError("draft_evaluated_operation_combination_unsupported", 409)
        self._validate_operations(normalized)
        self._reject_conflicts(normalized)
        if combined:
            return self._compile_combined(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_ORDER.value for row in normalized):
            return self._compile_stop_order(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_ROLE.value for row in normalized):
            return self._compile_stop_role(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_DURATION.value for row in normalized):
            return self._compile_stop_duration(normalized)
        if any(row["type"] == DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value for row in normalized):
            return self._compile_stop_time_window(normalized)
        request, exact_candidate = self._compile_request(normalized)
        started_at = datetime.now(UTC).isoformat()
        result = self._execute(request)
        completed_at = datetime.now(UTC).isoformat()
        planner_run = _record(result.planner_runs[0])
        executed_request = _record(result.request_records[0])
        if request.parsed_intent is not None:
            executed_request["parsed_intent"] = asdict(request.parsed_intent)
        receipt = {
            "schema_version": "runtime-repair-execution-v1",
            "execution_id": f"execution_{uuid4().hex}",
            "started_at": started_at,
            "completed_at": completed_at,
            "request_id": request.request_id,
            "request_content_hash": stable_content_hash(executed_request),
            "parent_plan_id": self.parent_plan_id,
            "parent_plan_content_hash": self.parent_plan.get("content_hash"),
            "planner_run": planner_run,
            "repair_request": executed_request,
            "planner_trace": _record(result.dashboard_records[0]),
            "child_plan_id": None,
            "child_plan_content_hash": None,
            "evaluation_id": None,
            "evaluation_content_hash": None,
            "child_plan": None,
            "plan_diff": None,
            "evaluation_certificate": None,
            "evaluation_executed": bool(result.evaluations),
            "evaluator": "PlanEvaluator.evaluate_final_plan",
        }
        if len(result.output_plans) == 0:
            if result.evaluations or result.diff_records:
                raise WorkspaceError("draft_pipeline_artifact_invalid", 409)
            state, reason, findings = _unevaluated_outcome(
                receipt["planner_trace"],
                self._base_bundle.route_matrix or {},
            )
            return CompiledDraftPreview(
                state=state,
                reason=reason,
                source_request_id=request.request_id,
                operations=normalized,
                source_bundle=self._base_bundle,
                planner_run=planner_run,
                runtime_execution=receipt,
                blocking_findings=findings,
            )
        if len(result.output_plans) != 1 or len(result.diff_records) != 1 or len(result.evaluations) != 1:
            raise WorkspaceError("draft_pipeline_artifact_invalid", 409)
        child = _record(result.output_plans[0])
        diff = _record(result.diff_records[0])
        certificate = _record(result.evaluations[0])
        self._verify_pipeline_artifacts(child, diff, certificate)
        if exact_candidate is not None and not self._coherent_product:
            self._cross_check_frozen_candidate(exact_candidate, child, diff, certificate)
        semantic_failures = self._semantic_failures(normalized, child)
        eligible = (
            not semantic_failures
            and certificate.get("comparison_eligibility") == "eligible"
            and certificate.get("evaluation_status") in {"PASSED", "PASSED_WITH_WARNINGS"}
        )
        receipt.update(
            {
                "child_plan_id": child["plan_id"],
                "child_plan_content_hash": child["content_hash"],
                "evaluation_id": certificate["certificate_id"],
                "evaluation_content_hash": certificate["content_hash"],
                "child_plan": child,
                "plan_diff": diff,
                "evaluation_certificate": certificate,
            }
        )
        parent_legs: tuple[dict[str, Any], ...] = ()
        child_legs: tuple[dict[str, Any], ...] = ()
        if self._coherent_product:
            matrix, day_config, _ = self._runtime_inputs()
            parent_legs = _route_legs(
                self._parent_artifact,
                matrix,
                day_config.start_anchor_by_day,
                day_config.end_anchor_by_day,
            )
            child_legs = _route_legs(
                _plan_artifact(child),
                matrix,
                day_config.start_anchor_by_day,
                day_config.end_anchor_by_day,
            )
            for row in (*parent_legs, *child_legs):
                row["evidence_scope"] = "runtime_optimizer_route_leg"
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None
            if eligible
            else (semantic_failures[0] if semantic_failures else "independent_evaluation_failed"),
            source_request_id=request.request_id,
            operations=normalized,
            source_bundle=exact_candidate.bundle if exact_candidate else self._base_bundle,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run,
            parent_route_legs=parent_legs,
            route_legs=child_legs,
            runtime_execution=receipt,
        )

    def _compile_combined(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> CompiledDraftPreview:
        """Compose in memory; only the complete child can become a preview.

        Existing typed compilers retain parameter/permission validation. Their
        intermediate certificates are diagnostic, never acceptance decisions.
        Attribute edits precede selection search so shortened visits, for
        example, are not evaluated only after a search using obsolete durations.
        Explicit ordering is applied after selection, then the final artifact is
        independently evaluated against the original accepted parent.
        """
        started_at = datetime.now(UTC).isoformat()
        handlers = {
            "set_stop_duration": "_compile_stop_duration",
            "set_stop_time_window": "_compile_stop_time_window",
            "set_stop_role": "_compile_stop_role",
            "set_stop_order": "_compile_stop_order",
        }
        groups = {kind: tuple(row for row in operations if row["type"] == kind) for kind in handlers}
        removed = {row["target"] for row in operations if row["type"] == "replace_nearby"}
        if any(row["target"] in removed for kind in handlers for row in groups[kind]):
            raise WorkspaceError("draft_conflicts_with_replacement", 409)
        # Validate every typed clause before dispatching the selection optimizer.
        for kind, rows in groups.items():
            if rows:
                # Already satisfied ordering rows must not block another typed
                # change. Each row has passed target/day/index validation, and
                # the complete child is still independently checked below.
                if kind == "set_stop_order" and all(
                    next(
                        day["stop_ids"]
                        for day in self.parent_plan["ordered_days"]
                        if day["day"] == row["parameters"]["day"]
                    ).index(row["target"])
                    == row["parameters"]["sequence_index"]
                    for row in rows
                ):
                    continue
                validated = getattr(self, handlers[kind])(rows)
                if kind == "set_stop_order":
                    by_day = {day["day"]: day["stop_ids"] for day in validated.child_plan["ordered_days"]}
                    if any(
                        by_day[row["parameters"]["day"]].index(row["target"]) != row["parameters"]["sequence_index"]
                        for row in rows
                    ):
                        raise WorkspaceError("draft_conflicting_stop_order", 409)
        matrix, day_config, _ = self._runtime_inputs()
        seed = {
            "parent_plan_id": self.parent_plan_id,
            "parent_content_hash": self._parent_artifact.content_hash,
            "route_matrix_id": matrix.matrix_id,
            "operations": [{key: row[key] for key in ("type", "target", "parameters")} for row in operations],
            "method": "atomic_combined_repair_v1",
        }
        request_id = f"combined_{stable_content_hash(seed)}"
        repair_request = {
            "schema_version": "atomic-combined-repair-request-v1",
            "request_id": request_id,
            "parent_plan_id": self.parent_plan_id,
            "request_family": "combined_typed_edit",
            "user_intent": "Review the complete staged draft as one independently evaluated child.",
            "operations": list(operations),
            "confirmed_constraints": {},
        }
        receipt = {
            "schema_version": "atomic-combined-repair-execution-v1",
            "execution_id": f"execution_{uuid4().hex}",
            "started_at": started_at,
            "request_id": request_id,
            "parent_plan_id": self.parent_plan_id,
            "parent_plan_content_hash": self._parent_artifact.content_hash,
            "operations": list(operations),
            "repair_request": repair_request,
            "request_content_hash": stable_content_hash(repair_request),
            "optimizer_execution": None,
            "planner_run": None,
            "child_plan": None,
            "plan_diff": None,
            "evaluation_certificate": None,
            "child_plan_id": None,
            "child_plan_content_hash": None,
            "evaluation_id": None,
            "evaluation_content_hash": None,
            "evaluation_executed": False,
            "evaluator": "PlanEvaluator.evaluate_final_plan",
        }
        current = self.parent_plan
        expected_attributes: dict[str, dict[str, Any]] = {}
        fields = {
            "set_stop_duration": ("duration_constraint", "visit_duration_minutes"),
            "set_stop_time_window": ("time_window_constraint",),
            "set_stop_role": ("itinerary_role", "itinerary_role_source"),
        }
        for kind, names in fields.items():
            if not groups[kind]:
                continue
            part = getattr(self.for_parent(current), handlers[kind])(groups[kind])
            current = part.child_plan
            for row in groups[kind]:
                stop = next(stop for stop in current["selected_stops"] if _stop_id(stop) == row["target"])
                expected_attributes.setdefault(row["target"], {}).update({name: stop[name] for name in names})
        selection = tuple(row for row in operations if row["type"] not in handlers)
        requirements = None
        if selection:
            selection_compiler = self.for_parent(current)
            selection_request, _ = selection_compiler._compile_request(selection)
            requirements = context_evaluation_requirements(selection_request)
            part = selection_compiler.compile(selection, accepted_plan_id=selection_compiler.parent_plan_id)
            receipt["optimizer_execution"] = part.runtime_execution
            if part.child_plan is None:
                receipt["completed_at"] = datetime.now(UTC).isoformat()
                return replace(
                    part,
                    source_request_id=request_id,
                    operations=operations,
                    source_bundle=self._base_bundle,
                    execution_mode="atomic_combined_independent_evaluation",
                    runtime_execution=receipt,
                )
            current = part.child_plan
        if groups["set_stop_order"]:
            # A selection solver may already produce the requested order. It is
            # not an error unless the original draft itself had no effect.
            for row in groups["set_stop_order"]:
                day = next(day for day in current["ordered_days"] if day["day"] == row["parameters"]["day"])
                if day["stop_ids"].index(row["target"]) != row["parameters"]["sequence_index"]:
                    current = self.for_parent(current)._compile_stop_order((row,)).child_plan
        run_id = f"combined_run_{stable_content_hash(seed)}"
        child = replace(
            _plan_artifact(current),
            plan_id=f"plan_{stable_content_hash({**seed, 'result_hash': current['content_hash']})}",
            parent_plan_id=self.parent_plan_id,
            source_run_id=run_id,
            planning_request_id=request_id,
            modeled_metrics={},
            context_exposure_components={},
            change_components={},
            certificate_id=None,
            created_at="",
        )
        planner_run = PlannerRun(
            run_id=run_id,
            parent_run_id=self._parent_artifact.source_run_id or None,
            planning_request_id=request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id="atomic_combined_repair_v1",
            method_requested="atomic_combined_repair_v1",
            method_executed="atomic_combined_repair_v1",
            execution_status="COMPLETED",
            result_plan_id=child.plan_id,
            solver_backend="deterministic_context_aware_heuristic" if selection else "typed_direct_edits",
        )
        config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=day_config.max_day_minutes,
            day_start_time=day_config.day_start_time,
            default_visit_minutes=day_config.default_visit_minutes,
            enforce_opening_windows=day_config.enforce_opening_windows,
            start_anchor_by_day=dict(day_config.start_anchor_by_day),
            end_anchor_by_day=dict(day_config.end_anchor_by_day),
            require_lodging_assignments=bool(child.lodging_assignments),
            include_schedule_metrics=True,
        )
        if requirements is not None:
            config = replace(
                config,
                required_stop_ids=requirements.required_stop_ids,
                excluded_stop_ids=requirements.excluded_stop_ids,
                closed_route_ids=requirements.closed_route_ids,
                unavailable_lodging_ids=requirements.unavailable_lodging_ids,
                max_day_minutes=min(config.max_day_minutes, requirements.max_day_minutes or config.max_day_minutes),
            )
        evaluator = PlanEvaluator(
            route_matrix=matrix, planner_runs={run_id: planner_run}, config=config, reference_plan=self._parent_artifact
        )
        certificate = evaluator.evaluate_final_plan(child, planner_run=planner_run).to_record()
        diff = (
            PlanDiffBuilder(default_visit_minutes=day_config.default_visit_minutes)
            .diff(self._parent_artifact, child)
            .to_record()
        )
        child_record = child.to_record()
        self._verify_pipeline_artifacts(child_record, diff, certificate)
        failures = list(self._semantic_failures(operations, child_record))
        stop_lookup = {_stop_id(stop): stop for stop in child.selected_stops}
        if any(
            stop_lookup.get(target, {}).get(name) != value
            for target, values in expected_attributes.items()
            for name, value in values.items()
        ):
            failures.append("requested_attribute_edit_not_satisfied")
        eligible = not failures and certificate["comparison_eligibility"] == "eligible"
        receipt.update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "evaluation_executed": True,
                "child_plan": child_record,
                "plan_diff": diff,
                "evaluation_certificate": certificate,
                "child_plan_id": child.plan_id,
                "child_plan_content_hash": child.content_hash,
                "evaluation_id": certificate["certificate_id"],
                "evaluation_content_hash": certificate["content_hash"],
                "planner_run": planner_run.to_record(),
            }
        )
        return CompiledDraftPreview(
            state="eligible" if eligible else "ineligible",
            reason=None if eligible else failures[0] if failures else "independent_evaluation_failed",
            source_request_id=request_id,
            operations=operations,
            source_bundle=self._base_bundle,
            child_plan=child_record,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run.to_record(),
            parent_route_legs=_route_legs(
                self._parent_artifact, matrix, day_config.start_anchor_by_day, day_config.end_anchor_by_day
            ),
            route_legs=_route_legs(child, matrix, day_config.start_anchor_by_day, day_config.end_anchor_by_day),
            execution_mode="atomic_combined_independent_evaluation",
            runtime_execution=receipt,
        )

    def _unsupported_product_operation(self, operation_type: str) -> bool:
        return self._coherent_product and operation_type in {
            "lock_stop",
            "mark_flexible",
            "move_day",
            "route_feedback",
            DraftOperationTypeV2.SET_STOP_DAY.value,
        }

    def _build_candidates(self, bundles_by_plan: dict[str, Any]) -> dict[str, ExecutableCandidate]:
        candidates: dict[str, ExecutableCandidate] = {}
        for plan_id, bundle in bundles_by_plan.items():
            source_parent = _plan_artifact(bundle.parent_plan or {})
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
                str(value) for value in (request.get("confirmed_constraints") or {}).get("target_stop_ids") or ()
            )
            if (
                not candidate_id
                or candidate_id != _stop_id(added[0])
                or replaced_stop_id not in source_parent.sequence
                or child.get("parent_plan_id") != source_parent.plan_id
                or diff.get("parent_plan_id") != source_parent.plan_id
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
                label=str(candidate_record.get("name") or candidate_record.get("attraction_name") or candidate_id),
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
            if self._unsupported_product_operation(operation_type):
                raise WorkspaceError("draft_optimizer_mapping_unsupported", 409)
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
            if operation_type == "replan_trip":
                from .trip_planning import validate_trip_parameters

                if target != "trip":
                    raise WorkspaceError("draft_target_not_in_parent", 409)
                try:
                    validate_trip_parameters(parameters)
                except (ValueError, TypeError, KeyError):
                    raise WorkspaceError("invalid_trip_planning_parameters", 422) from None
                continue
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
                if set(parameters) != {"day", "sequence_index"} or not _valid_day(parameters.get("day")):
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
                if self._candidates[_candidate_parameter(parameters)].replaced_stop_id != target:
                    raise WorkspaceError("draft_candidate_target_mismatch", 409)
            elif operation_type == "add_candidate":
                if set(parameters) != {"day"} or not _valid_day(parameters.get("day")):
                    raise WorkspaceError("draft_destination_day_invalid", 409)
                if target not in self._candidates:
                    raise WorkspaceError("draft_candidate_not_registered", 409)
                if target in self.parent_stop_ids:
                    raise WorkspaceError("draft_candidate_already_in_plan", 409)
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
        requested_candidates = {candidate for values in replacements.values() for candidate in values} | {
            row["target"] for row in operations if row["type"] == "add_candidate"
        }
        if len(requested_candidates) > 1:
            raise WorkspaceError("draft_conflicting_candidates", 409)
        if (
            preferences == {"reduce_driving"}
            and requested_candidates
            and not requested_candidates.issubset({"santa_barbara_museum_of_natural_history_sea_center"})
        ):
            raise WorkspaceError("draft_conflicting_preferences", 409)
        if (
            preferences == {"preserve_experience"}
            and requested_candidates
            and not requested_candidates.issubset({"bixby_creek_bridge_viewpoint"})
        ):
            raise WorkspaceError("draft_conflicting_preferences", 409)

    def _compile_request(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> tuple[RepairRequest, ExecutableCandidate | None]:
        exact_candidate = self._exact_replacement(operations)
        feedback_role = {
            "reduce_contextual_risk": "recommended",
            "reduce_driving": "low_driving",
        }.get(
            str(operations[0]["parameters"].get("preference") or "")
            if len(operations) == 1 and operations[0]["type"] == "route_feedback"
            else ""
        )
        if exact_candidate is None and feedback_role:
            matching = [
                candidate
                for candidate in self._candidates.values()
                if (candidate.bundle.metrics or {}).get("role") == feedback_role
            ]
            if len(matching) == 1:
                exact_candidate = matching[0]
        source = exact_candidate.bundle.request if exact_candidate else self._base_bundle.request
        constraints = dict(source.get("confirmed_constraints") or {})
        if self._coherent_product:
            # Scenario evidence defines the disruption and route requirements;
            # only the current typed draft defines edits and selection policies.
            constraints = {key: value for key, value in constraints.items() if key in {"require_road_validated_routes"}}
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
                affected_days.add(self._parent_day_by_stop[target])
                candidate_id = _candidate_parameter(row["parameters"])
                must_include.add(candidate_id)
                requested_candidates.add(candidate_id)
            elif kind == "add_candidate":
                must_include.add(target)
                requested_candidates.add(target)
                affected_days.add(int(row["parameters"]["day"]))
        if exact_candidate is not None and feedback_role:
            must_delete.add(exact_candidate.replaced_stop_id)
            must_include.add(exact_candidate.candidate_id)
            requested_candidates.add(exact_candidate.candidate_id)
            affected_days.add(self._parent_day_by_stop[exact_candidate.replaced_stop_id])
        if not requested_candidates and preferences == ["reduce_driving"]:
            requested_candidates.add("santa_barbara_museum_of_natural_history_sea_center")
        elif not requested_candidates and preferences == ["preserve_experience"]:
            requested_candidates.add("bixby_creek_bridge_viewpoint")
        # The product draft owns candidate generation. Research-only legacy
        # fixtures retain their declared optional-search universe.
        candidate_ids = requested_candidates or (set() if self._coherent_product else set(self._candidates))
        candidate_rows = tuple(dict(self._candidates[candidate_id].record) for candidate_id in sorted(candidate_ids))
        if self._coherent_product:
            for candidate_row in candidate_rows:
                candidate_id = _stop_id(candidate_row)
                matching = next(
                    (
                        row
                        for row in operations
                        if (row["type"] == "add_candidate" and row["target"] == candidate_id)
                        or (row["type"] == "replace_nearby" and _candidate_parameter(row["parameters"]) == candidate_id)
                    ),
                    None,
                )
                if matching:
                    candidate_row["day"] = (
                        matching["parameters"]["day"]
                        if matching["type"] == "add_candidate"
                        else self._parent_day_by_stop[matching["target"]]
                    )
            if affected_days:
                # Keeping a stop outside the edit scope is a frozen-boundary
                # condition, not permission to enumerate candidates on its day.
                frozen_keep = {stop for stop in must_keep if self._parent_day_by_stop[stop] not in affected_days}
                if frozen_keep:
                    constraints["preserved_stop_ids"] = sorted(frozen_keep)
                    must_keep.difference_update(frozen_keep)
                    must_include.difference_update(frozen_keep)
            # Other days are frozen by the same-day neighborhood. Marking their
            # stops as explicit targets would incorrectly expand that neighborhood.
            must_keep.update(
                stop_id
                for stop_id in self.parent_stop_ids - must_delete
                if self._parent_day_by_stop[stop_id] in affected_days
            )
            must_include.update(must_keep)
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

        if exact_candidate and not self._coherent_product:
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
            identity = {"parent": self.parent_plan_id, "operations": semantic_operations}
            if self._coherent_product:
                matrix = self._base_bundle.route_matrix or {}
                identity.update(
                    {
                        "parent_content_hash": self.parent_plan["content_hash"],
                        "route_matrix_id": matrix.get("matrix_id"),
                        "route_evidence_hash": matrix.get("source_content_sha256"),
                    }
                )
            request_id = f"w3_preview_{stable_content_hash(identity)}"
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
                baseline_route=tuple(
                    dict(row)
                    for row in (
                        self.parent_plan["selected_stops"]
                        if self.parent_plan_id != next(iter(self._bundles_by_plan.values())).parent_plan["plan_id"]
                        else source.get("baseline_route") or ()
                    )
                ),
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
        eligible = certificate.get("comparison_eligibility") == "eligible" and certificate.get("evaluation_status") in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        }
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
        eligible = certificate.get("comparison_eligibility") == "eligible" and certificate.get("evaluation_status") in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        }
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
        eligible = certificate.get("comparison_eligibility") == "eligible" and certificate.get("evaluation_status") in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        }
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
        eligible = certificate.get("comparison_eligibility") == "eligible" and certificate.get("evaluation_status") in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        }
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
        except TimeoutError as exc:
            raise WorkspaceError("draft_pipeline_execution_timed_out", 409) from exc
        except Exception as exc:
            raise WorkspaceError("draft_pipeline_execution_failed", 409) from exc

    def _runtime_inputs(self) -> tuple[RouteMatrix, DayRouteSolverConfig, Any]:
        with self._runtime_cache_lock:
            if self._runtime_cache is None:
                route_matrix = _route_matrix(self._base_bundle.route_matrix or {})
                run_manifest = self._base_bundle.run_manifest or {}
                scenario_days = (run_manifest.get("scenario") or {}).get("days") or ()
                declared_anchors = {
                    int(row["day"]): (
                        str(row["start_anchor_id"]),
                        str(row["end_anchor_id"]),
                    )
                    for row in scenario_days
                    if isinstance(row, dict)
                    and isinstance(row.get("day"), int)
                    and not isinstance(row.get("day"), bool)
                    and isinstance(row.get("start_anchor_id"), str)
                    and row.get("start_anchor_id")
                    and isinstance(row.get("end_anchor_id"), str)
                    and row.get("end_anchor_id")
                }
                anchors = (
                    declared_anchors
                    if set(declared_anchors)
                    == set(range(1, max(stop["day"] for stop in self.parent_plan["selected_stops"]) + 1))
                    else _DAY_ANCHORS
                )
                day_config = DayRouteSolverConfig(
                    max_day_minutes=720.0,
                    default_visit_minutes=45.0,
                    strict_route_matrix=True,
                    start_anchor_by_day={day: pair[0] for day, pair in anchors.items()},
                    end_anchor_by_day={day: pair[1] for day, pair in anchors.items()},
                )
                trip_config = load_trip_config(self.repository_root / "configs" / "default_trip_config.yaml")
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
        _verify_diff(
            diff,
            parent=self._parent_artifact,
            child=_plan_artifact(child),
            certificate=certificate,
            default_visit_minutes=self._runtime_inputs()[1].default_visit_minutes,
        )
        _verify_certificate_id(certificate)
        route_validation = certificate.get("route_validation") or {}
        matrix = self._base_bundle.route_matrix or {}
        if (
            route_validation.get("publication_ready") is not True
            or route_validation.get("fallback_leg_count") != 0
            or route_validation.get("missing_leg_count") != 0
            or route_validation.get("road_validated_leg_count") != route_validation.get("required_leg_count")
            or route_validation.get("matrix_id") != matrix.get("matrix_id")
            or route_validation.get("context_snapshot_id") != matrix.get("context_snapshot_id")
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

    def _semantic_failures(
        self,
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
        if self._coherent_product:
            expected = {row["day"]: list(row["stop_ids"]) for row in self.parent_plan["ordered_days"]}
            for row in operations:
                if row["type"] == "replace_nearby":
                    day = self._parent_day_by_stop[row["target"]]
                    expected[day] = [
                        _candidate_parameter(row["parameters"]) if stop == row["target"] else stop
                        for stop in expected[day]
                    ]
                elif row["type"] == "add_candidate":
                    expected[row["parameters"]["day"]].append(row["target"])
            for row in operations:
                if row["type"] == "set_stop_order":
                    stops = expected[row["parameters"]["day"]]
                    stops.remove(row["target"])
                    stops.insert(row["parameters"]["sequence_index"], row["target"])
            actual = {row["day"]: list(row["stop_ids"]) for row in child["ordered_days"]}
            if actual != expected:
                failures.append("unrequested_itinerary_change")
            if child.get("lodging_assignments") != self.parent_plan.get("lodging_assignments"):
                failures.append("unrequested_lodging_change")
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
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 21


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
        lodging_assignments={str(key): str(value) for key, value in (record.get("lodging_assignments") or {}).items()},
        ordered_days=tuple(dict(row) for row in record.get("ordered_days") or ()),
        route_ids_by_day={int(key): str(value) for key, value in (record.get("route_ids_by_day") or {}).items()},
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


def _unevaluated_outcome(
    trace: dict[str, Any],
    matrix: dict[str, Any],
) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    """Project the existing planner failure trace without inventing evaluation."""
    failures = tuple(str(value) for value in trace.get("failure_reasons") or ())
    missing = []
    for failure in failures:
        match = re.fullmatch(r"RouteMatrixCellMissing:missing route matrix cell: (.+) -> (.+)", failure)
        if match:
            try:
                origin, destination = (literal_eval(value) for value in match.groups())
            except (ValueError, SyntaxError):
                continue
            if not isinstance(origin, str) or not isinstance(destination, str):
                continue
            missing.append(
                {
                    "code": "missing_route_matrix_cell",
                    "origin_id": origin,
                    "destination_id": destination,
                    "matrix_id": matrix.get("matrix_id"),
                    "context_snapshot_id": matrix.get("context_snapshot_id"),
                    "corrective_action": "Provide road-validated evidence for this directed leg, then preview again.",
                }
            )
    if missing:
        return "blocked", "blocked_missing_route_evidence", tuple(missing)
    if any(value.startswith(("RouteMatrixMissing:", "RouteMatrixNotPublicationEligible:")) for value in failures):
        state, reason, action = (
            "blocked",
            "blocked_invalid_route_evidence",
            "Provide a complete road-validated route matrix, then preview again.",
        )
    elif any(value.startswith("TimeoutError:") for value in failures):
        state, reason, action = (
            "failed",
            "repair_execution_timed_out",
            "Retry the preview after resolving the local execution time limit.",
        )
    elif any(re.match(r"[A-Za-z_][A-Za-z_0-9]*(?:Error|Exception):", value) for value in failures):
        state, reason, action = (
            "failed",
            "repair_execution_failed",
            "Inspect the execution trace and resolve the local failure before retrying.",
        )
    else:
        state = "not_evaluated"
        reason = (
            "no_feasible_local_candidate"
            if any(value.startswith("no_feasible_local_candidate:") for value in failures)
            else "repair_not_evaluated"
        )
        action = "Adjust the draft or search constraints; no child reached independent evaluation."
    return state, reason, ({"code": reason, "corrective_action": action},)


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


def _verify_diff(
    diff: dict[str, Any],
    *,
    parent: PlanArtifactV2 | None = None,
    child: PlanArtifactV2 | None = None,
    certificate: dict[str, Any] | None = None,
    default_visit_minutes: float | None = None,
) -> None:
    if diff.get("schema_version") in {"plan-diff-v5", "plan-diff-v7"}:
        # Cost evidence is not approval. Only the real pipeline supplies this
        # parent/child context; acceptance still reruns independent evaluation.
        if (
            not isinstance(parent, PlanArtifactV2)
            or not isinstance(child, PlanArtifactV2)
            or not isinstance(certificate, dict)
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        certificate_payload = dict(certificate)
        certificate_hash = certificate_payload.pop("content_hash", None)
        if (
            child.parent_plan_id != parent.plan_id
            or certificate.get("plan_id") != child.plan_id
            or certificate.get("plan_content_hash") != child.content_hash
            or certificate.get("source_run_id") != child.source_run_id
            or certificate.get("evaluator_version") != PlanEvaluator.evaluator_version
            or stable_content_hash(certificate_payload) != certificate_hash
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        _verify_certificate_id(certificate)
        try:
            expected = PlanDiffBuilder(default_visit_minutes=default_visit_minutes).diff(parent, child).to_record()
        except (ValueError, TypeError) as error:
            raise WorkspaceError("draft_pipeline_diff_invalid", 409) from error
        if diff != expected:
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        return
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
    if schema_version == "plan-diff-v6":
        names = ("duration_changes", "time_window_changes", "role_changes")
        present = [name for name in names if name in diff]
        if (
            len(present) < 2
            or any(not isinstance(diff[name], list) or not diff[name] for name in present)
            or "attribute_ownership_evidence" in diff
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        groups.extend((name, name) for name in present)
    elif schema_version == "plan-diff-v2":
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
