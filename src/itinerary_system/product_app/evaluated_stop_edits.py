"""Independent evaluated artifacts for losslessly modeled direct stop edits."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from ..evaluation import PlanEvaluationCertificate, PlanEvaluator
from ..evaluation.plan_evaluator import PlanEvaluatorConfig
from ..plans import (
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    active_owned_constraints,
)
from ..plans.diff import PlanDiffBuilder
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix
from .workspace import WorkspaceError

EVALUATED_STOP_ROLES = (
    "attraction",
    "activity",
    "meal",
    "rest_stop",
    "scenic_stop",
)
ALL_STOP_ROLES = frozenset(
    {
        *EVALUATED_STOP_ROLES,
        "lodging",
        "transport_hub",
        "route_waypoint",
        "origin",
        "destination",
    }
)
ROLE_SOURCE = "user_declared_itinerary_role"
_ROLE_RELATIONS = frozenset(
    {"role", "itinerary_role", "role_equals", "set_role", "attribute:role"}
)


def build_role_constraint_evidence(
    parent: PlanArtifactV2 | dict[str, Any],
) -> dict[str, Any]:
    """Return sanitized accepted-parent evidence for role-scoped ownership only."""

    if isinstance(parent, PlanArtifactV2):
        plan_id = parent.plan_id
        plan_content_hash = parent.content_hash
        raw_constraints = parent.owned_constraints
        selected_stops = parent.selected_stops
    else:
        plan_id = str(parent.get("plan_id") or "")
        plan_content_hash = str(parent.get("content_hash") or "")
        raw_constraints = tuple(parent.get("owned_constraints") or ())
        selected_stops = tuple(parent.get("selected_stops") or ())
    stop_ids = {
        str(stop.get("stop_id") or stop.get("poi_id") or "")
        for stop in selected_stops
        if isinstance(stop, dict)
    }
    if not plan_id or not plan_content_hash or "" in stop_ids:
        raise WorkspaceError("draft_parent_role_constraint_invalid", 409)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    required_role_by_target: dict[str, str] = {}
    for raw in raw_constraints:
        if not isinstance(raw, dict):
            continue
        scope = str(raw.get("scope") or "").strip().lower()
        relation = str(raw.get("relation") or "").strip().lower()
        if scope not in {"stop", "poi"} or relation not in _ROLE_RELATIONS:
            continue
        try:
            constraint = OwnedConstraint.from_record(raw)
        except Exception as exc:
            raise WorkspaceError(
                "draft_parent_role_constraint_invalid", 409
            ) from exc
        if not constraint.is_active:
            continue
        if (
            not constraint.constraint_id
            or constraint.constraint_id in seen_ids
            or constraint.target_id not in stop_ids
            or not isinstance(constraint.value, str)
            or constraint.value not in ALL_STOP_ROLES
        ):
            raise WorkspaceError("draft_parent_role_constraint_invalid", 409)
        prior_role = required_role_by_target.get(constraint.target_id)
        if prior_role is not None and prior_role != constraint.value:
            raise WorkspaceError("draft_parent_role_constraint_invalid", 409)
        seen_ids.add(constraint.constraint_id)
        required_role_by_target[constraint.target_id] = constraint.value
        permission_semantics = (
            "explicit_permission_required_for_mismatch"
            if constraint.strength
            in {
                ConstraintStrength.LOCKED,
                ConstraintStrength.BOOKED,
                ConstraintStrength.HARD,
            }
            or not constraint.can_relax()
            else "weighted_mismatch_allowed"
        )
        rows.append(
            {
                "constraint_id": constraint.constraint_id,
                "target_stop_id": constraint.target_id,
                "required_role": constraint.value,
                "strength": constraint.strength.value,
                "scope": constraint.scope.value,
                "relation": constraint.relation.strip().lower(),
                "relaxation_policy": constraint.relaxation_policy.value,
                "permission_semantics": permission_semantics,
            }
        )
    payload = {
        "schema_version": "evaluated-role-constraint-evidence-v1",
        "parent_plan_id": plan_id,
        "parent_plan_content_hash": plan_content_hash,
        "constraints": sorted(
            rows,
            key=lambda row: (row["target_stop_id"], row["constraint_id"]),
        ),
    }
    return {**payload, "content_hash": stable_content_hash(payload)}


@dataclass(frozen=True)
class EvaluatedStopEditArtifacts:
    """One immutable child plus independently recomputed evidence."""

    source_request_id: str
    child_plan: PlanArtifactV2
    diff: Any
    certificate: PlanEvaluationCertificate
    planner_run: PlannerRun
    parent_route_legs: tuple[dict[str, Any], ...]
    route_legs: tuple[dict[str, Any], ...]
    schedule_impact: dict[str, Any] | None = None


class EvaluatedStopOrderCompiler:
    """Compile same-day order edits without invoking an optimizer.

    The edit itself is deterministic. Eligibility is still owned by the
    independent evaluator over the exact resulting sequence and frozen route
    matrix.
    """

    METHOD_ID = "typed_same_day_stop_order_v1"

    def __init__(
        self,
        *,
        parent: PlanArtifactV2,
        route_matrix: RouteMatrix,
        start_anchor_by_day: dict[int, str],
        end_anchor_by_day: dict[int, str],
        max_day_minutes: float,
        default_visit_minutes: float,
    ) -> None:
        self._parent = parent
        self._route_matrix = route_matrix
        self._start_anchors = dict(start_anchor_by_day)
        self._end_anchors = dict(end_anchor_by_day)
        self._config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=max_day_minutes,
            default_visit_minutes=default_visit_minutes,
            start_anchor_by_day=dict(start_anchor_by_day),
            end_anchor_by_day=dict(end_anchor_by_day),
            require_lodging_assignments=bool(parent.lodging_assignments),
        )

    def compile(self, operations: tuple[dict[str, Any], ...]) -> EvaluatedStopEditArtifacts:
        if not operations:
            raise WorkspaceError("draft_is_empty")
        ordered_by_day = _ordered_stop_ids(self._parent)
        parent_ordered_by_day = {day: list(stop_ids) for day, stop_ids in ordered_by_day.items()}
        day_by_stop = {
            stop_id: day
            for day, stop_ids in parent_ordered_by_day.items()
            for stop_id in stop_ids
        }
        seen_targets: set[str] = set()
        for operation in operations:
            target = operation["target"]
            parameters = operation["parameters"]
            if target in seen_targets:
                raise WorkspaceError("draft_conflicting_stop_order", 409)
            seen_targets.add(target)
            day = parameters.get("day")
            index = parameters.get("sequence_index")
            if day_by_stop.get(target) != day:
                raise WorkspaceError("draft_order_day_mismatch", 409)
            day_stops = ordered_by_day[day]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(day_stops)
            ):
                raise WorkspaceError("draft_order_index_invalid", 409)
            current_index = day_stops.index(target)
            if current_index == index:
                raise WorkspaceError("draft_no_effect", 409)
            day_stops.pop(current_index)
            day_stops.insert(index, target)

        if ordered_by_day == parent_ordered_by_day:
            raise WorkspaceError("draft_no_effect", 409)

        semantic_operations = tuple(
            {
                "type": operation["type"],
                "target": operation["target"],
                "parameters": dict(operation["parameters"]),
            }
            for operation in operations
        )
        request_seed = {
            "parent_plan_id": self._parent.plan_id,
            "parent_content_hash": self._parent.content_hash,
            "route_matrix_id": self._route_matrix.matrix_id,
            "operations": semantic_operations,
            "method": self.METHOD_ID,
        }
        request_hash = stable_content_hash(request_seed)
        source_request_id = f"w4r_order_{request_hash}"
        source_run_id = f"w4r_order_run_{request_hash}"
        child_plan_id = f"plan_{stable_content_hash({**request_seed, 'source_run_id': source_run_id})}"
        route_ids = dict(self._parent.route_ids_by_day)
        affected_days = {
            day
            for day in ordered_by_day
            if ordered_by_day[day] != parent_ordered_by_day[day]
        }
        for day in affected_days:
            route_ids[day] = self._route_id(day, ordered_by_day[day])

        selected_stops = _reordered_stop_records(self._parent, ordered_by_day)
        ordered_days = tuple(
            {"day": day, "stop_ids": tuple(stop_ids)}
            for day, stop_ids in sorted(ordered_by_day.items())
        )
        sequence = tuple(
            stop_id
            for _, stop_ids in sorted(ordered_by_day.items())
            for stop_id in stop_ids
        )
        child = replace(
            self._parent,
            plan_id=child_plan_id,
            parent_plan_id=self._parent.plan_id,
            source_run_id=source_run_id,
            planning_request_id=source_request_id,
            selected_stops=selected_stops,
            day_assignments={
                stop_id: day
                for day, stop_ids in ordered_by_day.items()
                for stop_id in stop_ids
            },
            sequence=sequence,
            ordered_days=ordered_days,
            route_ids_by_day=route_ids,
            modeled_metrics={"selected_attractions": float(len(selected_stops))},
            context_exposure_components={},
            change_components={},
            certificate_id=None,
            # Draft operations currently carry no trusted event timestamp. Keep
            # the child timestamp explicitly unavailable instead of copying the
            # parent creation time or hashing wall-clock state into the artifact.
            created_at="",
        )
        planner_run = PlannerRun(
            run_id=source_run_id,
            parent_run_id=self._parent.source_run_id or None,
            planning_request_id=source_request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id=self.METHOD_ID,
            method_requested=self.METHOD_ID,
            method_executed=self.METHOD_ID,
            execution_status="COMPLETED",
            solver_certification="FEASIBILITY_CERTIFIED",
            solver_backend="independent_plan_evaluator",
            result_plan_id=child.plan_id,
        )
        diff = PlanDiffBuilder().diff(self._parent, child)
        if not diff.reorder_changes or not diff.road_changes:
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        evaluator = PlanEvaluator(
            route_matrix=self._route_matrix,
            planner_runs={source_run_id: planner_run},
            config=self._config,
            reference_plan=self._parent,
        )
        certificate = evaluator.evaluate_final_plan(child, planner_run=planner_run)
        return EvaluatedStopEditArtifacts(
            source_request_id=source_request_id,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run,
            parent_route_legs=self.route_legs(self._parent),
            route_legs=self.route_legs(child),
        )

    def route_legs(self, plan: PlanArtifactV2) -> tuple[dict[str, Any], ...]:
        """Return every evaluator-required route leg in display order."""

        return _route_legs(
            plan,
            self._route_matrix,
            self._start_anchors,
            self._end_anchors,
        )

    def _route_id(self, day: int, stop_ids: list[str]) -> str:
        sequence = [self._start_anchors.get(day), *stop_ids, self._end_anchors.get(day)]
        sequence = [value for value in sequence if value]
        query_hashes: list[str] = []
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False):
            cell = _required_cell(self._route_matrix, origin_id, destination_id)
            query_hashes.append(cell.query_hash)
        return f"route_w4r_{stable_content_hash({'day': day, 'stops': stop_ids, 'matrix_id': self._route_matrix.matrix_id, 'query_hashes': query_hashes})}"


class EvaluatedStopDurationCompiler:
    """Compile canonical exact visit durations and independently evaluate them."""

    METHOD_ID = "typed_exact_stop_duration_v1"

    def __init__(
        self,
        *,
        parent: PlanArtifactV2,
        route_matrix: RouteMatrix,
        start_anchor_by_day: dict[int, str],
        end_anchor_by_day: dict[int, str],
        max_day_minutes: float,
        default_visit_minutes: float,
    ) -> None:
        if (
            isinstance(default_visit_minutes, bool)
            or not isinstance(default_visit_minutes, (int, float))
            or not isfinite(default_visit_minutes)
            or not 15 <= float(default_visit_minutes) <= 480
        ):
            raise WorkspaceError("draft_evaluator_duration_config_invalid", 409)
        self._parent = parent
        self._route_matrix = route_matrix
        self._start_anchors = dict(start_anchor_by_day)
        self._end_anchors = dict(end_anchor_by_day)
        self._default_visit_minutes = float(default_visit_minutes)
        self._config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=max_day_minutes,
            default_visit_minutes=default_visit_minutes,
            start_anchor_by_day=dict(start_anchor_by_day),
            end_anchor_by_day=dict(end_anchor_by_day),
            require_lodging_assignments=bool(parent.lodging_assignments),
            include_schedule_metrics=True,
        )

    def compile(self, operations: tuple[dict[str, Any], ...]) -> EvaluatedStopEditArtifacts:
        if not operations:
            raise WorkspaceError("draft_is_empty")
        stop_lookup = {
            str(stop.get("stop_id") or stop.get("poi_id") or ""): dict(stop)
            for stop in self._parent.selected_stops
        }
        if not stop_lookup or any(not stop_id for stop_id in stop_lookup):
            raise WorkspaceError("draft_parent_duration_invalid", 409)
        for stop in stop_lookup.values():
            _validate_parent_duration_record(stop)

        changed_stops = {stop_id: dict(stop) for stop_id, stop in stop_lookup.items()}
        seen_targets: set[str] = set()
        for operation in operations:
            if operation.get("type") != "set_stop_duration":
                raise WorkspaceError("draft_evaluated_operation_combination_unsupported", 409)
            target = str(operation.get("target") or "")
            if target not in stop_lookup:
                raise WorkspaceError("draft_target_not_in_parent", 409)
            if target in seen_targets:
                raise WorkspaceError("draft_conflicting_attribute_edits", 409)
            seen_targets.add(target)
            duration = _canonical_exact_duration(operation.get("parameters"))
            minutes = int(duration["preferred_minutes"])
            if _is_explicit_exact_no_effect(stop_lookup[target], duration):
                raise WorkspaceError("draft_no_effect", 409)
            for legacy_key in ("duration_minutes", "service_minutes", "dwell_minutes"):
                changed_stops[target].pop(legacy_key, None)
            changed_stops[target]["duration_constraint"] = duration
            changed_stops[target]["visit_duration_minutes"] = minutes

        semantic_operations = tuple(
            {
                "type": operation["type"],
                "target": operation["target"],
                "parameters": dict(operation["parameters"]),
            }
            for operation in operations
        )
        request_seed = {
            "parent_plan_id": self._parent.plan_id,
            "parent_content_hash": self._parent.content_hash,
            "route_matrix_id": self._route_matrix.matrix_id,
            "operations": semantic_operations,
            "method": self.METHOD_ID,
        }
        request_hash = stable_content_hash(request_seed)
        source_request_id = f"w4r_duration_{request_hash}"
        source_run_id = f"w4r_duration_run_{request_hash}"
        child_plan_id = f"plan_{stable_content_hash({**request_seed, 'source_run_id': source_run_id})}"
        selected_stops = tuple(
            changed_stops[str(stop.get("stop_id") or stop.get("poi_id") or "")]
            for stop in self._parent.selected_stops
        )
        child = replace(
            self._parent,
            plan_id=child_plan_id,
            parent_plan_id=self._parent.plan_id,
            source_run_id=source_run_id,
            planning_request_id=source_request_id,
            selected_stops=selected_stops,
            modeled_metrics={"selected_attractions": float(len(selected_stops))},
            context_exposure_components={},
            change_components={},
            certificate_id=None,
            created_at="",
        )
        planner_run = PlannerRun(
            run_id=source_run_id,
            parent_run_id=self._parent.source_run_id or None,
            planning_request_id=source_request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id=self.METHOD_ID,
            method_requested=self.METHOD_ID,
            method_executed=self.METHOD_ID,
            execution_status="COMPLETED",
            solver_certification="FEASIBILITY_CERTIFIED",
            solver_backend="independent_plan_evaluator",
            result_plan_id=child.plan_id,
        )
        diff = PlanDiffBuilder(default_visit_minutes=self._default_visit_minutes).diff(
            self._parent,
            child,
        )
        if (
            diff.schema_version != "plan-diff-v2"
            or len(diff.duration_changes) != len(operations)
            or diff.added_stops
            or diff.deleted_stops
            or diff.day_moves
            or diff.time_shifts
            or diff.reorder_changes
            or diff.lodging_changes
            or diff.road_changes
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        evaluator = PlanEvaluator(
            route_matrix=self._route_matrix,
            planner_runs={source_run_id: planner_run},
            config=self._config,
            reference_plan=self._parent,
        )
        certificate = evaluator.evaluate_final_plan(child, planner_run=planner_run)
        parent_schedule = evaluator.evaluate_schedule_accounting(self._parent)
        child_schedule = evaluator.evaluate_schedule_accounting(child)
        schedule_payload = {
            "schema_version": "evaluated-duration-schedule-impact-v1",
            "evaluator_version": evaluator.evaluator_version,
            "route_matrix_id": self._route_matrix.matrix_id,
            "configured_default_visit_minutes": self._default_visit_minutes,
            "duration_cost_policy": {
                "base_change_cost": 0.25,
                "formula": (
                    "base_times_strength_weight_times_max_1_abs_delta_minutes_over_60"
                ),
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
            },
            "affected_days": sorted(
                {
                    change.day
                    for change in diff.duration_changes
                    if change.day is not None
                }
            ),
            "target_stop_ids": sorted(change.stop_id for change in diff.duration_changes),
            "parent": parent_schedule,
            "child": child_schedule,
        }
        schedule_impact = {
            **schedule_payload,
            "content_hash": stable_content_hash(schedule_payload),
        }
        return EvaluatedStopEditArtifacts(
            source_request_id=source_request_id,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run,
            parent_route_legs=_route_legs(
                self._parent,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            route_legs=_route_legs(
                child,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            schedule_impact=schedule_impact,
        )


class EvaluatedStopRoleCompiler:
    """Compile visit-bearing itinerary labels and independently re-evaluate."""

    METHOD_ID = "typed_stop_role_v1"

    def __init__(
        self,
        *,
        parent: PlanArtifactV2,
        route_matrix: RouteMatrix,
        start_anchor_by_day: dict[int, str],
        end_anchor_by_day: dict[int, str],
        max_day_minutes: float,
        default_visit_minutes: float,
        day_start_time: str | int | float,
    ) -> None:
        if (
            isinstance(default_visit_minutes, bool)
            or not isinstance(default_visit_minutes, (int, float))
            or not isfinite(default_visit_minutes)
            or not 15 <= float(default_visit_minutes) <= 480
        ):
            raise WorkspaceError("draft_evaluator_role_config_invalid", 409)
        self._parent = parent
        self._route_matrix = route_matrix
        self._start_anchors = dict(start_anchor_by_day)
        self._end_anchors = dict(end_anchor_by_day)
        self._default_visit_minutes = float(default_visit_minutes)
        self._day_start_minute = _configured_clock_minute(day_start_time)
        self._config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=max_day_minutes,
            day_start_time=day_start_time,
            default_visit_minutes=default_visit_minutes,
            start_anchor_by_day=dict(start_anchor_by_day),
            end_anchor_by_day=dict(end_anchor_by_day),
            require_lodging_assignments=bool(parent.lodging_assignments),
            include_schedule_metrics=True,
        )

    def compile(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> EvaluatedStopEditArtifacts:
        if not operations:
            raise WorkspaceError("draft_is_empty")
        stop_lookup = {
            str(stop.get("stop_id") or stop.get("poi_id") or ""): dict(stop)
            for stop in self._parent.selected_stops
        }
        if not stop_lookup or any(not stop_id for stop_id in stop_lookup):
            raise WorkspaceError("draft_parent_role_invalid", 409)
        for stop in stop_lookup.values():
            _validate_parent_role_record(stop)
            _validate_parent_duration_record(stop)
            _validate_parent_time_window_record(stop)
        role_constraint_evidence = build_role_constraint_evidence(self._parent)
        role_constraints = _role_constraints_by_target(self._parent)

        changed_stops = {stop_id: dict(stop) for stop_id, stop in stop_lookup.items()}
        seen_targets: set[str] = set()
        for operation in operations:
            if operation.get("type") != "set_stop_role":
                raise WorkspaceError(
                    "draft_evaluated_operation_combination_unsupported",
                    409,
                )
            target = str(operation.get("target") or "")
            if target not in stop_lookup:
                raise WorkspaceError("draft_target_not_in_parent", 409)
            if target in seen_targets:
                raise WorkspaceError("draft_conflicting_attribute_edits", 409)
            seen_targets.add(target)
            role = _canonical_evaluated_role(operation.get("parameters"))
            if (
                stop_lookup[target].get("itinerary_role") == role
                and stop_lookup[target].get("itinerary_role_source") == ROLE_SOURCE
            ):
                raise WorkspaceError("draft_no_effect", 409)
            _enforce_role_constraints(target, role, role_constraints.get(target, ()))
            changed_stops[target]["itinerary_role"] = role
            changed_stops[target]["itinerary_role_source"] = ROLE_SOURCE

        semantic_operations = tuple(
            {
                "type": operation["type"],
                "target": operation["target"],
                "parameters": dict(operation["parameters"]),
            }
            for operation in operations
        )
        request_seed = {
            "parent_plan_id": self._parent.plan_id,
            "parent_content_hash": self._parent.content_hash,
            "route_matrix_id": self._route_matrix.matrix_id,
            "operations": semantic_operations,
            "method": self.METHOD_ID,
        }
        request_hash = stable_content_hash(request_seed)
        source_request_id = f"w4r_role_{request_hash}"
        source_run_id = f"w4r_role_run_{request_hash}"
        child_plan_id = (
            f"plan_{stable_content_hash({**request_seed, 'source_run_id': source_run_id})}"
        )
        selected_stops = tuple(
            changed_stops[str(stop.get("stop_id") or stop.get("poi_id") or "")]
            for stop in self._parent.selected_stops
        )
        child = replace(
            self._parent,
            plan_id=child_plan_id,
            parent_plan_id=self._parent.plan_id,
            source_run_id=source_run_id,
            planning_request_id=source_request_id,
            selected_stops=selected_stops,
            modeled_metrics={},
            context_exposure_components={},
            change_components={},
            certificate_id=None,
            created_at="",
        )
        planner_run = PlannerRun(
            run_id=source_run_id,
            parent_run_id=self._parent.source_run_id or None,
            planning_request_id=source_request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id=self.METHOD_ID,
            method_requested=self.METHOD_ID,
            method_executed=self.METHOD_ID,
            execution_status="COMPLETED",
            solver_certification="FEASIBILITY_CERTIFIED",
            solver_backend="independent_plan_evaluator",
            result_plan_id=child.plan_id,
        )
        diff = PlanDiffBuilder(default_visit_minutes=self._default_visit_minutes).diff(
            self._parent,
            child,
        )
        if (
            diff.schema_version != "plan-diff-v4"
            or len(diff.role_changes) != len(operations)
            or diff.added_stops
            or diff.deleted_stops
            or diff.day_moves
            or diff.time_shifts
            or diff.duration_changes
            or diff.time_window_changes
            or diff.reorder_changes
            or diff.lodging_changes
            or diff.road_changes
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        evaluator = PlanEvaluator(
            route_matrix=self._route_matrix,
            planner_runs={source_run_id: planner_run},
            config=self._config,
            reference_plan=self._parent,
        )
        certificate = evaluator.evaluate_final_plan(child, planner_run=planner_run)
        parent_schedule = evaluator.evaluate_time_window_schedule_accounting(
            self._parent
        )
        child_schedule = evaluator.evaluate_time_window_schedule_accounting(child)
        parent_route_legs = _route_legs(
            self._parent,
            self._route_matrix,
            self._start_anchors,
            self._end_anchors,
        )
        child_route_legs = _route_legs(
            child,
            self._route_matrix,
            self._start_anchors,
            self._end_anchors,
        )
        schedule_invariant = _accounting_without_identity(parent_schedule) == (
            _accounting_without_identity(child_schedule)
        )
        invariance = {
            "sequence_unchanged": child.sequence == self._parent.sequence,
            "ordered_days_unchanged": child.ordered_days == self._parent.ordered_days,
            "route_ids_by_day_unchanged": (
                child.route_ids_by_day == self._parent.route_ids_by_day
            ),
            "route_legs_unchanged": child_route_legs == parent_route_legs,
            "schedule_accounting_unchanged": schedule_invariant,
        }
        if not all(invariance.values()):
            raise WorkspaceError("draft_role_invariance_failed", 409)
        impact_payload = {
            "schema_version": "evaluated-role-impact-v1",
            "evaluator_version": evaluator.evaluator_version,
            "route_matrix_id": self._route_matrix.matrix_id,
            "configured_default_visit_minutes": self._default_visit_minutes,
            "configured_day_start_minute": self._day_start_minute,
            "supported_evaluated_roles": list(EVALUATED_STOP_ROLES),
            "semantic_scope": {
                "itinerary_role_semantics": "trip_specific_user_declared_use",
                "place_identity_unchanged": True,
                "place_categories_unchanged": True,
                "route_schedule_effect": "none_for_supported_visit_roles",
                "semantic_fit_claim": "unavailable",
                "recommendation_claim": "unavailable",
            },
            "role_constraint_evidence": role_constraint_evidence,
            "route_leg_evidence": _time_window_route_evidence(
                self._parent,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            "role_cost_policy": {
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
            },
            "target_stop_ids": sorted(change.stop_id for change in diff.role_changes),
            "affected_days": sorted(
                {change.day for change in diff.role_changes if change.day is not None}
            ),
            "parent_role_states": _role_states(self._parent),
            "child_role_states": _role_states(child),
            "parent_schedule": parent_schedule,
            "child_schedule": child_schedule,
            "invariance": invariance,
        }
        return EvaluatedStopEditArtifacts(
            source_request_id=source_request_id,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run,
            parent_route_legs=parent_route_legs,
            route_legs=child_route_legs,
            schedule_impact={
                **impact_payload,
                "content_hash": stable_content_hash(impact_payload),
            },
        )


class EvaluatedStopTimeWindowCompiler:
    """Compile trip-specific required windows and independently evaluate them."""

    METHOD_ID = "typed_stop_time_window_v1"

    def __init__(
        self,
        *,
        parent: PlanArtifactV2,
        route_matrix: RouteMatrix,
        start_anchor_by_day: dict[int, str],
        end_anchor_by_day: dict[int, str],
        max_day_minutes: float,
        default_visit_minutes: float,
        day_start_time: str | int | float,
    ) -> None:
        if (
            isinstance(default_visit_minutes, bool)
            or not isinstance(default_visit_minutes, (int, float))
            or not isfinite(default_visit_minutes)
            or not 15 <= float(default_visit_minutes) <= 480
        ):
            raise WorkspaceError("draft_evaluator_time_window_config_invalid", 409)
        self._parent = parent
        self._route_matrix = route_matrix
        self._start_anchors = dict(start_anchor_by_day)
        self._end_anchors = dict(end_anchor_by_day)
        self._default_visit_minutes = float(default_visit_minutes)
        self._day_start_minute = _configured_clock_minute(day_start_time)
        self._config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=max_day_minutes,
            day_start_time=day_start_time,
            default_visit_minutes=default_visit_minutes,
            start_anchor_by_day=dict(start_anchor_by_day),
            end_anchor_by_day=dict(end_anchor_by_day),
            require_lodging_assignments=bool(parent.lodging_assignments),
            include_schedule_metrics=True,
        )

    def compile(
        self,
        operations: tuple[dict[str, Any], ...],
    ) -> EvaluatedStopEditArtifacts:
        if not operations:
            raise WorkspaceError("draft_is_empty")
        stop_lookup = {
            str(stop.get("stop_id") or stop.get("poi_id") or ""): dict(stop)
            for stop in self._parent.selected_stops
        }
        if not stop_lookup or any(not stop_id for stop_id in stop_lookup):
            raise WorkspaceError("draft_parent_time_window_invalid", 409)
        for stop in stop_lookup.values():
            _validate_parent_time_window_record(stop)
            _validate_parent_duration_record(stop)

        changed_stops = {stop_id: dict(stop) for stop_id, stop in stop_lookup.items()}
        seen_targets: set[str] = set()
        for operation in operations:
            if operation.get("type") != "set_stop_time_window":
                raise WorkspaceError(
                    "draft_evaluated_operation_combination_unsupported",
                    409,
                )
            target = str(operation.get("target") or "")
            if target not in stop_lookup:
                raise WorkspaceError("draft_target_not_in_parent", 409)
            if target in seen_targets:
                raise WorkspaceError("draft_conflicting_attribute_edits", 409)
            seen_targets.add(target)
            constraint = _canonical_time_window_parameters(
                operation.get("parameters")
            )
            if stop_lookup[target].get("time_window_constraint") == constraint:
                raise WorkspaceError("draft_no_effect", 409)
            changed_stops[target]["time_window_constraint"] = constraint

        semantic_operations = tuple(
            {
                "type": operation["type"],
                "target": operation["target"],
                "parameters": dict(operation["parameters"]),
            }
            for operation in operations
        )
        request_seed = {
            "parent_plan_id": self._parent.plan_id,
            "parent_content_hash": self._parent.content_hash,
            "route_matrix_id": self._route_matrix.matrix_id,
            "operations": semantic_operations,
            "method": self.METHOD_ID,
        }
        request_hash = stable_content_hash(request_seed)
        source_request_id = f"w4r_time_window_{request_hash}"
        source_run_id = f"w4r_time_window_run_{request_hash}"
        child_plan_id = (
            f"plan_{stable_content_hash({**request_seed, 'source_run_id': source_run_id})}"
        )
        selected_stops = tuple(
            changed_stops[str(stop.get("stop_id") or stop.get("poi_id") or "")]
            for stop in self._parent.selected_stops
        )
        child = replace(
            self._parent,
            plan_id=child_plan_id,
            parent_plan_id=self._parent.plan_id,
            source_run_id=source_run_id,
            planning_request_id=source_request_id,
            selected_stops=selected_stops,
            modeled_metrics={"selected_attractions": float(len(selected_stops))},
            context_exposure_components={},
            change_components={},
            certificate_id=None,
            created_at="",
        )
        planner_run = PlannerRun(
            run_id=source_run_id,
            parent_run_id=self._parent.source_run_id or None,
            planning_request_id=source_request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id=self.METHOD_ID,
            method_requested=self.METHOD_ID,
            method_executed=self.METHOD_ID,
            execution_status="COMPLETED",
            solver_certification="FEASIBILITY_CERTIFIED",
            solver_backend="independent_plan_evaluator",
            result_plan_id=child.plan_id,
        )
        diff = PlanDiffBuilder(default_visit_minutes=self._default_visit_minutes).diff(
            self._parent,
            child,
        )
        if (
            diff.schema_version != "plan-diff-v3"
            or len(diff.time_window_changes) != len(operations)
            or diff.added_stops
            or diff.deleted_stops
            or diff.day_moves
            or diff.time_shifts
            or diff.duration_changes
            or diff.reorder_changes
            or diff.lodging_changes
            or diff.road_changes
        ):
            raise WorkspaceError("draft_pipeline_diff_invalid", 409)
        evaluator = PlanEvaluator(
            route_matrix=self._route_matrix,
            planner_runs={source_run_id: planner_run},
            config=self._config,
            reference_plan=self._parent,
        )
        certificate = evaluator.evaluate_final_plan(child, planner_run=planner_run)
        parent_schedule = evaluator.evaluate_time_window_schedule_accounting(
            self._parent
        )
        child_schedule = evaluator.evaluate_time_window_schedule_accounting(child)
        schedule_payload = {
            "schema_version": "evaluated-time-window-schedule-impact-v1",
            "evaluator_version": evaluator.evaluator_version,
            "route_matrix_id": self._route_matrix.matrix_id,
            "configured_default_visit_minutes": self._default_visit_minutes,
            "configured_day_start_minute": self._day_start_minute,
            "route_leg_evidence": _time_window_route_evidence(
                self._parent,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            "time_window_cost_policy": {
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
            },
            "affected_days": sorted(
                {
                    change.day
                    for change in diff.time_window_changes
                    if change.day is not None
                }
            ),
            "target_stop_ids": sorted(
                change.stop_id for change in diff.time_window_changes
            ),
            "parent": parent_schedule,
            "child": child_schedule,
        }
        schedule_impact = {
            **schedule_payload,
            "content_hash": stable_content_hash(schedule_payload),
        }
        return EvaluatedStopEditArtifacts(
            source_request_id=source_request_id,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            planner_run=planner_run,
            parent_route_legs=_route_legs(
                self._parent,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            route_legs=_route_legs(
                child,
                self._route_matrix,
                self._start_anchors,
                self._end_anchors,
            ),
            schedule_impact=schedule_impact,
        )


def _canonical_evaluated_role(parameters: Any) -> str:
    if not isinstance(parameters, dict) or set(parameters) != {"role"}:
        raise WorkspaceError("invalid_stop_role", 422)
    role = parameters.get("role")
    if role not in EVALUATED_STOP_ROLES:
        if isinstance(role, str) and role in ALL_STOP_ROLES:
            raise WorkspaceError("stop_role_evaluation_not_supported", 409)
        raise WorkspaceError("invalid_stop_role", 422)
    return role


def _validate_parent_role_record(stop: dict[str, Any]) -> None:
    role = stop.get("itinerary_role")
    source = stop.get("itinerary_role_source")
    if role is None:
        if source not in {None, "unavailable"}:
            raise WorkspaceError("draft_parent_role_invalid", 409)
        return
    if (
        not isinstance(role, str)
        or role not in ALL_STOP_ROLES
        or source != ROLE_SOURCE
    ):
        raise WorkspaceError("draft_parent_role_invalid", 409)


def _role_constraints_by_target(
    plan: PlanArtifactV2,
) -> dict[str, tuple[Any, ...]]:
    try:
        constraints = active_owned_constraints(
            tuple(dict(record) for record in plan.owned_constraints)
        )
    except Exception as exc:
        raise WorkspaceError("draft_parent_role_constraint_invalid", 409) from exc
    result: dict[str, list[Any]] = {}
    for constraint in constraints:
        if (
            constraint.scope not in {ConstraintScope.STOP, ConstraintScope.POI}
            or constraint.relation.strip().lower() not in _ROLE_RELATIONS
        ):
            continue
        if (
            not isinstance(constraint.value, str)
            or constraint.value not in ALL_STOP_ROLES
        ):
            raise WorkspaceError("draft_parent_role_constraint_invalid", 409)
        result.setdefault(constraint.target_id, []).append(constraint)
    return {target: tuple(rows) for target, rows in result.items()}


def _enforce_role_constraints(
    target: str,
    role: str,
    constraints: tuple[Any, ...],
) -> None:
    for constraint in constraints:
        if constraint.value == role:
            continue
        if constraint.strength in {
            ConstraintStrength.LOCKED,
            ConstraintStrength.BOOKED,
            ConstraintStrength.HARD,
        } or not constraint.can_relax():
            raise WorkspaceError("role_edit_permission_required", 409)


def _role_states(plan: PlanArtifactV2) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for stop in plan.selected_stops:
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or "")
        role = stop.get("itinerary_role")
        states.append(
            {
                "stop_id": stop_id,
                "day": stop.get("day"),
                "itinerary_role": role,
                "itinerary_role_source": (
                    ROLE_SOURCE if role is not None else "unavailable"
                ),
            }
        )
    return states


def _accounting_without_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"plan_id", "plan_content_hash", "content_hash"}
    }


def _canonical_time_window_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict) or set(parameters) != {
        "earliest_arrival",
        "latest_departure",
    }:
        raise WorkspaceError("invalid_stop_time_window", 422)
    earliest = parameters.get("earliest_arrival")
    latest = parameters.get("latest_departure")
    if (
        (earliest is None and latest is None)
        or not _valid_clock(earliest)
        or not _valid_clock(latest)
        or (
            earliest is not None
            and latest is not None
            and _clock_minutes(earliest) > _clock_minutes(latest)
        )
    ):
        raise WorkspaceError("invalid_stop_time_window", 422)
    return {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": earliest,
        "latest_departure": latest,
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }


def _configured_clock_minute(value: Any) -> float:
    if isinstance(value, bool):
        raise WorkspaceError("draft_evaluator_time_window_config_invalid", 409)
    if isinstance(value, str):
        if not _valid_clock(value):
            raise WorkspaceError("draft_evaluator_time_window_config_invalid", 409)
        return float(_clock_minutes(value))
    if (
        not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= float(value) < 1440
    ):
        raise WorkspaceError("draft_evaluator_time_window_config_invalid", 409)
    return float(value)


def _time_window_route_evidence(
    plan: PlanArtifactV2,
    route_matrix: RouteMatrix,
    start_anchors: dict[int, str],
    end_anchors: dict[int, str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for day, stop_ids in sorted(_ordered_stop_ids(plan).items()):
        sequence = [start_anchors.get(day), *stop_ids, end_anchors.get(day)]
        sequence = [value for value in sequence if value]
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False):
            cell = _required_cell(route_matrix, origin_id, destination_id)
            evidence.append(
                {
                    "day": day,
                    "origin_id": origin_id,
                    "destination_id": destination_id,
                    "duration_minutes": float(cell.duration_s) / 60.0,
                    "query_hash": cell.query_hash,
                    "road_validated": cell.road_validated,
                    "fallback_used": cell.fallback_used,
                }
            )
    return evidence


def _validate_parent_time_window_record(stop: dict[str, Any]) -> None:
    value = stop.get("time_window_constraint")
    if value is None:
        return
    if not isinstance(value, dict):
        raise WorkspaceError("draft_parent_time_window_invalid", 409)
    try:
        expected = _canonical_time_window_parameters(
            {
                "earliest_arrival": value.get("earliest_arrival"),
                "latest_departure": value.get("latest_departure"),
            }
        )
    except WorkspaceError:
        raise WorkspaceError("draft_parent_time_window_invalid", 409) from None
    if value != expected:
        raise WorkspaceError("draft_parent_time_window_invalid", 409)


def _valid_clock(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is not None
    )


def _clock_minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _canonical_exact_duration(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict) or set(parameters) != {"duration"}:
        raise WorkspaceError("draft_operation_parameters_invalid", 409)
    value = parameters.get("duration")
    fields = {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
    if not isinstance(value, dict) or set(value) != fields:
        raise WorkspaceError("invalid_stop_duration", 422)
    preferred = value.get("preferred_minutes")
    if (
        value.get("mode") != "exact"
        or isinstance(preferred, bool)
        or not isinstance(preferred, int)
        or not 15 <= preferred <= 480
        or value.get("minimum_minutes") != preferred
        or value.get("maximum_minutes") != preferred
    ):
        mode = value.get("mode")
        if mode in {"preferred", "minimum", "maximum", "range"}:
            raise WorkspaceError("duration_mode_evaluation_not_supported", 409)
        raise WorkspaceError("invalid_stop_duration", 422)
    return {
        "mode": "exact",
        "preferred_minutes": preferred,
        "minimum_minutes": preferred,
        "maximum_minutes": preferred,
    }


def _is_explicit_exact_no_effect(stop: dict[str, Any], duration: dict[str, Any]) -> bool:
    constraint = stop.get("duration_constraint")
    if not isinstance(constraint, dict) or constraint != duration:
        return False
    scalar = next(
        (
            stop.get(key)
            for key in (
                "visit_duration_minutes",
                "duration_minutes",
                "service_minutes",
                "dwell_minutes",
            )
            if stop.get(key) is not None
        ),
        None,
    )
    return isinstance(scalar, int) and not isinstance(scalar, bool) and scalar == duration["preferred_minutes"]


def _validate_parent_duration_record(stop: dict[str, Any]) -> None:
    scalar_keys = (
        "visit_duration_minutes",
        "duration_minutes",
        "service_minutes",
        "dwell_minutes",
    )
    scalar_values: list[int] = []
    for key in scalar_keys:
        value = stop.get(key)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 15 <= value <= 480
        ):
            raise WorkspaceError("draft_parent_duration_invalid", 409)
        scalar_values.append(value)
    if len(set(scalar_values)) > 1:
        raise WorkspaceError("draft_parent_duration_invalid", 409)

    constraint = stop.get("duration_constraint")
    if constraint is None:
        return
    if not isinstance(constraint, dict):
        raise WorkspaceError("draft_parent_duration_invalid", 409)
    try:
        canonical = _canonical_exact_duration({"duration": constraint})
    except WorkspaceError as exc:
        raise WorkspaceError("draft_parent_duration_invalid", 409) from exc
    if not scalar_values or scalar_values[0] != canonical["preferred_minutes"]:
        raise WorkspaceError("draft_parent_duration_invalid", 409)


def _route_legs(
    plan: PlanArtifactV2,
    route_matrix: RouteMatrix,
    start_anchors: dict[int, str],
    end_anchors: dict[int, str],
) -> tuple[dict[str, Any], ...]:
    ordered_by_day = _ordered_stop_ids(plan)
    result: list[dict[str, Any]] = []
    previous_destination: str | None = None
    for day, stop_ids in sorted(ordered_by_day.items()):
        sequence: list[str] = []
        start = start_anchors.get(day)
        end = end_anchors.get(day)
        if start:
            sequence.append(start)
        sequence.extend(stop_ids)
        if end:
            sequence.append(end)
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False):
            if previous_destination is not None and previous_destination != origin_id:
                raise WorkspaceError("draft_route_sequence_discontinuous", 409)
            _required_cell(route_matrix, origin_id, destination_id)
            result.append(
                {
                    "day": day,
                    "origin_id": origin_id,
                    "destination_id": destination_id,
                    "evidence_scope": "evaluated_direct_edit_route_leg",
                }
            )
            previous_destination = destination_id
    return tuple(result)


def _ordered_stop_ids(plan: PlanArtifactV2) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    seen: set[str] = set()
    for row in plan.ordered_days:
        day = row.get("day")
        stop_ids = row.get("stop_ids")
        if (
            isinstance(day, bool)
            or not isinstance(day, int)
            or not isinstance(stop_ids, (list, tuple))
        ):
            raise WorkspaceError("draft_parent_order_invalid", 409)
        values = [str(stop_id) for stop_id in stop_ids]
        if not values or any(not stop_id or stop_id in seen for stop_id in values):
            raise WorkspaceError("draft_parent_order_invalid", 409)
        result[day] = values
        seen.update(values)
    if seen != set(plan.sequence) or tuple(
        stop_id for _, stop_ids in sorted(result.items()) for stop_id in stop_ids
    ) != tuple(plan.sequence):
        raise WorkspaceError("draft_parent_order_invalid", 409)
    return result


def _reordered_stop_records(
    parent: PlanArtifactV2,
    ordered_by_day: dict[int, list[str]],
) -> tuple[dict[str, Any], ...]:
    lookup = {
        str(stop.get("stop_id") or stop.get("poi_id") or ""): dict(stop)
        for stop in parent.selected_stops
    }
    if not lookup or any(not stop_id for stop_id in lookup):
        raise WorkspaceError("draft_parent_order_invalid", 409)
    result: list[dict[str, Any]] = []
    for day, stop_ids in sorted(ordered_by_day.items()):
        for stop_order, stop_id in enumerate(stop_ids, start=1):
            stop = lookup.get(stop_id)
            if stop is None or stop.get("day") != day:
                raise WorkspaceError("draft_parent_order_invalid", 409)
            result.append({**stop, "day": day, "stop_order": stop_order})
    return tuple(result)


def _required_cell(route_matrix: RouteMatrix, origin_id: str, destination_id: str) -> Any:
    cell = route_matrix.cells.get((origin_id, destination_id))
    if cell is None:
        raise WorkspaceError("draft_candidate_route_evidence_missing", 409)
    try:
        cell.require_publication_eligible()
    except Exception as exc:
        raise WorkspaceError("draft_candidate_route_evidence_missing", 409) from exc
    if len(cell.geometry) < 2:
        raise WorkspaceError("draft_candidate_route_evidence_missing", 409)
    return cell
