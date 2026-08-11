"""Independent final-plan evaluator for PlanArtifactV2 records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..plans import ConstraintScope, ConstraintStrength, active_owned_constraints
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixError, validate_route_matrix
from .certificate import EvaluationFinding, PlanEvaluationCertificate

_ITINERARY_STOP_ROLES = frozenset(
    {
        "attraction",
        "activity",
        "meal",
        "lodging",
        "transport_hub",
        "rest_stop",
        "scenic_stop",
        "route_waypoint",
        "origin",
        "destination",
    }
)


@dataclass(frozen=True)
class PlanEvaluatorConfig:
    """Configuration for independent final-plan checks."""

    strict_routes: bool = True
    max_day_minutes: float = 480.0
    day_start_time: str | int | float = "09:00"
    default_visit_minutes: float = 60.0
    enforce_opening_windows: bool = True
    start_anchor_by_day: dict[int, str] = field(default_factory=dict)
    end_anchor_by_day: dict[int, str] = field(default_factory=dict)
    weather_warning_threshold: float | None = None
    budget_limit: float | None = None
    budget_metric_names: tuple[str, ...] = ("total_cost", "estimated_cost", "budget_cost")
    required_stop_ids: tuple[str, ...] = ()
    excluded_stop_ids: tuple[str, ...] = ()
    closed_route_ids: tuple[str, ...] = ()
    unavailable_lodging_ids: tuple[str, ...] = ()
    require_lodging_assignments: bool = True
    include_schedule_metrics: bool = False


class PlanEvaluator:
    """Recompute final-plan eligibility from artifacts and context evidence."""

    evaluator_version = "verify-001-plan-evaluator-v1"

    def __init__(
        self,
        *,
        route_matrix: RouteMatrix | None = None,
        planner_runs: Mapping[str, PlannerRun | dict[str, Any]] | None = None,
        config: PlanEvaluatorConfig | None = None,
        reference_plan: PlanArtifactV2 | None = None,
    ) -> None:
        self.route_matrix = route_matrix
        self.planner_runs = dict(planner_runs or {})
        self.config = config or PlanEvaluatorConfig()
        self.reference_plan = reference_plan

    def evaluate(self, plan: PlanArtifactV2) -> PlanEvaluationCertificate:
        return self.evaluate_final_plan(plan)

    def evaluate_final_plan(
        self,
        plan: PlanArtifactV2,
        *,
        planner_run: PlannerRun | dict[str, Any] | None = None,
        route_matrix: RouteMatrix | None = None,
        expected_content_hash: str | None = None,
        expected_certificate: PlanEvaluationCertificate | None = None,
    ) -> PlanEvaluationCertificate:
        failures: list[EvaluationFinding] = []
        warnings: list[EvaluationFinding] = []
        metrics: dict[str, float] = {
            "selected_stop_count": float(len(plan.selected_stops)),
            "sequence_stop_count": float(len(plan.sequence)),
        }
        if self.reference_plan is not None:
            metrics.update(_quality_metrics(self.reference_plan, plan))
        resolved_run = _coerce_run(planner_run) or _coerce_run(self.planner_runs.get(plan.source_run_id))
        matrix = route_matrix or self.route_matrix
        stop_lookup = _stop_lookup(plan)
        day_sequences = _day_sequences(plan)

        self._check_artifact_linkage(
            plan,
            resolved_run,
            expected_content_hash,
            expected_certificate,
            failures,
            warnings,
        )
        self._check_plan_shape(plan, stop_lookup, day_sequences, failures, warnings)
        self._check_owned_constraints(plan, failures, warnings)
        self._check_context_requirements(plan, failures)
        self._check_lodging(plan, stop_lookup, day_sequences, failures)
        self._check_budget_weather_and_closures(plan, stop_lookup, failures, warnings)
        route_record = self._check_routes_and_schedule(plan, matrix, stop_lookup, day_sequences, failures, metrics)
        if self.config.include_schedule_metrics:
            missing_windows = tuple(
                sorted(
                    stop_id
                    for stop_id, stop in stop_lookup.items()
                    if _opening_window(stop) == (None, None)
                )
            )
            metrics["schedule_missing_opening_window_count"] = float(len(missing_windows))
            if missing_windows:
                warnings.append(
                    _warning(
                        "opening_window_evidence_missing",
                        "Opening-window evidence is unavailable for one or more evaluated stops.",
                        "soft",
                        missing_windows,
                    )
                )

        artifact_failed = any(finding.category == "artifact" for finding in failures)
        hard_failed = any(finding.category != "artifact" for finding in failures)
        artifact_status = "FAILED" if artifact_failed else "PASSED"
        hard_status = "FAILED" if hard_failed else "PASSED"
        if failures:
            evaluation_status = "FAILED"
            eligibility = "ineligible"
        elif warnings:
            evaluation_status = "PASSED_WITH_WARNINGS"
            eligibility = "eligible"
        else:
            evaluation_status = "PASSED"
            eligibility = "eligible"
        certificate_seed = {
            "plan_id": plan.plan_id,
            "plan_content_hash": plan.content_hash,
            "source_run_id": plan.source_run_id,
            "failures": [finding.code for finding in failures],
            "warnings": [finding.code for finding in warnings],
        }
        return PlanEvaluationCertificate(
            certificate_id=f"cert_{stable_content_hash(certificate_seed)}",
            plan_id=plan.plan_id,
            source_run_id=plan.source_run_id,
            plan_content_hash=plan.content_hash,
            evaluator_version=self.evaluator_version,
            artifact_grounding_status=artifact_status,
            hard_feasibility_status=hard_status,
            evaluation_status=evaluation_status,
            comparison_eligibility=eligibility,
            failures=tuple(failures),
            warnings=tuple(warnings),
            metrics=metrics,
            route_validation=route_record,
        )

    def evaluate_schedule_accounting(
        self,
        plan: PlanArtifactV2,
        *,
        route_matrix: RouteMatrix | None = None,
    ) -> dict[str, Any]:
        """Return evaluator-owned schedule accounting without certifying a plan."""

        matrix = route_matrix or self.route_matrix
        stop_lookup = _stop_lookup(plan)
        day_sequences = _day_sequences(plan)
        failures: list[EvaluationFinding] = []
        metrics: dict[str, float] = {}
        route_validation = self._check_routes_and_schedule(
            plan,
            matrix,
            stop_lookup,
            day_sequences,
            failures,
            metrics,
        )
        missing_windows = tuple(
            sorted(
                stop_id
                for stop_id, stop in stop_lookup.items()
                if _opening_window(stop) == (None, None)
            )
        )
        payload = {
            "schema_version": "evaluated-schedule-accounting-v1",
            "evaluator_version": self.evaluator_version,
            "plan_id": plan.plan_id,
            "plan_content_hash": plan.content_hash,
            "route_matrix_id": route_validation.get("matrix_id") if route_validation else None,
            "status": (
                "failed"
                if failures
                else "incomplete_evidence"
                if missing_windows
                else "passed"
            ),
            "modeled_components": [
                "road_travel",
                "visit_duration",
                "opening_wait_when_window_available",
                "day_limit",
            ],
            "unavailable_components": [
                "parking_dropoff",
                "walking_transfer",
                "queue_wait",
                "service_buffer",
            ],
            "blocking_codes": list(dict.fromkeys(finding.code for finding in failures)),
            "missing_opening_window_stop_ids": list(missing_windows),
            "metrics": metrics,
        }
        return {**payload, "content_hash": stable_content_hash(payload)}

    def evaluate_time_window_schedule_accounting(
        self,
        plan: PlanArtifactV2,
        *,
        route_matrix: RouteMatrix | None = None,
    ) -> dict[str, Any]:
        """Return evaluator-owned required-window accounting with stop traces."""

        matrix = route_matrix or self.route_matrix
        stop_lookup = _stop_lookup(plan)
        day_sequences = _day_sequences(plan)
        failures: list[EvaluationFinding] = []
        metrics: dict[str, float] = {}
        stop_traces: list[dict[str, Any]] = []
        route_validation = self._check_routes_and_schedule(
            plan,
            matrix,
            stop_lookup,
            day_sequences,
            failures,
            metrics,
            stop_traces=stop_traces,
        )
        missing_windows = tuple(
            sorted(
                stop_id
                for stop_id, stop in stop_lookup.items()
                if _opening_window(stop) == (None, None)
            )
        )
        payload = {
            "schema_version": "evaluated-time-window-schedule-accounting-v1",
            "evaluator_version": self.evaluator_version,
            "plan_id": plan.plan_id,
            "plan_content_hash": plan.content_hash,
            "route_matrix_id": (
                route_validation.get("matrix_id") if route_validation else None
            ),
            "status": (
                "failed"
                if failures
                else "incomplete_evidence"
                if missing_windows
                else "passed"
            ),
            "modeled_components": [
                "road_travel",
                "visit_duration",
                "opening_wait_when_window_available",
                "required_time_window_wait",
                "required_latest_departure",
                "day_limit",
            ],
            "unavailable_components": [
                "parking_dropoff",
                "walking_transfer",
                "queue_wait",
                "service_buffer",
            ],
            "blocking_codes": list(
                dict.fromkeys(finding.code for finding in failures)
            ),
            "missing_opening_window_stop_ids": list(missing_windows),
            "metrics": metrics,
            "stop_traces": stop_traces,
        }
        return {**payload, "content_hash": stable_content_hash(payload)}

    def _check_artifact_linkage(
        self,
        plan: PlanArtifactV2,
        planner_run: PlannerRun | None,
        expected_content_hash: str | None,
        expected_certificate: PlanEvaluationCertificate | None,
        failures: list[EvaluationFinding],
        warnings: list[EvaluationFinding],
    ) -> None:
        if not plan.plan_id:
            failures.append(_failure("plan_id_missing", "Plan artifact is missing plan_id.", "artifact"))
        if not plan.source_run_id:
            failures.append(_failure("source_run_missing", "Plan artifact is missing source_run_id.", "artifact"))
        if not plan.selected_stops:
            failures.append(_failure("selected_stops_missing", "Plan artifact has no selected stops.", "artifact"))
        if expected_content_hash is not None and plan.content_hash != expected_content_hash:
            failures.append(
                _failure(
                    "expected_content_hash_mismatch",
                    "Plan content hash does not match the expected frozen content hash.",
                    "artifact",
                )
            )
        if expected_certificate is not None and not expected_certificate.valid_for_plan(plan):
            failures.append(
                _failure(
                    "certificate_content_hash_mismatch",
                    "Existing certificate does not apply to the current plan content hash.",
                    "artifact",
                    (expected_certificate.certificate_id,),
                )
            )
        if planner_run is None:
            failures.append(_failure("source_run_not_found", "No matching planner run was available for the plan.", "artifact"))
            return
        if plan.source_run_id != planner_run.run_id:
            failures.append(_failure("source_run_mismatch", "Plan source_run_id does not match planner run.", "artifact"))
        if plan.planning_request_id != planner_run.planning_request_id:
            failures.append(_failure("planning_request_mismatch", "Plan request does not match planner run.", "artifact"))
        if plan.catalog_snapshot_id != planner_run.catalog_snapshot_id:
            failures.append(_failure("catalog_snapshot_mismatch", "Plan catalog snapshot does not match planner run.", "artifact"))
        if plan.context_snapshot_id != planner_run.context_snapshot_id:
            failures.append(_failure("context_snapshot_mismatch", "Plan context snapshot does not match planner run.", "artifact"))
        if planner_run.execution_status not in {"COMPLETED", "FALLBACK_USED"}:
            failures.append(
                _failure(
                    "planner_run_not_successful",
                    "The linked planner run did not complete successfully.",
                    "artifact",
                    (planner_run.run_id,),
                )
            )
        if planner_run.solver_certification == "INVALIDATED_AFTER_EDIT":
            failures.append(
                _failure(
                    "planner_certification_invalidated",
                    "The linked planner certification was invalidated after a post-solve edit.",
                    "artifact",
                    (planner_run.run_id,),
                )
            )
        if planner_run.result_plan_id and planner_run.result_plan_id != plan.plan_id:
            warnings.append(
                _warning(
                    "planner_run_result_plan_mismatch",
                    "Planner run result_plan_id does not match this plan_id.",
                    "artifact",
                    (planner_run.run_id,),
                )
            )

    def _check_plan_shape(
        self,
        plan: PlanArtifactV2,
        stop_lookup: dict[str, dict[str, Any]],
        day_sequences: dict[int, tuple[str, ...]],
        failures: list[EvaluationFinding],
        warnings: list[EvaluationFinding],
    ) -> None:
        sequence = tuple(str(stop_id) for stop_id in plan.sequence)
        selected_stop_ids = tuple(
            str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name") or "").strip()
            for stop in plan.selected_stops
        )
        nonempty_selected_ids = tuple(stop_id for stop_id in selected_stop_ids if stop_id)
        duplicate_selected_ids = sorted(
            stop_id for stop_id in set(nonempty_selected_ids) if nonempty_selected_ids.count(stop_id) > 1
        )
        if duplicate_selected_ids:
            failures.append(
                _failure(
                    "duplicate_selected_stop_record",
                    "Selected stop records contain duplicate identifiers.",
                    "artifact",
                    tuple(duplicate_selected_ids),
                )
            )
        invalid_role_stop_ids = tuple(
            sorted(
                stop_id
                for stop_id, stop in stop_lookup.items()
                if stop.get("itinerary_role") is not None
                and (
                    not isinstance(stop.get("itinerary_role"), str)
                    or stop.get("itinerary_role") not in _ITINERARY_STOP_ROLES
                    or stop.get("itinerary_role_source")
                    != "user_declared_itinerary_role"
                )
            )
        )
        invalid_missing_role_source_ids = tuple(
            sorted(
                stop_id
                for stop_id, stop in stop_lookup.items()
                if stop.get("itinerary_role") is None
                and stop.get("itinerary_role_source") not in {None, "unavailable"}
            )
        )
        if invalid_role_stop_ids or invalid_missing_role_source_ids:
            failures.append(
                _failure(
                    "itinerary_stop_role_invalid",
                    "A trip-owned itinerary role or its provenance is invalid.",
                    "artifact",
                    tuple(
                        sorted(
                            set(invalid_role_stop_ids)
                            | set(invalid_missing_role_source_ids)
                        )
                    ),
                )
            )
        if len(sequence) != len(set(sequence)):
            failures.append(_failure("duplicate_visit", "Plan sequence contains duplicate stop visits.", "hard"))
        assigned_stop_ids = {
            stop_id
            for stop_ids in day_sequences.values()
            for stop_id in stop_ids
        }
        for stop_id in sequence:
            if stop_id not in stop_lookup:
                failures.append(_failure("sequence_stop_missing_record", f"Sequence stop {stop_id!r} lacks a stop record.", "artifact"))
            if stop_id not in assigned_stop_ids:
                failures.append(_failure("day_assignment_missing", f"Sequence stop {stop_id!r} lacks a day assignment.", "hard"))
        selected_ids = set(stop_lookup)
        missing_from_sequence = selected_ids.difference(sequence)
        if missing_from_sequence:
            failures.append(
                _failure(
                    "selected_stop_not_in_sequence",
                    "Selected stop records are not all present in the displayed sequence.",
                    "artifact",
                    tuple(sorted(missing_from_sequence)),
                )
            )
        if not day_sequences:
            failures.append(_failure("day_sequences_missing", "Plan has no day-grouped displayed sequence.", "artifact"))

    def _check_owned_constraints(
        self,
        plan: PlanArtifactV2,
        failures: list[EvaluationFinding],
        warnings: list[EvaluationFinding],
    ) -> None:
        selected = set(str(stop_id) for stop_id in plan.sequence)
        stop_lookup = _stop_lookup(plan)
        lodging_ids = set(str(value) for value in plan.lodging_assignments.values())
        for constraint in active_owned_constraints(tuple(dict(record) for record in plan.owned_constraints)):
            hard = constraint.strength in {
                ConstraintStrength.LOCKED,
                ConstraintStrength.BOOKED,
                ConstraintStrength.HARD,
                ConstraintStrength.STRONG,
            }
            if constraint.scope in {ConstraintScope.STOP, ConstraintScope.POI} and constraint.target_id not in selected:
                code = "locked_stop_missing" if constraint.strength == ConstraintStrength.LOCKED else "owned_stop_missing"
                finding = _failure(code, f"Owned stop constraint {constraint.constraint_id!r} is not satisfied.", "hard", (constraint.constraint_id,))
                (failures if hard else warnings).append(finding if hard else _warning(code, finding.message, "soft", finding.evidence_refs))
            if _is_role_owned_constraint(constraint):
                required_role = constraint.value
                if (
                    not isinstance(required_role, str)
                    or required_role not in _ITINERARY_STOP_ROLES
                ):
                    failures.append(
                        _failure(
                            "owned_role_constraint_invalid",
                            "An owned itinerary-role constraint has an invalid value.",
                            "artifact",
                            (constraint.constraint_id,),
                        )
                    )
                elif constraint.target_id in stop_lookup and (
                    stop_lookup[constraint.target_id].get("itinerary_role")
                    != required_role
                ):
                    role_hard = constraint.strength in {
                        ConstraintStrength.LOCKED,
                        ConstraintStrength.BOOKED,
                        ConstraintStrength.HARD,
                    } or not constraint.can_relax()
                    code = "owned_role_constraint_unsatisfied"
                    message = (
                        f"Owned itinerary-role constraint "
                        f"{constraint.constraint_id!r} is not satisfied."
                    )
                    if role_hard:
                        failures.append(
                            _failure(
                                code,
                                message,
                                "hard",
                                (constraint.constraint_id,),
                            )
                        )
                    else:
                        warnings.append(
                            _warning(
                                code,
                                message,
                                "soft",
                                (constraint.constraint_id,),
                            )
                        )
            if constraint.scope == ConstraintScope.LODGING and constraint.target_id not in lodging_ids:
                code = "booked_lodging_missing" if constraint.strength == ConstraintStrength.BOOKED else "owned_lodging_missing"
                finding = _failure(code, f"Owned lodging constraint {constraint.constraint_id!r} is not satisfied.", "hard", (constraint.constraint_id,))
                (failures if hard else warnings).append(finding if hard else _warning(code, finding.message, "soft", finding.evidence_refs))

    def _check_context_requirements(
        self,
        plan: PlanArtifactV2,
        failures: list[EvaluationFinding],
    ) -> None:
        selected = set(str(stop_id) for stop_id in plan.sequence)
        for stop_id in self.config.required_stop_ids:
            if stop_id not in selected:
                failures.append(
                    _failure(
                        "context_required_stop_missing",
                        f"Required contextual stop {stop_id!r} is missing.",
                        "hard",
                        (stop_id,),
                    )
                )
        for stop_id in self.config.excluded_stop_ids:
            if stop_id in selected:
                failures.append(
                    _failure(
                        "context_excluded_stop_selected",
                        f"Context-infeasible stop {stop_id!r} remains selected.",
                        "hard",
                        (stop_id,),
                    )
                )
        closed_routes = set(self.config.closed_route_ids)
        for day, route_id in plan.route_ids_by_day.items():
            if str(route_id) in closed_routes:
                failures.append(
                    _failure(
                        "context_closed_route_selected",
                        f"Day {day} retains closed route {route_id!r}.",
                        "hard",
                        (str(route_id), str(day)),
                    )
                )
        unavailable_lodging = set(self.config.unavailable_lodging_ids)
        for day, lodging_id in plan.lodging_assignments.items():
            if str(lodging_id) in unavailable_lodging:
                failures.append(
                    _failure(
                        "context_unavailable_lodging_selected",
                        f"Day {day} retains unavailable lodging {lodging_id!r}.",
                        "hard",
                        (str(lodging_id), str(day)),
                    )
                )

    def _check_lodging(
        self,
        plan: PlanArtifactV2,
        stop_lookup: dict[str, dict[str, Any]],
        day_sequences: dict[int, tuple[str, ...]],
        failures: list[EvaluationFinding],
    ) -> None:
        for day, stop_ids in sorted(day_sequences.items()):
            lodging = str(plan.lodging_assignments.get(str(day), "")).strip()
            if not lodging:
                if self.config.require_lodging_assignments:
                    failures.append(
                        _failure("lodging_missing", f"Day {day} has stops but no lodging assignment.", "hard")
                    )
                continue
            for stop_id in stop_ids:
                stop_lodging = str(stop_lookup.get(stop_id, {}).get("lodging_id", "")).strip()
                if stop_lodging and stop_lodging != lodging:
                    failures.append(
                        _failure(
                            "lodging_mismatch",
                            f"Stop {stop_id!r} lodging does not match the day {day} lodging assignment.",
                            "hard",
                            (stop_id, str(day)),
                        )
                    )

    def _check_budget_weather_and_closures(
        self,
        plan: PlanArtifactV2,
        stop_lookup: dict[str, dict[str, Any]],
        failures: list[EvaluationFinding],
        warnings: list[EvaluationFinding],
    ) -> None:
        if self.config.budget_limit is not None:
            cost = _first_float(plan.modeled_metrics, *self.config.budget_metric_names)
            if cost is None:
                warnings.append(_warning("budget_metric_missing", "Budget limit configured but no cost metric was found.", "soft"))
            elif cost > self.config.budget_limit:
                failures.append(_failure("budget_exceeded", "Plan exceeds the configured budget limit.", "hard"))
        for stop_id, stop in stop_lookup.items():
            if _truthy(stop.get("closed")) or str(stop.get("closure_status", "")).lower() in {"closed", "cancelled", "canceled"}:
                failures.append(_failure("stop_closed", f"Stop {stop_id!r} is closed.", "hard", (stop_id,)))
            if stop.get("weather_feasible") is False:
                failures.append(_failure("weather_infeasible", f"Stop {stop_id!r} is weather infeasible.", "hard", (stop_id,)))
            weather_risk = _coerce_float(stop.get("weather_risk"))
            if (
                self.config.weather_warning_threshold is not None
                and weather_risk is not None
                and weather_risk > self.config.weather_warning_threshold
            ):
                warnings.append(_warning("weather_risk_warning", f"Stop {stop_id!r} exceeds weather risk warning threshold.", "soft", (stop_id,)))

    def _check_routes_and_schedule(
        self,
        plan: PlanArtifactV2,
        route_matrix: RouteMatrix | None,
        stop_lookup: dict[str, dict[str, Any]],
        day_sequences: dict[int, tuple[str, ...]],
        failures: list[EvaluationFinding],
        metrics: dict[str, float],
        *,
        stop_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        route_sequences = tuple(_route_sequence(day, sequence, self.config) for day, sequence in sorted(day_sequences.items()))
        if route_matrix is None:
            failures.append(_failure("route_matrix_missing", "No RouteMatrix was provided for independent route validation.", "hard"))
            return None
        report = validate_route_matrix(
            route_matrix,
            required_sequences=route_sequences,
            require_publication_ready=self.config.strict_routes,
        )
        metrics.update(
            {
                "route_required_leg_count": float(report.required_leg_count),
                "route_present_leg_count": float(report.present_leg_count),
                "route_road_validated_leg_count": float(report.road_validated_leg_count),
                "route_fallback_leg_count": float(report.fallback_leg_count),
                "route_missing_leg_count": float(report.missing_leg_count),
            }
        )
        if not report.publication_ready:
            failures.append(
                _failure(
                    "route_matrix_not_publication_ready",
                    "Route matrix lacks complete road-validated evidence for the displayed plan.",
                    "hard",
                    tuple(report.errors),
                )
            )
        for day, sequence in sorted(day_sequences.items()):
            day_metrics = self._check_day_schedule(
                day,
                sequence,
                route_matrix,
                stop_lookup,
                failures,
                stop_traces=stop_traces,
            )
            if self.config.include_schedule_metrics:
                metrics.update(
                    {f"day_{day}_{name}": value for name, value in day_metrics.items()}
                )
        return report.to_record()

    def _check_day_schedule(
        self,
        day: int,
        sequence: tuple[str, ...],
        route_matrix: RouteMatrix,
        stop_lookup: dict[str, dict[str, Any]],
        failures: list[EvaluationFinding],
        *,
        stop_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        route_sequence = _route_sequence(day, sequence, self.config)
        current_time = float(_parse_minutes(self.config.day_start_time) or 0)
        travel_minutes = 0.0
        visit_minutes = 0.0
        waiting_minutes = 0.0
        current_id = route_sequence[0] if route_sequence and route_sequence[0] not in sequence else None
        for sequence_index, stop_id in enumerate(sequence):
            incoming_origin_id = current_id
            incoming_travel_minutes = 0.0
            incoming_route_query_hash: str | None = None
            if current_id:
                leg_minutes = _route_minutes(route_matrix, current_id, stop_id, strict=self.config.strict_routes)
                if leg_minutes is None:
                    failures.append(_failure("route_leg_unavailable", f"Missing or invalid route leg {current_id!r}->{stop_id!r}.", "hard"))
                    leg_minutes = 0.0
                else:
                    cell = route_matrix.cells.get((current_id, stop_id))
                    incoming_route_query_hash = cell.query_hash if cell is not None else None
                incoming_travel_minutes = float(leg_minutes)
                travel_minutes += leg_minutes
                current_time += leg_minutes
            road_arrival = float(current_time)
            stop = stop_lookup.get(stop_id, {"stop_id": stop_id})
            window_start, window_end = _opening_window(stop)
            opening_window_source = _opening_window_source(stop)
            opening_window_missing = window_start is None and window_end is None
            opening_wait: float | None = None if opening_window_missing else 0.0
            if self.config.enforce_opening_windows and window_start is not None and current_time < window_start:
                opening_wait = float(window_start - current_time)
                waiting_minutes += opening_wait
                current_time = float(window_start)
            trace_failure_codes: list[str] = []
            try:
                required_window = _time_window_constraint(stop.get("time_window_constraint"))
            except ValueError:
                required_window = None
                failures.append(
                    _failure(
                        "time_window_constraint_invalid",
                        f"Stop {stop_id!r} has a malformed required time-window constraint.",
                        "artifact",
                        (stop_id,),
                    )
                )
                trace_failure_codes.append("time_window_constraint_invalid")
            earliest = (
                _parse_minutes(required_window.get("earliest_arrival"))
                if required_window is not None
                else None
            )
            required_wait = None if earliest is None else 0.0
            if earliest is not None and current_time < earliest:
                required_wait = float(earliest - current_time)
                waiting_minutes += required_wait
                current_time = float(earliest)
            service_start = float(current_time)
            if (
                self.config.enforce_opening_windows
                and window_end is not None
                and service_start > window_end
            ):
                failures.append(
                    _failure(
                        "opening_window_missed",
                        f"Stop {stop_id!r} starts service after its opening window.",
                        "hard",
                        (stop_id,),
                    )
                )
                trace_failure_codes.append("opening_window_missed")
            duration, duration_source = _visit_duration_state(
                stop,
                self.config.default_visit_minutes,
            )
            visit_minutes += duration
            current_time += duration
            departure = float(current_time)
            latest = (
                _parse_minutes(required_window.get("latest_departure"))
                if required_window is not None
                else None
            )
            latest_status = "not_set"
            latest_overrun: float | None = None
            if latest is not None:
                latest_overrun = max(0.0, departure - latest)
                latest_status = "violated" if latest_overrun > 0 else "satisfied"
                if latest_overrun > 0:
                    failures.append(
                        _failure(
                            "stop_time_window_latest_departure_exceeded",
                            (
                                f"Stop {stop_id!r} departs after its trip-specific "
                                "latest-departure bound."
                            ),
                            "hard",
                            (stop_id,),
                        )
                    )
                    trace_failure_codes.append(
                        "stop_time_window_latest_departure_exceeded"
                    )
            if stop_traces is not None:
                stop_traces.append(
                    {
                        "stop_id": stop_id,
                        "day": day,
                        "sequence_index": sequence_index,
                        "incoming_origin_id": incoming_origin_id,
                        "incoming_travel_minutes": incoming_travel_minutes,
                        "incoming_route_query_hash": incoming_route_query_hash,
                        "road_arrival_minute": road_arrival,
                        "opening_start_minute": (
                            float(window_start) if window_start is not None else None
                        ),
                        "opening_end_minute": (
                            float(window_end) if window_end is not None else None
                        ),
                        "opening_wait_minutes": opening_wait,
                        "opening_window_source": opening_window_source,
                        "time_window_constraint": required_window,
                        "time_window_source": (
                            "trip_specific_user_constraint"
                            if required_window is not None
                            else None
                        ),
                        "required_window_wait_minutes": required_wait,
                        "service_start_minute": service_start,
                        "visit_minutes": float(duration),
                        "visit_duration_source": duration_source,
                        "departure_minute": departure,
                        "latest_departure_status": latest_status,
                        "latest_departure_overrun_minutes": latest_overrun,
                        "failure_codes": list(dict.fromkeys(trace_failure_codes)),
                    }
                )
            current_id = stop_id
        end_anchor = route_sequence[-1] if route_sequence and route_sequence[-1] not in sequence else None
        if end_anchor and current_id:
            leg_minutes = _route_minutes(route_matrix, current_id, end_anchor, strict=self.config.strict_routes)
            if leg_minutes is None:
                failures.append(_failure("route_leg_unavailable", f"Missing or invalid route leg {current_id!r}->{end_anchor!r}.", "hard"))
                leg_minutes = 0.0
            travel_minutes += leg_minutes
        total = travel_minutes + visit_minutes + waiting_minutes
        slack = max(0.0, self.config.max_day_minutes - total)
        overrun = max(0.0, total - self.config.max_day_minutes)
        if total > self.config.max_day_minutes:
            failures.append(_failure("day_time_exceeded", f"Day {day} exceeds the configured time limit.", "hard", (str(day),)))
        return {
            "travel_minutes": float(travel_minutes),
            "visit_minutes": float(visit_minutes),
            "waiting_minutes": float(waiting_minutes),
            "total_minutes": float(total),
            "limit_minutes": float(self.config.max_day_minutes),
            "slack_minutes": float(slack),
            "overrun_minutes": float(overrun),
        }


def _quality_metrics(reference: PlanArtifactV2, plan: PlanArtifactV2) -> dict[str, float]:
    reference_lookup = _sequence_stop_lookup(reference)
    plan_lookup = _sequence_stop_lookup(plan)
    reference_ids = set(reference_lookup)
    plan_ids = set(plan_lookup)
    reference_utility, reference_utility_key = _utility_total(reference_lookup.values())
    plan_utility, plan_utility_key = _utility_total(plan_lookup.values())
    reference_risk = _complete_metric_total(reference_lookup.values(), "weather_risk")
    plan_risk = _complete_metric_total(plan_lookup.values(), "weather_risk")
    preserved = len(reference_ids & plan_ids)
    metrics = {
        "preserved_stop_count": float(preserved),
        "preservation_rate": float(preserved / max(1, len(reference_ids))),
    }
    if reference_ids == plan_ids and reference_risk is not None and plan_risk is not None:
        metrics["weather_risk_delta"] = float(reference_risk - plan_risk)
    if (
        reference_utility is not None
        and reference_utility > 0
        and plan_utility is not None
        and reference_utility_key == plan_utility_key
    ):
        metrics["utility_retained"] = float(plan_utility / reference_utility)
    return metrics


def _sequence_stop_lookup(plan: PlanArtifactV2) -> dict[str, dict[str, Any]]:
    selected = _stop_lookup(plan)
    return {str(stop_id): selected[str(stop_id)] for stop_id in plan.sequence if str(stop_id) in selected}


def _complete_metric_total(stops: Any, key: str) -> float | None:
    total = 0.0
    stop_count = 0
    for stop in stops:
        stop_count += 1
        value = _coerce_float(stop.get(key))
        if value is None:
            return None
        total += value
    return total if stop_count else None


def _utility_total(stops: Any) -> tuple[float | None, str | None]:
    total = 0.0
    shared_key: str | None = None
    stop_count = 0
    for stop in stops:
        stop_count += 1
        matched_key: str | None = None
        matched_value: float | None = None
        for key in ("utility", "final_poi_value", "score", "rating"):
            value = _coerce_float(stop.get(key))
            if value is not None:
                matched_key = key
                matched_value = value
                break
        if matched_key is None or matched_value is None:
            return None, None
        if shared_key is None:
            shared_key = matched_key
        elif matched_key != shared_key:
            return None, None
        total += matched_value
    if stop_count == 0 or shared_key is None:
        return None, None
    return total, shared_key


def _failure(code: str, message: str, category: str, evidence_refs: tuple[str, ...] = ()) -> EvaluationFinding:
    return EvaluationFinding(code=code, message=message, severity="failure", category=category, evidence_refs=evidence_refs)


def _warning(code: str, message: str, category: str, evidence_refs: tuple[str, ...] = ()) -> EvaluationFinding:
    return EvaluationFinding(code=code, message=message, severity="warning", category=category, evidence_refs=evidence_refs)


def _coerce_run(value: PlannerRun | dict[str, Any] | None) -> PlannerRun | None:
    if value is None or isinstance(value, PlannerRun):
        return value
    return PlannerRun(
        run_id=str(value.get("run_id", "")),
        parent_run_id=value.get("parent_run_id"),
        planning_request_id=str(value.get("planning_request_id", "")),
        catalog_snapshot_id=str(value.get("catalog_snapshot_id", "")),
        context_snapshot_id=str(value.get("context_snapshot_id", "")),
        planner_specification_id=str(value.get("planner_specification_id", "")),
        method_requested=str(value.get("method_requested", "")),
        method_executed=str(value.get("method_executed", "")),
        execution_status=value.get("execution_status", "FAILED"),
        solver_certification=value.get("solver_certification", "NO_CERTIFICATE"),
        result_plan_id=value.get("result_plan_id"),
    )


def _stop_lookup(plan: PlanArtifactV2) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for index, raw_stop in enumerate(plan.selected_stops, start=1):
        stop = dict(raw_stop)
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name") or f"stop_{index}").strip()
        if stop_id:
            lookup[stop_id] = stop
    return lookup


def _day_sequences(plan: PlanArtifactV2) -> dict[int, tuple[str, ...]]:
    ordered_day: dict[str, int] = {}
    for row in plan.ordered_days:
        day = _coerce_int(row.get("day"))
        stop_ids = row.get("stop_ids")
        if day is None or not isinstance(stop_ids, (list, tuple)):
            continue
        for stop_id in stop_ids:
            ordered_day[str(stop_id)] = day
    grouped: dict[int, list[str]] = {}
    for stop_id in plan.sequence:
        key = str(stop_id)
        day = _coerce_int(plan.day_assignments.get(key))
        if day is None:
            day = ordered_day.get(key)
        if day is not None:
            grouped.setdefault(day, []).append(key)
    return {day: tuple(stop_ids) for day, stop_ids in sorted(grouped.items())}


def _route_sequence(day: int, sequence: tuple[str, ...], config: PlanEvaluatorConfig) -> tuple[str, ...]:
    parts: list[str] = []
    start = config.start_anchor_by_day.get(day)
    end = config.end_anchor_by_day.get(day)
    if start:
        parts.append(str(start))
    parts.extend(sequence)
    if end:
        parts.append(str(end))
    return tuple(parts)


def _route_minutes(route_matrix: RouteMatrix, origin: str, destination: str, *, strict: bool) -> float | None:
    try:
        return route_matrix.duration_minutes(origin, destination, strict=strict)
    except RouteMatrixError:
        return None


def _opening_window(stop: dict[str, Any]) -> tuple[int | None, int | None]:
    return (
        _parse_minutes(_first_nonempty(stop, "opening_start", "window_start", "start_time", "earliest_start", "open_time")),
        _parse_minutes(_first_nonempty(stop, "opening_end", "window_end", "latest_start", "close_time")),
    )


def _opening_window_source(stop: dict[str, Any]) -> str:
    start_key = _first_nonempty_key(
        stop,
        "opening_start",
        "window_start",
        "start_time",
        "earliest_start",
        "open_time",
    )
    end_key = _first_nonempty_key(
        stop,
        "opening_end",
        "window_end",
        "latest_start",
        "close_time",
    )
    if start_key is None and end_key is None:
        return "unavailable"
    return f"plan_stop_fields:{start_key or 'none'}:{end_key or 'none'}"


def _is_role_owned_constraint(constraint: Any) -> bool:
    return bool(
        constraint.scope in {ConstraintScope.STOP, ConstraintScope.POI}
        and constraint.relation.strip().lower()
        in {
            "role",
            "itinerary_role",
            "role_equals",
            "set_role",
            "attribute:role",
        }
    )


def _first_nonempty_key(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return key
    return None


def _time_window_constraint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    fields = (
        "schema_version",
        "earliest_arrival",
        "latest_departure",
        "early_arrival_policy",
        "latest_departure_semantics",
    )
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("time_window_constraint_invalid")
    earliest = value.get("earliest_arrival")
    latest = value.get("latest_departure")
    if (
        value.get("schema_version") != "stop-time-window-constraint-v1"
        or value.get("early_arrival_policy") != "wait_until_earliest_arrival"
        or value.get("latest_departure_semantics") != "departure_after_visit"
        or (earliest is None and latest is None)
        or not _valid_clock(earliest)
        or not _valid_clock(latest)
        or (
            earliest is not None
            and latest is not None
            and _parse_minutes(earliest) > _parse_minutes(latest)
        )
    ):
        raise ValueError("time_window_constraint_invalid")
    return {field: value[field] for field in fields}


def _valid_clock(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is not None
    )


def _visit_duration_state(stop: dict[str, Any], default: float) -> tuple[float, str]:
    for key in ("visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"):
        value = stop.get(key)
        if value is None or str(value).strip() == "":
            continue
        parsed = _coerce_float(value)
        if parsed is not None:
            return max(0.0, parsed), key
    return float(default), "configured_evaluator_default"


def _parse_minutes(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            hours, minutes = text.split(":", 1)
            return int(hours) * 60 + int(minutes[:2])
        return int(float(text))
    except Exception:
        return None


def _first_nonempty(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _first_float(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(record.get(key))
        if value is not None:
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None or str(value).strip() == "":
            return None
        number = float(value)
        return number if isfinite(number) else None
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "closed"}
