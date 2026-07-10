"""Independent final-plan evaluator for PlanArtifactV2 records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..plans import ConstraintScope, ConstraintStrength, active_owned_constraints
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixError, validate_route_matrix
from .certificate import EvaluationFinding, PlanEvaluationCertificate


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


class PlanEvaluator:
    """Recompute final-plan eligibility from artifacts and context evidence."""

    evaluator_version = "verify-001-plan-evaluator-v1"

    def __init__(
        self,
        *,
        route_matrix: RouteMatrix | None = None,
        planner_runs: Mapping[str, PlannerRun | dict[str, Any]] | None = None,
        config: PlanEvaluatorConfig | None = None,
    ) -> None:
        self.route_matrix = route_matrix
        self.planner_runs = dict(planner_runs or {})
        self.config = config or PlanEvaluatorConfig()

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
        self._check_lodging(plan, stop_lookup, day_sequences, failures)
        self._check_budget_weather_and_closures(plan, stop_lookup, failures, warnings)
        route_record = self._check_routes_and_schedule(plan, matrix, stop_lookup, day_sequences, failures, metrics)

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
        if len(sequence) != len(set(sequence)):
            failures.append(_failure("duplicate_visit", "Plan sequence contains duplicate stop visits.", "hard"))
        for stop_id in sequence:
            if stop_id not in stop_lookup:
                failures.append(_failure("sequence_stop_missing_record", f"Sequence stop {stop_id!r} lacks a stop record.", "artifact"))
            if stop_id not in plan.day_assignments:
                failures.append(_failure("day_assignment_missing", f"Sequence stop {stop_id!r} lacks a day assignment.", "hard"))
        selected_ids = set(stop_lookup)
        missing_from_sequence = selected_ids.difference(sequence)
        if missing_from_sequence:
            warnings.append(
                _warning(
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
            if constraint.scope == ConstraintScope.LODGING and constraint.target_id not in lodging_ids:
                code = "booked_lodging_missing" if constraint.strength == ConstraintStrength.BOOKED else "owned_lodging_missing"
                finding = _failure(code, f"Owned lodging constraint {constraint.constraint_id!r} is not satisfied.", "hard", (constraint.constraint_id,))
                (failures if hard else warnings).append(finding if hard else _warning(code, finding.message, "soft", finding.evidence_refs))

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
                failures.append(_failure("lodging_missing", f"Day {day} has stops but no lodging assignment.", "hard"))
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
            self._check_day_schedule(day, sequence, route_matrix, stop_lookup, failures)
        return report.to_record()

    def _check_day_schedule(
        self,
        day: int,
        sequence: tuple[str, ...],
        route_matrix: RouteMatrix,
        stop_lookup: dict[str, dict[str, Any]],
        failures: list[EvaluationFinding],
    ) -> None:
        route_sequence = _route_sequence(day, sequence, self.config)
        current_time = float(_parse_minutes(self.config.day_start_time) or 0)
        travel_minutes = 0.0
        visit_minutes = 0.0
        waiting_minutes = 0.0
        current_id = route_sequence[0] if route_sequence and route_sequence[0] not in sequence else None
        for stop_id in sequence:
            if current_id:
                leg_minutes = _route_minutes(route_matrix, current_id, stop_id, strict=self.config.strict_routes)
                if leg_minutes is None:
                    failures.append(_failure("route_leg_unavailable", f"Missing or invalid route leg {current_id!r}->{stop_id!r}.", "hard"))
                    leg_minutes = 0.0
                travel_minutes += leg_minutes
                current_time += leg_minutes
            stop = stop_lookup.get(stop_id, {"stop_id": stop_id})
            window_start, window_end = _opening_window(stop)
            if self.config.enforce_opening_windows and window_start is not None and current_time < window_start:
                waiting_minutes += float(window_start - current_time)
                current_time = float(window_start)
            if self.config.enforce_opening_windows and window_end is not None and current_time > window_end:
                failures.append(_failure("opening_window_missed", f"Stop {stop_id!r} is reached after its opening window.", "hard", (stop_id,)))
            duration = _visit_duration(stop, self.config.default_visit_minutes)
            visit_minutes += duration
            current_time += duration
            current_id = stop_id
        end_anchor = route_sequence[-1] if route_sequence and route_sequence[-1] not in sequence else None
        if end_anchor and current_id:
            leg_minutes = _route_minutes(route_matrix, current_id, end_anchor, strict=self.config.strict_routes)
            if leg_minutes is None:
                failures.append(_failure("route_leg_unavailable", f"Missing or invalid route leg {current_id!r}->{end_anchor!r}.", "hard"))
                leg_minutes = 0.0
            travel_minutes += leg_minutes
        total = travel_minutes + visit_minutes + waiting_minutes
        if total > self.config.max_day_minutes:
            failures.append(_failure("day_time_exceeded", f"Day {day} exceeds the configured time limit.", "hard", (str(day),)))


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
    grouped: dict[int, list[str]] = {}
    for stop_id in plan.sequence:
        day = _coerce_int(plan.day_assignments.get(str(stop_id)))
        if day is not None:
            grouped.setdefault(day, []).append(str(stop_id))
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


def _visit_duration(stop: dict[str, Any], default: float) -> float:
    for key in ("visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"):
        value = stop.get(key)
        if value is None or str(value).strip() == "":
            continue
        parsed = _coerce_float(value)
        if parsed is not None:
            return max(0.0, parsed)
    return float(default)


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
        if value is None or str(value).strip() == "":
            return None
        return float(value)
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
