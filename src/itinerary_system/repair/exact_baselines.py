"""Exact finite-universe context-blind and full-reoptimization baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations, product
from math import factorial, prod
from typing import Any

from ..plans import OwnershipPolicy, compute_plan_diff
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix
from .baselines import BaselineDecisionRecord
from .context import PlannerContextMode, context_burden, stop_utility
from .day_route_solver import DayRouteSolverConfig, _visit_duration, evaluate_route_sequence
from .master_model import RepairModel, RepairSolution, build_repair_master_model
from .neighborhood import RepairRadius, build_repair_neighborhood

CONTEXT_BLIND_SOLVER = "context_blind_solver"
FULL_REOPTIMIZATION = "full_reoptimization"


@dataclass(frozen=True)
class ExactBaselinePlanningResult:
    """Exact search result with explicit completeness evidence."""

    method_id: str
    status: str
    parent_plan_id: str
    request_id: str
    planner_run: PlannerRun
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    decision_records: tuple[BaselineDecisionRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    search_complete: bool = True
    candidate_count: int = 0
    candidate_space_lower_bound: int = 0
    schema_version: str = "exact-baseline-planning-result-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "status": self.status,
            "parent_plan_id": self.parent_plan_id,
            "request_id": self.request_id,
            "planner_run": self.planner_run.to_record(),
            "child_plan_id": self.child_plan.plan_id if self.child_plan is not None else None,
            "diff_id": self.diff_record.get("diff_id") if self.diff_record else None,
            "decision_records": [record.to_record() for record in self.decision_records],
            "failure_reasons": list(self.failure_reasons),
            "search_complete": self.search_complete,
            "candidate_count": self.candidate_count,
            "candidate_space_lower_bound": self.candidate_space_lower_bound,
            "schema_version": self.schema_version,
        }


def plan_context_blind_solver(
    parent_plan: PlanArtifactV2,
    request: Any,
    route_matrix: RouteMatrix,
    *,
    day_route_config: DayRouteSolverConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> ExactBaselinePlanningResult:
    """Optimize static utility/travel without consuming disruption context."""

    return _plan_exact_baseline(
        method_id=CONTEXT_BLIND_SOLVER,
        context_mode=PlannerContextMode.BLIND,
        parent_plan=parent_plan,
        request=request,
        route_matrix=route_matrix,
        day_route_config=day_route_config,
        publication_mode=publication_mode,
        max_complete_candidates=max_complete_candidates,
    )


def plan_full_reoptimization(
    parent_plan: PlanArtifactV2,
    request: Any,
    route_matrix: RouteMatrix,
    *,
    day_route_config: DayRouteSolverConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> ExactBaselinePlanningResult:
    """Optimize context burden then utility/travel without preservation objectives."""

    return _plan_exact_baseline(
        method_id=FULL_REOPTIMIZATION,
        context_mode=PlannerContextMode.AWARE,
        parent_plan=parent_plan,
        request=request,
        route_matrix=route_matrix,
        day_route_config=day_route_config,
        publication_mode=publication_mode,
        max_complete_candidates=max_complete_candidates,
    )


def _plan_exact_baseline(
    *,
    method_id: str,
    context_mode: PlannerContextMode,
    parent_plan: PlanArtifactV2,
    request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None,
    publication_mode: bool,
    max_complete_candidates: int,
) -> ExactBaselinePlanningResult:
    request_id = _request_id(request)
    run_id = f"{request_id}:{method_id}"
    config = day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode)
    failure_reasons: list[str] = []
    decisions: list[BaselineDecisionRecord] = []
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    search_complete = True
    candidate_count = 0
    candidate_space_lower_bound = 0
    best: tuple[tuple[Any, ...], RepairModel, RepairSolution] | None = None

    try:
        if max_complete_candidates < 1:
            raise ValueError("max_complete_candidates must be positive")
        neighborhood = build_repair_neighborhood(parent_plan, request, RepairRadius.FULL_REOPTIMIZATION)
        model = build_repair_master_model(
            parent_plan,
            request,
            neighborhood,
            route_matrix,
            publication_mode=publication_mode,
            planner_context_mode=context_mode,
        )
        stop_lookup = _stop_lookup(model)
        stop_records = tuple(stop_lookup.values())
        days = tuple(sorted(neighborhood.editable_days))
        selection_choices = _selection_day_choices(model, days)
        lodging_choices = _lodging_choices(model)
        stop_ids = tuple(selection_choices)
        lodging_days = tuple(sorted(lodging_choices))
        budget_limit = _budget_limit(request)

        assignment_count = prod(len(selection_choices[stop_id]) for stop_id in stop_ids)
        empty_assignment_count = int(
            bool(stop_ids)
            and all(None in selection_choices[stop_id] for stop_id in stop_ids)
        )
        lodging_count = prod(len(lodging_choices[day]) for day in lodging_days)
        candidate_space_lower_bound = max(0, assignment_count - empty_assignment_count) * lodging_count
        if candidate_space_lower_bound > max_complete_candidates:
            search_complete = False
            failure_reasons.extend(
                (
                    f"complete_candidate_limit_exceeded:{max_complete_candidates}",
                    f"candidate_space_lower_bound:{candidate_space_lower_bound}",
                )
            )

        assignment_options = (
            ()
            if not search_complete
            else product(*(selection_choices[stop_id] for stop_id in stop_ids))
        )
        for assignments in assignment_options:
            selected_assignment = {
                stop_id: int(day)
                for stop_id, day in zip(stop_ids, assignments, strict=True)
                if day is not None
            }
            if not selected_assignment:
                continue
            ids_by_day = {
                day: tuple(stop_id for stop_id in stop_ids if selected_assignment.get(stop_id) == day)
                for day in days
            }
            if budget_limit is not None:
                assignment_cost = sum(_stop_cost(stop_lookup.get(sid, {})) for sid in selected_assignment)
                if assignment_cost > budget_limit:
                    continue
            skip_assignment = False
            for day in days:
                day_stops = ids_by_day[day]
                if day_stops:
                    min_visit = sum(
                        _visit_duration(stop_lookup.get(sid, {}), config.default_visit_minutes)
                        for sid in day_stops
                    )
                    if min_visit > config.max_day_minutes:
                        skip_assignment = True
                        break
            if skip_assignment:
                continue
            lodging_options = tuple(lodging_choices[day] for day in lodging_days)
            assignment_candidate_count = prod(
                factorial(len(ids_by_day[day])) for day in days
            ) * prod(len(options) for options in lodging_options)
            if candidate_count + assignment_candidate_count > max_complete_candidates:
                search_complete = False
                failure_reasons.append(f"complete_candidate_limit_exceeded:{max_complete_candidates}")
                break
            sequence_options = tuple(
                tuple(permutations(ids_by_day[day])) if ids_by_day[day] else ((),)
                for day in days
            )
            for sequences in product(*sequence_options):
                for lodging_values in product(*lodging_options):
                    candidate_count += 1
                    if candidate_count > max_complete_candidates:
                        search_complete = False
                        failure_reasons.append(f"complete_candidate_limit_exceeded:{max_complete_candidates}")
                        break
                    day_sequences = {
                        str(day): tuple(sequence)
                        for day, sequence in zip(days, sequences, strict=True)
                    }
                    lodging_assignments = {
                        str(day): str(value)
                        for day, value in zip(lodging_days, lodging_values, strict=True)
                    }
                    route_ids = _route_ids_for_sequences(
                        model,
                        route_matrix,
                        day_sequences,
                        closed_route_ids=set(model.metadata.get("context_closed_route_ids", ())),
                    )
                    ordered_ids = tuple(
                        stop_id for day in days for stop_id in day_sequences[str(day)]
                    )
                    solution = RepairSolution(
                        selected_stop_ids=ordered_ids,
                        day_assignments=selected_assignment,
                        lodging_assignments=lodging_assignments,
                        route_ids_by_day=route_ids,
                        metadata={
                            "candidate_id": f"{method_id}:{stable_content_hash({'days': day_sequences, 'lodging': lodging_assignments})}",
                            "day_sequences": day_sequences,
                        },
                    )
                    violations = list(model.validate_solution(solution))
                    travel_minutes = 0.0
                    for day in days:
                        route_evaluation = evaluate_route_sequence(
                            route_matrix,
                            day=day,
                            stop_sequence=day_sequences[str(day)],
                            stop_records=stop_records,
                            config=config,
                        )
                        travel_minutes += route_evaluation.travel_minutes
                        violations.extend(route_evaluation.violations)
                        for stop_id in day_sequences[str(day)]:
                            stop_lodging = str(stop_lookup.get(stop_id, {}).get("lodging_id") or "")
                            if stop_lodging and stop_lodging != lodging_assignments.get(str(day), ""):
                                violations.append(f"lodging_mismatch:{stop_id}:day:{day}")
                    selected_records = tuple(stop_lookup[stop_id] for stop_id in ordered_ids)
                    total_cost = sum(_stop_cost(record) for record in selected_records)
                    if budget_limit is not None and total_cost > budget_limit:
                        violations.append("budget_exceeded")
                    burden = sum(context_burden(record) for record in selected_records)
                    utility = sum(stop_utility(record) for record in selected_records)
                    candidate_id = str(solution.metadata["candidate_id"])
                    score = _score(
                        context_mode=context_mode,
                        burden=burden,
                        utility=utility,
                        travel_minutes=travel_minutes,
                        total_cost=total_cost,
                        candidate_id=candidate_id,
                    )
                    feasible = not violations
                    decisions.append(
                        BaselineDecisionRecord(
                            candidate_id=candidate_id,
                            day=0,
                            stop_sequence=ordered_ids,
                            feasible=feasible,
                            contextual_burden=burden if context_mode == PlannerContextMode.AWARE else None,
                            utility=utility,
                            travel_minutes=travel_minutes,
                            violations=tuple(dict.fromkeys(violations)),
                        )
                    )
                    if feasible and (best is None or score < best[0]):
                        best = score, model, solution
                if not search_complete:
                    break
            if not search_complete:
                break

        if search_complete and best is not None:
            _, selected_model, selected_solution = best
            selected_id = str(selected_solution.metadata["candidate_id"])
            decisions = [replace(record, selected=record.candidate_id == selected_id) for record in decisions]
            child_id = f"plan_{stable_content_hash({'method': method_id, 'parent': parent_plan.plan_id, 'request': request_id, 'solution': selected_solution.metadata})}"
            extracted = selected_model.extract_plan(selected_solution, child_plan_id=child_id)
            child_plan = replace(extracted, source_run_id=run_id, created_at=parent_plan.created_at)
            diff_record = compute_plan_diff(parent_plan, child_plan, OwnershipPolicy()).to_record()
        elif search_complete:
            failure_reasons.append("no_feasible_complete_candidate")
    except Exception as exc:
        failure_reasons.append(f"{type(exc).__name__}:{exc}")

    status = "completed" if child_plan is not None and search_complete else "failed"
    planner_run = PlannerRun(
        run_id=run_id,
        planning_request_id=request_id,
        catalog_snapshot_id=parent_plan.catalog_snapshot_id,
        context_snapshot_id=parent_plan.context_snapshot_id,
        planner_specification_id=f"{method_id}-finite-universe-v1",
        method_requested=method_id,
        method_executed=method_id,
        execution_status="COMPLETED" if status == "completed" else "FAILED",
        solver_certification="OPTIMALITY_CERTIFIED" if status == "completed" else "NO_CERTIFICATE",
        solver_backend="exact_finite_enumeration",
        solver_status_raw="complete" if search_complete else "candidate_limit_exceeded",
        optimality_gap=0.0 if status == "completed" else None,
        result_plan_id=child_plan.plan_id if child_plan is not None else None,
        error_summary=";".join(dict.fromkeys(failure_reasons)),
    )
    return ExactBaselinePlanningResult(
        method_id=method_id,
        status=status,
        parent_plan_id=parent_plan.plan_id,
        request_id=request_id,
        planner_run=planner_run,
        child_plan=child_plan,
        diff_record=diff_record,
        decision_records=tuple(decisions),
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
        search_complete=search_complete,
        candidate_count=candidate_count,
        candidate_space_lower_bound=candidate_space_lower_bound,
    )


def _selection_day_choices(
    model: RepairModel,
    days: tuple[int, ...],
) -> dict[str, tuple[int | None, ...]]:
    day_variables = {variable.target_id: variable for variable in model.variables.day_assignment_variables}
    choices: dict[str, tuple[int | None, ...]] = {}
    for selection in model.variables.selection_variables:
        day_variable = day_variables.get(selection.target_id)
        if day_variable is None:
            continue
        if day_variable.fixed_value not in {None, 0, ""}:
            allowed_days = (int(day_variable.fixed_value),)
        else:
            allowed_days = tuple(
                sorted(
                    {
                        int(value)
                        for value in day_variable.allowed_values
                        if str(value).strip() and int(value) in days
                    }
                )
            )
        if selection.fixed_value == 0:
            choices[selection.target_id] = (None,)
        elif selection.fixed_value == 1:
            choices[selection.target_id] = allowed_days
        else:
            choices[selection.target_id] = (None, *allowed_days)
    return choices


def _lodging_choices(model: RepairModel) -> dict[int, tuple[str, ...]]:
    choices: dict[int, tuple[str, ...]] = {}
    for variable in model.variables.lodging_variables:
        day = int(variable.target_id)
        if variable.fixed_value not in {None, ""}:
            values = (str(variable.fixed_value),)
        else:
            values = tuple(sorted(str(value) for value in variable.allowed_values if str(value).strip()))
        # A plan with no lodging assignments has no lodging decision to enumerate.
        # Keeping an empty domain in ``product`` would yield zero candidates for
        # every stop assignment and can turn a bounded exact search into a scan
        # of the entire stop-assignment space.
        if values:
            choices[day] = values
    return choices


def _route_ids_for_sequences(
    model: RepairModel,
    route_matrix: RouteMatrix,
    day_sequences: dict[str, tuple[str, ...]],
    *,
    closed_route_ids: set[str],
) -> dict[int, str]:
    parent = model.metadata["parent_plan"]
    parent_day_by_stop = model.metadata.get("parent_day_by_stop", {})
    route_ids: dict[int, str] = {}
    for raw_day, sequence in day_sequences.items():
        day = int(raw_day)
        parent_sequence = tuple(
            str(stop_id)
            for stop_id in parent.sequence
            if int(parent_day_by_stop.get(str(stop_id), -1)) == day
        )
        parent_route = str(parent.route_ids_by_day.get(day) or "")
        if sequence == parent_sequence and parent_route not in closed_route_ids:
            route_ids[day] = parent_route
        else:
            route_ids[day] = f"route_matrix:{route_matrix.matrix_id}:day:{day}:{stable_content_hash(sequence)}"
    return route_ids


def _score(
    *,
    context_mode: PlannerContextMode,
    burden: float,
    utility: float,
    travel_minutes: float,
    total_cost: float,
    candidate_id: str,
) -> tuple[Any, ...]:
    if context_mode == PlannerContextMode.AWARE:
        return burden, -utility, travel_minutes, total_cost, candidate_id
    return -utility, travel_minutes, total_cost, candidate_id


def _stop_lookup(model: RepairModel) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in (*model.metadata.get("parent_selected_stops", ()), *model.metadata.get("candidate_stops", ())):
        stop = dict(record)
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name") or "").strip()
        if stop_id:
            lookup[stop_id] = stop
    return lookup


def _budget_limit(request: Any) -> float | None:
    constraints = getattr(request, "confirmed_constraints", {})
    tolerance = getattr(request, "tolerance_profile", {})
    values: list[Any] = []
    if isinstance(constraints, dict):
        values.extend(constraints.get(key) for key in ("budget_limit", "max_budget", "total_budget"))
    if isinstance(tolerance, dict):
        values.extend(tolerance.get(key) for key in ("budget_limit", "max_budget", "total_budget"))
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _stop_cost(record: dict[str, Any]) -> float:
    for key in ("estimated_cost", "cost", "price", "visit_cost"):
        value = record.get(key)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _request_id(request: Any) -> str:
    return str(getattr(request, "request_id", "") or getattr(request, "repair_request_id", "") or "repair_request")
