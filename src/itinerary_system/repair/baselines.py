"""Honest deterministic baseline planners for the paired repair benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from ..plans import OwnershipPolicy, compute_plan_diff
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix
from .context import PlannerContextMode, context_burden, stop_utility
from .day_route_solver import DayRouteCandidate, DayRouteSolverConfig, solve_day_route_subproblem
from .master_model import RepairModel, RepairSolution, build_repair_master_model
from .neighborhood import RepairRadius, build_repair_neighborhood

DETERMINISTIC_CONTEXT_AWARE_HEURISTIC = "deterministic_context_aware_heuristic"


@dataclass(frozen=True)
class BaselineDecisionRecord:
    """One deterministic candidate considered by a baseline method."""

    candidate_id: str
    day: int
    stop_sequence: tuple[str, ...]
    feasible: bool
    contextual_burden: float | None
    utility: float | None
    travel_minutes: float
    selected: bool = False
    violations: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselinePlanningResult:
    """Planner-owned result; independent evaluation happens in the pipeline."""

    method_id: str
    status: str
    parent_plan_id: str
    request_id: str
    planner_run: PlannerRun
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    decision_records: tuple[BaselineDecisionRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = "baseline-planning-result-v1"

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
            "schema_version": self.schema_version,
        }


def plan_deterministic_context_aware_heuristic(
    parent_plan: PlanArtifactV2,
    request: Any,
    route_matrix: RouteMatrix,
    *,
    day_route_config: DayRouteSolverConfig | None = None,
    ownership_policy: OwnershipPolicy | None = None,
    publication_mode: bool = False,
) -> BaselinePlanningResult:
    """Apply a one-pass, affected-day-only, deterministic context-aware repair."""

    method_id = DETERMINISTIC_CONTEXT_AWARE_HEURISTIC
    request_id = _request_id(request)
    run_id = f"{request_id}:{method_id}"
    config = day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode)
    policy = ownership_policy or OwnershipPolicy()
    failure_reasons: list[str] = []
    decision_records: list[BaselineDecisionRecord] = []
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    try:
        neighborhood = build_repair_neighborhood(parent_plan, request, RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(
            parent_plan,
            request,
            neighborhood,
            route_matrix,
            ownership_policy=policy,
            publication_mode=publication_mode,
            planner_context_mode=PlannerContextMode.AWARE,
        )
        selected_by_day: dict[int, DayRouteCandidate] = {}
        for day in neighborhood.editable_days:
            result = solve_day_route_subproblem(model, route_matrix, day=day, config=config)
            feasible = tuple(candidate for candidate in result.candidates if candidate.feasible)
            scored = sorted(
                ((_heuristic_score(model, candidate), candidate) for candidate in feasible),
                key=lambda item: item[0],
            )
            selected = scored[0][1] if scored else None
            if selected is None:
                failure_reasons.append(f"no_feasible_local_candidate:day:{day}")
            else:
                selected_by_day[day] = selected
            for candidate in result.candidates:
                score = _heuristic_score(model, candidate) if candidate.feasible else None
                decision_records.append(
                    BaselineDecisionRecord(
                        candidate_id=str(candidate.repair_solution.metadata.get("candidate_id", "")),
                        day=day,
                        stop_sequence=candidate.stop_sequence,
                        feasible=candidate.feasible,
                        contextual_burden=score[0] if score is not None else None,
                        utility=-score[1] if score is not None else None,
                        travel_minutes=candidate.travel_minutes,
                        selected=candidate is selected,
                        violations=candidate.violations,
                    )
                )
        if not failure_reasons:
            solution = _merge_day_candidates(model, selected_by_day, method_id=method_id)
            violations = model.validate_solution(solution)
            if violations:
                failure_reasons.extend(violations)
            else:
                child_id = f"plan_{stable_content_hash({'method': method_id, 'parent': parent_plan.plan_id, 'request': request_id, 'solution': solution.metadata})}"
                extracted = model.extract_plan(solution, child_plan_id=child_id)
                child_plan = replace(extracted, source_run_id=run_id, created_at=parent_plan.created_at)
                diff_record = compute_plan_diff(parent_plan, child_plan, policy).to_record()
    except Exception as exc:
        failure_reasons.append(f"{type(exc).__name__}:{exc}")
    status = "completed" if child_plan is not None else "failed"
    planner_run = PlannerRun(
        run_id=run_id,
        planning_request_id=request_id,
        catalog_snapshot_id=parent_plan.catalog_snapshot_id,
        context_snapshot_id=parent_plan.context_snapshot_id,
        planner_specification_id="deterministic-context-aware-heuristic-v1",
        method_requested=method_id,
        method_executed=method_id,
        execution_status="COMPLETED" if child_plan is not None else "FAILED",
        solver_certification="HEURISTIC_ONLY",
        result_plan_id=child_plan.plan_id if child_plan is not None else None,
    )
    return BaselinePlanningResult(
        method_id=method_id,
        status=status,
        parent_plan_id=parent_plan.plan_id,
        request_id=request_id,
        planner_run=planner_run,
        child_plan=child_plan,
        diff_record=diff_record,
        decision_records=tuple(decision_records),
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )


def _heuristic_score(model: RepairModel, candidate: DayRouteCandidate) -> tuple[float, float, float, str]:
    lookup = _stop_lookup(model)
    burden = sum(context_burden(lookup.get(stop_id, {})) for stop_id in candidate.stop_sequence)
    utility = sum(stop_utility(lookup.get(stop_id, {})) for stop_id in candidate.stop_sequence)
    stable_id = str(candidate.repair_solution.metadata.get("candidate_id", ""))
    return float(burden), -float(utility), float(candidate.travel_minutes), stable_id


def _merge_day_candidates(
    model: RepairModel,
    selected_by_day: dict[int, DayRouteCandidate],
    *,
    method_id: str,
) -> RepairSolution:
    parent = model.metadata["parent_plan"]
    parent_day_by_stop = model.metadata.get("parent_day_by_stop", {})
    parent_days = sorted(set(int(day) for day in parent_day_by_stop.values()) | set(selected_by_day))
    day_sequences: dict[str, tuple[str, ...]] = {}
    selected_stop_ids: list[str] = []
    day_assignments: dict[str, int] = {}
    route_ids = {int(day): str(route_id) for day, route_id in parent.route_ids_by_day.items()}
    selected_candidate_ids: list[str] = []
    for day in parent_days:
        candidate = selected_by_day.get(day)
        if candidate is None:
            sequence = tuple(
                str(stop_id)
                for stop_id in parent.sequence
                if int(parent_day_by_stop.get(str(stop_id), -1)) == day
            )
        else:
            sequence = candidate.stop_sequence
            route_ids.update({int(key): str(value) for key, value in candidate.repair_solution.route_ids_by_day.items()})
            selected_candidate_ids.append(str(candidate.repair_solution.metadata.get("candidate_id", "")))
        day_sequences[str(day)] = sequence
        for stop_id in sequence:
            if stop_id not in selected_stop_ids:
                selected_stop_ids.append(stop_id)
            day_assignments[str(stop_id)] = day
    return RepairSolution(
        selected_stop_ids=tuple(selected_stop_ids),
        day_assignments=day_assignments,
        lodging_assignments={str(day): str(value) for day, value in parent.lodging_assignments.items()},
        route_ids_by_day=route_ids,
        metadata={
            "candidate_id": f"{method_id}:{stable_content_hash(selected_candidate_ids)}",
            "day_sequences": day_sequences,
            "selected_candidate_ids": tuple(selected_candidate_ids),
        },
    )


def _stop_lookup(model: RepairModel) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in (*model.metadata.get("parent_selected_stops", ()), *model.metadata.get("candidate_stops", ())):
        stop = dict(record)
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name") or "").strip()
        if stop_id:
            lookup[stop_id] = stop
    return lookup


def _request_id(request: Any) -> str:
    return str(getattr(request, "request_id", "") or getattr(request, "repair_request_id", "") or "repair_request")
