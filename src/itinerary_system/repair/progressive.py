"""Progressive repair controller over neighborhoods, candidates, and evaluator gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..plans import OwnershipPolicy, compute_plan_diff
from ..plans.repository import PlanRepository
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix
from .day_route_solver import DayRouteSolverConfig, DayRouteSubproblemResult, solve_day_route_subproblem
from .lexicographic import (
    LexicographicRepairSolver,
    ObjectiveTolerances,
    planner_runs_from_lexicographic_result,
)
from .master_model import RepairModel, RepairSolution, build_repair_master_model
from .neighborhood import REPAIR_RADIUS_ORDER, RepairRadius, build_repair_neighborhood


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    candidate_id: str
    child_plan_id: str | None
    eligible: bool
    evaluation_id: str
    evaluation_status: str
    comparison_eligibility: str
    hard_or_booked_relaxation_count: float | None = None
    weighted_edit_cost: float | None = None
    component_values: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairAttempt:
    attempt_id: str
    radius: RepairRadius
    status: str
    model_id: str | None = None
    candidate_count: int = 0
    feasible_candidate_count: int = 0
    selected_child_plan_id: str | None = None
    selected_diff_id: str | None = None
    selected_evaluation_id: str | None = None
    stage_records: tuple[dict[str, Any], ...] = ()
    planner_runs: tuple[PlannerRun, ...] = ()
    day_route_results: tuple[DayRouteSubproblemResult, ...] = ()
    candidate_evaluations: tuple[CandidateEvaluationRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "radius": self.radius.value,
            "status": self.status,
            "model_id": self.model_id,
            "candidate_count": self.candidate_count,
            "feasible_candidate_count": self.feasible_candidate_count,
            "selected_child_plan_id": self.selected_child_plan_id,
            "selected_diff_id": self.selected_diff_id,
            "selected_evaluation_id": self.selected_evaluation_id,
            "stage_records": list(self.stage_records),
            "planner_runs": [run.to_record() for run in self.planner_runs],
            "day_route_results": [result.to_record() for result in self.day_route_results],
            "candidate_evaluations": [record.to_record() for record in self.candidate_evaluations],
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class RepairDiagnosis:
    status: str
    attempted_radii: tuple[RepairRadius, ...]
    minimum_relaxation_radius: RepairRadius | None = None
    minimum_relaxation_count: float | None = None
    best_weighted_edit_cost: float | None = None
    failure_reasons: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempted_radii": [radius.value for radius in self.attempted_radii],
            "minimum_relaxation_radius": self.minimum_relaxation_radius.value if self.minimum_relaxation_radius else None,
            "minimum_relaxation_count": self.minimum_relaxation_count,
            "best_weighted_edit_cost": self.best_weighted_edit_cost,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class RepairOutcome:
    repair_outcome_id: str
    parent_plan_id: str
    repair_request_id: str
    status: str
    accepted_radius: RepairRadius | None
    attempts: tuple[RepairAttempt, ...]
    planner_runs: tuple[PlannerRun, ...]
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    evaluation_record: dict[str, Any] | None = None
    diagnosis: RepairDiagnosis | None = None

    @property
    def child_plan_id(self) -> str | None:
        return self.child_plan.plan_id if self.child_plan is not None else None

    def to_record(self) -> dict[str, Any]:
        return {
            "repair_outcome_id": self.repair_outcome_id,
            "parent_plan_id": self.parent_plan_id,
            "repair_request_id": self.repair_request_id,
            "status": self.status,
            "accepted_radius": self.accepted_radius.value if self.accepted_radius else None,
            "child_plan_id": self.child_plan_id,
            "diff_id": self.diff_record.get("diff_id") if self.diff_record else None,
            "evaluation_id": _evaluation_id(self.evaluation_record),
            "attempts": [attempt.to_record() for attempt in self.attempts],
            "planner_runs": [run.to_record() for run in self.planner_runs],
            "child_plan": self.child_plan.to_record() if self.child_plan is not None else None,
            "diff_record": self.diff_record,
            "evaluation_record": self.evaluation_record,
            "diagnosis": self.diagnosis.to_record() if self.diagnosis is not None else None,
        }


class ProgressiveRepairController:
    """Try progressive repair radii and stop at the smallest eligible child."""

    def __init__(
        self,
        *,
        plan_repository: PlanRepository,
        route_matrix: RouteMatrix,
        evaluator: Any,
        day_route_config: DayRouteSolverConfig | None = None,
        tolerances: ObjectiveTolerances | None = None,
        ownership_policy: OwnershipPolicy | None = None,
        publication_mode: bool = False,
    ) -> None:
        self.plan_repository = plan_repository
        self.route_matrix = route_matrix
        self.evaluator = evaluator
        self.day_route_config = day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode)
        self.tolerances = tolerances or ObjectiveTolerances()
        self.ownership_policy = ownership_policy or OwnershipPolicy()
        self.publication_mode = publication_mode

    def repair_progressively(self, request: Any) -> RepairOutcome:
        return repair_progressively(
            request,
            self.plan_repository,
            self.evaluator,
            route_matrix=self.route_matrix,
            day_route_config=self.day_route_config,
            tolerances=self.tolerances,
            ownership_policy=self.ownership_policy,
            publication_mode=self.publication_mode,
        )


def repair_progressively(
    request: Any,
    plan_repository: PlanRepository,
    evaluator: Any,
    *,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    tolerances: ObjectiveTolerances | None = None,
    ownership_policy: OwnershipPolicy | None = None,
    publication_mode: bool = False,
) -> RepairOutcome:
    if evaluator is None:
        raise ValueError("REPAIR-005 requires an independent evaluator")
    request_id = _request_id(request)
    parent_plan_id = _parent_plan_id(request)
    parent = plan_repository.load(parent_plan_id)
    radii = _ordered_allowed_radii(request)
    attempts: list[RepairAttempt] = []
    planner_runs: list[PlannerRun] = []
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    evaluation_record: dict[str, Any] | None = None
    accepted_radius: RepairRadius | None = None

    for radius in radii:
        attempt, accepted = _attempt_radius(
            parent=parent,
            request=request,
            radius=radius,
            route_matrix=route_matrix,
            evaluator=evaluator,
            plan_repository=plan_repository,
            day_route_config=day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode),
            tolerances=tolerances or ObjectiveTolerances(),
            ownership_policy=ownership_policy or OwnershipPolicy(),
            publication_mode=publication_mode,
        )
        attempts.append(attempt)
        planner_runs.extend(attempt.planner_runs)
        if accepted is not None:
            child_plan, diff_record, evaluation_record = accepted
            accepted_radius = radius
            break

    diagnosis = None
    status = "accepted" if child_plan is not None else "infeasible"
    if child_plan is None:
        diagnosis = diagnose_failure(tuple(attempts))
    outcome_seed = {
        "request_id": request_id,
        "parent_plan_id": parent_plan_id,
        "status": status,
        "accepted_radius": accepted_radius.value if accepted_radius else None,
        "attempts": [attempt.attempt_id for attempt in attempts],
    }
    return RepairOutcome(
        repair_outcome_id=f"repair_outcome_{stable_content_hash(outcome_seed)}",
        parent_plan_id=parent_plan_id,
        repair_request_id=request_id,
        status=status,
        accepted_radius=accepted_radius,
        attempts=tuple(attempts),
        planner_runs=tuple(planner_runs),
        child_plan=child_plan,
        diff_record=diff_record,
        evaluation_record=evaluation_record,
        diagnosis=diagnosis,
    )


def diagnose_failure(attempts: tuple[RepairAttempt, ...]) -> RepairDiagnosis:
    attempted_radii = tuple(attempt.radius for attempt in attempts)
    failure_reasons = tuple(dict.fromkeys(reason for attempt in attempts for reason in attempt.failure_reasons))
    best_radius: RepairRadius | None = None
    best_relaxation: float | None = None
    best_weighted_cost: float | None = None
    best_score: tuple[float, float, int] | None = None
    for attempt in attempts:
        for record in attempt.candidate_evaluations:
            relaxation = record.hard_or_booked_relaxation_count
            weighted_cost = record.weighted_edit_cost
            if relaxation is None and weighted_cost is None:
                continue
            relaxation_score = float(relaxation if relaxation is not None else 0.0)
            weighted_score = float(weighted_cost if weighted_cost is not None else 0.0)
            radius_index = _radius_order_index(attempt.radius)
            score = (relaxation_score, weighted_score, radius_index)
            if best_score is None or score < best_score:
                best_score = score
                best_radius = attempt.radius
                best_relaxation = relaxation_score
                best_weighted_cost = weighted_score
    return RepairDiagnosis(
        status="no_eligible_radius",
        attempted_radii=attempted_radii,
        minimum_relaxation_radius=best_radius,
        minimum_relaxation_count=best_relaxation,
        best_weighted_edit_cost=best_weighted_cost,
        failure_reasons=failure_reasons,
    )


def _attempt_radius(
    *,
    parent: PlanArtifactV2,
    request: Any,
    radius: RepairRadius,
    route_matrix: RouteMatrix,
    evaluator: Any,
    plan_repository: PlanRepository,
    day_route_config: DayRouteSolverConfig,
    tolerances: ObjectiveTolerances,
    ownership_policy: OwnershipPolicy,
    publication_mode: bool,
) -> tuple[RepairAttempt, tuple[PlanArtifactV2, dict[str, Any], dict[str, Any]] | None]:
    request_id = _request_id(request)
    failure_reasons: list[str] = []
    candidate_evaluations: list[CandidateEvaluationRecord] = []
    stage_records: list[dict[str, Any]] = []
    planner_runs: list[PlannerRun] = []
    day_results: list[DayRouteSubproblemResult] = []
    child_plan: PlanArtifactV2 | None = None
    diff_record: dict[str, Any] | None = None
    evaluation_record: dict[str, Any] | None = None
    selected_diff_id: str | None = None
    selected_evaluation_id: str | None = None
    model: RepairModel | None = None

    try:
        neighborhood = build_repair_neighborhood(parent, request, radius)
        model = build_repair_master_model(
            parent,
            request,
            neighborhood,
            route_matrix,
            ownership_policy=ownership_policy,
            publication_mode=publication_mode,
        )
        candidates = _generate_candidates(model, route_matrix, day_route_config)
        day_results = list(candidates[0])
        remaining = tuple(candidates[1])
        if not remaining:
            failure_reasons.append("no_feasible_day_route_candidates")
        solver = LexicographicRepairSolver(tolerances=tolerances)
        while remaining:
            result = solver.solve(model, remaining)
            stage_records.extend(stage.to_record() for stage in result.stage_results)
            planner_runs.extend(planner_runs_from_lexicographic_result(result))
            if not result.accepted or result.selected_solution is None:
                failure_reasons.append(result.message or "lexicographic_infeasible")
                break
            candidate_child = model.extract_plan(
                result.selected_solution,
                child_plan_id=_child_plan_id(parent, request_id, radius, result.selected_solution),
            )
            candidate_diff = compute_plan_diff(parent, candidate_child, ownership_policy).to_record()
            component_values = dict(result.selected_components or model.export_components(result.selected_solution))
            candidate_evaluation_record = _evaluate_child(evaluator, candidate_child, candidate_diff, result)
            candidate_evaluations.append(
                CandidateEvaluationRecord(
                    candidate_id=str(result.selected_solution.metadata.get("candidate_id", "")),
                    child_plan_id=candidate_child.plan_id,
                    eligible=_evaluation_eligible(candidate_evaluation_record),
                    evaluation_id=_evaluation_id(candidate_evaluation_record),
                    evaluation_status=str(candidate_evaluation_record.get("evaluation_status", "")),
                    comparison_eligibility=str(candidate_evaluation_record.get("comparison_eligibility", "")),
                    hard_or_booked_relaxation_count=component_values.get("hard_or_booked_relaxation_count"),
                    weighted_edit_cost=component_values.get("weighted_edit_cost"),
                    component_values=component_values,
                    reason=str(candidate_evaluation_record.get("reason", "")),
                )
            )
            if _evaluation_eligible(candidate_evaluation_record):
                plan_repository.save(candidate_child)
                child_plan = candidate_child
                diff_record = candidate_diff
                evaluation_record = candidate_evaluation_record
                selected_diff_id = str(candidate_diff.get("diff_id", ""))
                selected_evaluation_id = _evaluation_id(candidate_evaluation_record)
                break
            failure_reasons.append("evaluation_ineligible")
            selected_id = str(result.selected_solution.metadata.get("candidate_id", ""))
            remaining = tuple(
                candidate
                for candidate in remaining
                if str(candidate.metadata.get("candidate_id", "")) != selected_id
            )
    except Exception as exc:
        failure_reasons.append(f"{type(exc).__name__}:{exc}")

    status = "accepted" if child_plan is not None else "failed"
    attempt_seed = {
        "parent": parent.plan_id,
        "request": request_id,
        "radius": radius.value,
        "status": status,
        "model": model.model_id if model is not None else None,
    }
    attempt = RepairAttempt(
        attempt_id=f"repair_attempt_{stable_content_hash(attempt_seed)}",
        radius=radius,
        status=status,
        model_id=model.model_id if model is not None else None,
        candidate_count=sum(len(result.candidates) for result in day_results),
        feasible_candidate_count=sum(len(result.feasible_candidates) for result in day_results),
        selected_child_plan_id=child_plan.plan_id if child_plan is not None else None,
        selected_diff_id=selected_diff_id,
        selected_evaluation_id=selected_evaluation_id,
        stage_records=tuple(stage_records),
        planner_runs=tuple(planner_runs),
        day_route_results=tuple(day_results),
        candidate_evaluations=tuple(candidate_evaluations),
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )
    accepted = (child_plan, diff_record, evaluation_record) if child_plan is not None and diff_record is not None and evaluation_record is not None else None
    return attempt, accepted


def _generate_candidates(
    model: RepairModel,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig,
) -> tuple[tuple[DayRouteSubproblemResult, ...], tuple[RepairSolution, ...]]:
    day_results: list[DayRouteSubproblemResult] = []
    candidates: list[RepairSolution] = []
    target_days = _target_days(model)
    for day in target_days:
        result = solve_day_route_subproblem(model, route_matrix, day=day, config=day_route_config)
        day_results.append(result)
        candidates.extend(candidate.repair_solution for candidate in result.feasible_candidates)
    return tuple(day_results), _dedupe_solutions(tuple(candidates))


def _target_days(model: RepairModel) -> tuple[int, ...]:
    return model.neighborhood.editable_days


def _evaluate_child(
    evaluator: Any,
    child_plan: PlanArtifactV2,
    diff_record: dict[str, Any],
    result: LexicographicRepairSolver | Any,
) -> dict[str, Any]:
    if callable(evaluator):
        raw = evaluator(child_plan)
    elif hasattr(evaluator, "evaluate"):
        raw = evaluator.evaluate(child_plan)
    elif hasattr(evaluator, "evaluate_final_plan"):
        raw = evaluator.evaluate_final_plan(child_plan)
    else:
        raise TypeError("evaluator must be callable or expose evaluate()/evaluate_final_plan()")
    record = _record_from_evaluation(raw)
    record.setdefault("evaluation_id", f"eval_{stable_content_hash({'plan': child_plan.plan_id, 'diff': diff_record.get('diff_id')})}")
    record.setdefault("plan_id", child_plan.plan_id)
    record.setdefault("diff_id", diff_record.get("diff_id"))
    return record


def _record_from_evaluation(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bool):
        return {
            "comparison_eligibility": "eligible" if raw else "ineligible",
            "evaluation_status": "PASSED" if raw else "FAILED",
        }
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_record"):
        return dict(raw.to_record())
    record: dict[str, Any] = {}
    for key in ("evaluation_id", "comparison_eligibility", "evaluation_status", "status", "eligible", "reason"):
        if hasattr(raw, key):
            record[key] = getattr(raw, key)
    return record


def _evaluation_eligible(record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    if record.get("eligible") is True:
        return True
    if str(record.get("comparison_eligibility", "")).lower() == "eligible":
        return True
    if str(record.get("evaluation_status", "")).lower() in {"eligible", "accepted", "passed"}:
        return True
    return str(record.get("status", "")).lower() in {"eligible", "accepted", "passed"}


def _evaluation_id(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str(record.get("evaluation_id", ""))


def _ordered_allowed_radii(request: Any) -> tuple[RepairRadius, ...]:
    raw = getattr(request, "allowed_radii", None)
    if raw is None:
        confirmed_constraints = getattr(request, "confirmed_constraints", {})
        if isinstance(confirmed_constraints, dict):
            raw = confirmed_constraints.get("allowed_radii")
    if raw is None:
        raw = REPAIR_RADIUS_ORDER
    requested = [_coerce_radius(radius) for radius in raw]
    deduped: list[RepairRadius] = []
    for radius in requested:
        if radius not in deduped:
            deduped.append(radius)
    full = RepairRadius.FULL_REOPTIMIZATION
    without_full = [radius for radius in deduped if radius != full]
    return tuple((*without_full, full)) if full in deduped else tuple(without_full)


def _coerce_radius(radius: RepairRadius | str) -> RepairRadius:
    if isinstance(radius, RepairRadius):
        return radius
    return RepairRadius(str(radius))


def _radius_order_index(radius: RepairRadius) -> int:
    try:
        return REPAIR_RADIUS_ORDER.index(radius)
    except ValueError:
        return len(REPAIR_RADIUS_ORDER)


def _parent_plan_id(request: Any) -> str:
    plan_id = str(getattr(request, "parent_plan_id", "") or "").strip()
    if plan_id:
        return plan_id
    confirmed_constraints = getattr(request, "confirmed_constraints", {})
    if isinstance(confirmed_constraints, dict):
        plan_id = str(confirmed_constraints.get("parent_plan_id", "")).strip()
    if not plan_id:
        raise ValueError("repair request must provide parent_plan_id")
    return plan_id


def _request_id(request: Any) -> str:
    return str(getattr(request, "request_id", "") or getattr(request, "repair_request_id", "") or "repair_request")


def _child_plan_id(parent: PlanArtifactV2, request_id: str, radius: RepairRadius, solution: RepairSolution) -> str:
    return f"plan_{stable_content_hash({'parent': parent.plan_id, 'request': request_id, 'radius': radius.value, 'solution': solution.metadata})}"


def _dedupe_solutions(solutions: tuple[RepairSolution, ...]) -> tuple[RepairSolution, ...]:
    seen: set[str] = set()
    result: list[RepairSolution] = []
    for solution in solutions:
        key = stable_content_hash(
            {
                "selected_stop_ids": solution.selected_stop_ids,
                "day_assignments": solution.day_assignments,
                "lodging_assignments": solution.lodging_assignments,
                "route_ids_by_day": solution.route_ids_by_day,
                "metadata": solution.metadata,
            }
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(solution)
    return tuple(result)
