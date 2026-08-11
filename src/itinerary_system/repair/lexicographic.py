"""Sequential lexicographic repair solver over candidate assignments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..research_artifacts import PlannerRun, stable_content_hash
from .change_variables import ObjectiveComponent
from .master_model import RepairModel, RepairSolution

StageStatus = Literal["OPTIMAL", "INFEASIBLE", "FAILED"]
SolveStatus = Literal["OPTIMAL", "INFEASIBLE", "FAILED"]


@dataclass(frozen=True)
class ObjectiveTolerances:
    """Absolute stage tolerances used when fixing previous optima."""

    default_fixed_tolerance: float = 0.0
    stage_tolerances: dict[str, float] = field(default_factory=dict)

    def tolerance_for(self, component: ObjectiveComponent | str) -> float:
        name = component.name if isinstance(component, ObjectiveComponent) else str(component)
        return float(self.stage_tolerances.get(name, self.default_fixed_tolerance))


@dataclass(frozen=True)
class LexicographicStageResult:
    stage: str
    status: StageStatus
    objective_value: float | None
    best_bound: float | None
    optimality_gap: float | None
    fixed_tolerance: float
    candidate_count: int = 0
    selected_candidate_id: str = ""
    fixed_limit: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LexicographicResult:
    model_id: str
    request_id: str
    status: SolveStatus
    stage_results: tuple[LexicographicStageResult, ...]
    selected_solution: RepairSolution | None = None
    selected_components: dict[str, float] = field(default_factory=dict)
    rejected_candidate_count: int = 0
    invalid_candidate_count: int = 0
    solver_backend: str = "enumeration"
    fallback_used: bool = False
    message: str = ""
    repair_model: RepairModel | None = field(default=None, repr=False, compare=False)

    @property
    def accepted(self) -> bool:
        return self.status == "OPTIMAL" and self.selected_solution is not None

    def to_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "request_id": self.request_id,
            "status": self.status,
            "stage_results": [stage.to_record() for stage in self.stage_results],
            "selected_solution": _solution_record(self.selected_solution),
            "selected_components": self.selected_components,
            "rejected_candidate_count": self.rejected_candidate_count,
            "invalid_candidate_count": self.invalid_candidate_count,
            "solver_backend": self.solver_backend,
            "fallback_used": self.fallback_used,
            "message": self.message,
        }


class LexicographicRepairSolver:
    """Exact lexicographic selector over explicit candidate repair solutions."""

    def __init__(
        self,
        *,
        tolerances: ObjectiveTolerances | None = None,
        solver_backend: str = "enumeration",
    ) -> None:
        self.tolerances = tolerances or ObjectiveTolerances()
        self.solver_backend = solver_backend

    def solve(
        self,
        model: RepairModel,
        candidate_solutions: tuple[RepairSolution, ...] | list[RepairSolution] | None = None,
    ) -> LexicographicResult:
        return solve_lexicographically(
            model,
            self.tolerances,
            candidate_solutions=tuple(candidate_solutions or ()),
            solver_backend=self.solver_backend,
        )

    def to_planner_runs(self, result: LexicographicResult) -> tuple[PlannerRun, ...]:
        return planner_runs_from_lexicographic_result(result)


def solve_lexicographically(
    model: RepairModel,
    tolerances: ObjectiveTolerances | None = None,
    *,
    candidate_solutions: tuple[RepairSolution, ...] | list[RepairSolution] | None = None,
    solver_backend: str = "enumeration",
) -> LexicographicResult:
    """Solve model objective stages exactly over candidate solutions."""

    tolerances = tolerances or ObjectiveTolerances()
    candidates = _candidate_pool(model, tuple(candidate_solutions or ()))
    valid_candidates: list[RepairSolution] = []
    invalid_count = 0
    for candidate in candidates:
        if model.validate_solution(candidate):
            invalid_count += 1
        else:
            valid_candidates.append(candidate)
    if not valid_candidates:
        failed_stage = _failed_stage(model, tolerances, "no feasible candidate satisfies fixed repair constraints")
        return LexicographicResult(
            model_id=model.model_id,
            request_id=model.request_id,
            status="INFEASIBLE",
            stage_results=(failed_stage,),
            rejected_candidate_count=len(candidates),
            invalid_candidate_count=invalid_count,
            solver_backend=solver_backend,
            message="no feasible candidate satisfies fixed repair constraints",
            repair_model=model,
        )

    active_candidates = tuple(valid_candidates)
    stage_results: list[LexicographicStageResult] = []
    for component in sorted(model.objective_components, key=lambda item: item.priority):
        stage_values = [(candidate, component.evaluate(model.variable_values(candidate))) for candidate in active_candidates]
        selected_solution, objective_value = _best_candidate(stage_values, component.sense)
        tolerance = tolerances.tolerance_for(component)
        fixed_limit = _fixed_limit(objective_value, tolerance, component.sense)
        active_candidates = tuple(
            candidate
            for candidate, value in stage_values
            if _within_tolerance(value, objective_value, tolerance, component.sense)
        )
        if not active_candidates:
            stage_results.append(
                LexicographicStageResult(
                    stage=component.name,
                    status="INFEASIBLE",
                    objective_value=objective_value,
                    best_bound=objective_value,
                    optimality_gap=None,
                    fixed_tolerance=tolerance,
                    candidate_count=0,
                    selected_candidate_id=_candidate_id(selected_solution),
                    fixed_limit=fixed_limit,
                )
            )
            return LexicographicResult(
                model_id=model.model_id,
                request_id=model.request_id,
                status="INFEASIBLE",
                stage_results=tuple(stage_results),
                rejected_candidate_count=len(candidates) - len(active_candidates),
                invalid_candidate_count=invalid_count,
                solver_backend=solver_backend,
                message=f"stage {component.name} eliminated all candidates",
                repair_model=model,
            )
        stage_results.append(
            LexicographicStageResult(
                stage=component.name,
                status="OPTIMAL",
                objective_value=objective_value,
                best_bound=objective_value,
                optimality_gap=0.0,
                fixed_tolerance=tolerance,
                candidate_count=len(active_candidates),
                selected_candidate_id=_candidate_id(selected_solution),
                fixed_limit=fixed_limit,
            )
        )

    final_solution = active_candidates[0]
    return LexicographicResult(
        model_id=model.model_id,
        request_id=model.request_id,
        status="OPTIMAL",
        stage_results=tuple(stage_results),
        selected_solution=final_solution,
        selected_components=model.export_components(final_solution),
        rejected_candidate_count=len(candidates) - len(active_candidates),
        invalid_candidate_count=invalid_count,
        solver_backend=solver_backend,
        repair_model=model,
    )


def extract_child_plan(parent: Any, result: LexicographicResult):
    """Extract the selected child plan from a lexicographic result."""

    if result.repair_model is None or result.selected_solution is None:
        return None
    return result.repair_model.extract_plan(result.selected_solution)


def planner_runs_from_lexicographic_result(result: LexicographicResult) -> tuple[PlannerRun, ...]:
    """Create one PlannerRun record per lexicographic stage."""

    model = result.repair_model
    catalog_snapshot_id = str(model.metadata.get("catalog_snapshot_id", "")) if model is not None else ""
    context_snapshot_id = str(model.metadata.get("context_snapshot_id", "")) if model is not None else ""
    runs: list[PlannerRun] = []
    parent_run_id: str | None = None
    for index, stage in enumerate(result.stage_results):
        run_id = f"run_{stable_content_hash({'model': result.model_id, 'stage': stage.stage, 'index': index})}"
        execution_status = "COMPLETED" if stage.status == "OPTIMAL" else "FAILED"
        runs.append(
            PlannerRun(
                run_id=run_id,
                parent_run_id=parent_run_id,
                planning_request_id=result.request_id,
                catalog_snapshot_id=catalog_snapshot_id,
                context_snapshot_id=context_snapshot_id,
                planner_specification_id="repair-003-lexicographic-v1",
                method_requested="lexicographic_repair",
                method_executed=f"lexicographic_repair:{result.solver_backend}",
                execution_status=execution_status,
                solver_certification="FEASIBILITY_CERTIFIED" if stage.status == "OPTIMAL" else "NO_CERTIFICATE",
                solver_backend=result.solver_backend,
                objective_value=stage.objective_value,
                best_bound=stage.best_bound,
                optimality_gap=stage.optimality_gap,
                result_plan_id=None,
                error_class="" if stage.status == "OPTIMAL" else "lexicographic_stage_infeasible",
                error_summary="" if stage.status == "OPTIMAL" else result.message or f"stage {stage.stage} failed",
            )
        )
        parent_run_id = run_id
    return tuple(runs)


def solve_gurobi_candidate_reference(
    model: RepairModel,
    candidate_solutions: tuple[RepairSolution, ...] | list[RepairSolution],
    tolerances: ObjectiveTolerances | None = None,
) -> LexicographicResult:
    """Solve the same candidate-choice problem with Gurobi for small reference tests."""

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - depends on local optional dependency
        raise RuntimeError("gurobipy is not available") from exc

    tolerances = tolerances or ObjectiveTolerances()
    candidates = tuple(candidate_solutions)
    valid_candidates = tuple(candidate for candidate in candidates if not model.validate_solution(candidate))
    if not valid_candidates:
        return solve_lexicographically(
            model,
            tolerances,
            candidate_solutions=candidates,
            solver_backend="gurobi_reference",
        )

    gp_model = gp.Model("repair_lexicographic_reference")
    gp_model.Params.OutputFlag = 0
    choice_vars = {
        index: gp_model.addVar(vtype=GRB.BINARY, name=f"choose_{index}")
        for index, _candidate in enumerate(valid_candidates)
    }
    gp_model.addConstr(gp.quicksum(choice_vars.values()) == 1, name="choose_one")
    stage_results: list[LexicographicStageResult] = []
    for component in sorted(model.objective_components, key=lambda item: item.priority):
        values = [component.evaluate(model.variable_values(candidate)) for candidate in valid_candidates]
        expression = gp.quicksum(values[index] * choice_vars[index] for index in range(len(valid_candidates)))
        gp_model.setObjective(expression, GRB.MINIMIZE if component.sense == "minimize" else GRB.MAXIMIZE)
        gp_model.optimize()
        if gp_model.Status != GRB.OPTIMAL:
            stage_results.append(
                LexicographicStageResult(
                    stage=component.name,
                    status="INFEASIBLE",
                    objective_value=None,
                    best_bound=None,
                    optimality_gap=None,
                    fixed_tolerance=tolerances.tolerance_for(component),
                )
            )
            return LexicographicResult(
                model_id=model.model_id,
                request_id=model.request_id,
                status="INFEASIBLE",
                stage_results=tuple(stage_results),
                solver_backend="gurobi_reference",
                message=f"gurobi status {gp_model.Status} at stage {component.name}",
                repair_model=model,
            )
        objective_value = float(gp_model.ObjVal)
        tolerance = tolerances.tolerance_for(component)
        fixed_limit = _fixed_limit(objective_value, tolerance, component.sense)
        if component.sense == "minimize":
            gp_model.addConstr(expression <= fixed_limit, name=f"fix_{component.name}")
        else:
            gp_model.addConstr(expression >= fixed_limit, name=f"fix_{component.name}")
        chosen_index = max(choice_vars, key=lambda index: choice_vars[index].X)
        stage_results.append(
            LexicographicStageResult(
                stage=component.name,
                status="OPTIMAL",
                objective_value=objective_value,
                best_bound=objective_value,
                optimality_gap=0.0,
                fixed_tolerance=tolerance,
                candidate_count=_candidate_count_within(values, objective_value, tolerance, component.sense),
                selected_candidate_id=_candidate_id(valid_candidates[chosen_index]),
                fixed_limit=fixed_limit,
            )
        )

    chosen_index = max(choice_vars, key=lambda index: choice_vars[index].X)
    selected = valid_candidates[chosen_index]
    return LexicographicResult(
        model_id=model.model_id,
        request_id=model.request_id,
        status="OPTIMAL",
        stage_results=tuple(stage_results),
        selected_solution=selected,
        selected_components=model.export_components(selected),
        rejected_candidate_count=len(candidates) - 1,
        invalid_candidate_count=len(candidates) - len(valid_candidates),
        solver_backend="gurobi_reference",
        repair_model=model,
    )


def _candidate_pool(model: RepairModel, supplied: tuple[RepairSolution, ...]) -> tuple[RepairSolution, ...]:
    candidates = supplied or (_parent_baseline_solution(model),)
    seen: set[str] = set()
    unique: list[RepairSolution] = []
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append(candidate)
    return tuple(unique)


def _parent_baseline_solution(model: RepairModel) -> RepairSolution:
    parent = model.metadata["parent_plan"]
    return RepairSolution(
        selected_stop_ids=tuple(model.metadata["ordered_stop_ids"][: len(parent.selected_stops)]),
        day_assignments={
            str(stop_id): int(day)
            for stop_id, day in model.metadata.get("parent_day_by_stop", {}).items()
        },
        lodging_assignments={str(day): str(lodging) for day, lodging in parent.lodging_assignments.items()},
        route_ids_by_day={int(day): str(route_id) for day, route_id in parent.route_ids_by_day.items()},
        metadata={"candidate_id": "parent_baseline"},
    )


def _best_candidate(
    stage_values: list[tuple[RepairSolution, float]],
    sense: str,
) -> tuple[RepairSolution, float]:
    if sense == "maximize":
        return max(stage_values, key=lambda item: (item[1], _candidate_id(item[0])))
    return min(stage_values, key=lambda item: (item[1], _candidate_id(item[0])))


def _fixed_limit(objective_value: float, tolerance: float, sense: str) -> float:
    return float(objective_value - tolerance if sense == "maximize" else objective_value + tolerance)


def _within_tolerance(value: float, objective_value: float, tolerance: float, sense: str) -> bool:
    if sense == "maximize":
        return value >= objective_value - tolerance
    return value <= objective_value + tolerance


def _candidate_count_within(values: list[float], objective_value: float, tolerance: float, sense: str) -> int:
    return sum(1 for value in values if _within_tolerance(value, objective_value, tolerance, sense))


def _failed_stage(model: RepairModel, tolerances: ObjectiveTolerances, message: str) -> LexicographicStageResult:
    component = sorted(model.objective_components, key=lambda item: item.priority)[0]
    return LexicographicStageResult(
        stage=component.name,
        status="INFEASIBLE",
        objective_value=None,
        best_bound=None,
        optimality_gap=None,
        fixed_tolerance=tolerances.tolerance_for(component),
        candidate_count=0,
        selected_candidate_id="",
        fixed_limit=None,
    )


def _candidate_id(solution: RepairSolution) -> str:
    explicit = str(solution.metadata.get("candidate_id", "")).strip()
    if explicit:
        return explicit
    return stable_content_hash(_solution_record(solution))


def _solution_record(solution: RepairSolution | None) -> dict[str, Any] | None:
    if solution is None:
        return None
    return {
        "selected_stop_ids": list(solution.selected_stop_ids),
        "day_assignments": solution.day_assignments,
        "lodging_assignments": solution.lodging_assignments,
        "relaxed_constraint_ids": list(solution.relaxed_constraint_ids),
        "route_ids_by_day": {str(day): route_id for day, route_id in solution.route_ids_by_day.items()},
        "metadata": solution.metadata,
    }
