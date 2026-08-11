"""Authoritative package-level pipeline artifact runner.

This module is intentionally a thin orchestration boundary. Expensive
generation and repair logic is injected through an executor so the canonical
run-directory contract can stabilize before notebook and benchmark migration.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import TripConfig, load_trip_config
from .data import DatasetBundle
from .evaluation import PlanEvaluationCertificate, PlanEvaluator, PlanEvaluatorConfig
from .explanation import ContrastiveEvidence, EvidenceRecord, ExplanationClaim, WhyNotEvidence
from .phase0_exporter import write_phase0_research_artifacts
from .plans.repository import PlanRepository
from .repair.baselines import (
    DETERMINISTIC_CONTEXT_AWARE_HEURISTIC,
    plan_deterministic_context_aware_heuristic,
)
from .repair.context import context_evaluation_requirements, contextualize_plan_for_evaluation
from .repair.day_route_solver import DayRouteSolverConfig
from .repair.exact_baselines import (
    ExactBaselinePlanningResult,
    plan_context_blind_solver,
    plan_full_reoptimization,
)
from .repair.lexicographic import ObjectiveTolerances
from .repair.progressive import repair_progressively
from .research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from .routing import RouteMatrix


class RefreshPolicy(StrEnum):
    """Provider refresh policy for reproducible pipeline runs."""

    NEVER = "never"
    ALLOW_CACHED = "allow_cached"
    REFRESH_ALLOWED = "refresh_allowed"


class RunDirectoryExists(RuntimeError):
    """Raised when a pipeline run would overwrite an existing run directory."""


class PipelineExecutionMissing(RuntimeError):
    """Raised when the runner has no generation/repair executor."""


class PipelineStrictModeError(RuntimeError):
    """Raised after diagnostics are written for an ineligible strict run."""

    def __init__(self, message: str, *, pipeline_run: PipelineRun | None = None) -> None:
        super().__init__(message)
        self.pipeline_run = pipeline_run


@dataclass(frozen=True)
class PipelineRunContext:
    """Resolved inputs passed to a generation or repair executor."""

    run_id: str
    mode: str
    config: TripConfig
    catalog_snapshot_id: str
    context_snapshot_id: str
    output_dir: Path
    refresh_policy: RefreshPolicy
    parent_plan_id: str | None = None
    repair_request_id: str | None = None
    data_bundle: DatasetBundle | None = None


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Artifacts produced by an injected generation or repair executor."""

    planner_runs: tuple[PlannerRun | dict[str, Any], ...] = ()
    output_plans: tuple[PlanArtifactV2 | dict[str, Any], ...] = ()
    evaluations: tuple[PlanEvaluationCertificate | dict[str, Any], ...] = ()
    parent_plan: PlanArtifactV2 | dict[str, Any] | None = None
    diff_records: tuple[dict[str, Any], ...] = ()
    route_records: tuple[RouteMatrix | dict[str, Any], ...] = ()
    explanation_records: tuple[Any, ...] = ()
    request_records: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    dashboard_records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PipelineRun:
    """Completed or diagnostically blocked pipeline run."""

    run_id: str
    mode: str
    data_bundle: DatasetBundle | None
    parent_plan: PlanArtifactV2 | dict[str, Any] | None
    planner_runs: tuple[PlannerRun | dict[str, Any], ...]
    output_plans: tuple[PlanArtifactV2 | dict[str, Any], ...]
    evaluations: tuple[PlanEvaluationCertificate | dict[str, Any], ...]
    output_dir: Path
    status: str
    manifest_path: Path
    metrics_path: Path
    schema_version: str = "pipeline-run-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "planner_run_count": len(self.planner_runs),
            "output_plan_count": len(self.output_plans),
            "evaluation_count": len(self.evaluations),
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "schema_version": self.schema_version,
        }


PipelineExecutor = Callable[[PipelineRunContext], PipelineExecutionResult | Mapping[str, Any]]


def build_phase0_generation_executor(
    *,
    method_df: Any | None = None,
    route_stops_df: Any | None = None,
    legacy_subdir: str | Path = "phase0_legacy",
) -> PipelineExecutor:
    """Build a package-level executor for the existing Phase 0 generation artifact path."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_phase0_generation_executor(
            context,
            method_df=method_df,
            route_stops_df=route_stops_df,
            legacy_subdir=legacy_subdir,
        )

    return executor


def build_production_generation_executor(
    *,
    all_business_df: Any,
    hotels_df: Any,
    city_names: Iterable[str],
    production_runner: Callable[..., Mapping[str, Any]] | None = None,
    legacy_subdir: str | Path = "production_legacy",
    primary_city: str = "Santa Barbara",
) -> PipelineExecutor:
    """Build an executor that runs the existing production optimizer inside a pipeline run."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_production_generation_executor(
            context,
            all_business_df=all_business_df,
            hotels_df=hotels_df,
            city_names=city_names,
            production_runner=production_runner,
            legacy_subdir=legacy_subdir,
            primary_city=primary_city,
        )

    return executor


def run_production_generation_executor(
    context: PipelineRunContext,
    *,
    all_business_df: Any,
    hotels_df: Any,
    city_names: Iterable[str],
    production_runner: Callable[..., Mapping[str, Any]] | None = None,
    legacy_subdir: str | Path = "production_legacy",
    primary_city: str = "Santa Barbara",
) -> PipelineExecutionResult:
    """Run the production optimizer and adapt its method outputs to canonical artifacts."""

    legacy_path = Path(legacy_subdir)
    legacy_dir = (legacy_path if legacy_path.is_absolute() else context.output_dir / legacy_path).resolve()
    legacy_dir.mkdir(parents=True, exist_ok=True)
    runner = production_runner or _default_production_generation_runner
    raw_outputs = runner(
        all_business_df=all_business_df,
        hotels_df=hotels_df,
        city_names=list(city_names),
        output_dir=legacy_dir,
        config=context.config,
        primary_city=primary_city,
    )
    outputs = dict(raw_outputs or {})
    method_df = _production_output_frame(
        outputs,
        legacy_dir=legacy_dir,
        key="production_method_comparison_df",
        filename="production_method_comparison.csv",
    )
    route_stops_df = _production_output_frame(
        outputs,
        legacy_dir=legacy_dir,
        key="production_method_route_stops_df",
        filename="production_method_route_stops.csv",
    )
    phase0_result = run_phase0_generation_executor(
        context,
        method_df=method_df,
        route_stops_df=route_stops_df,
        legacy_subdir=legacy_dir,
    )
    return PipelineExecutionResult(
        planner_runs=phase0_result.planner_runs,
        output_plans=phase0_result.output_plans,
        evaluations=phase0_result.evaluations,
        parent_plan=phase0_result.parent_plan,
        diff_records=phase0_result.diff_records,
        route_records=phase0_result.route_records,
        explanation_records=phase0_result.explanation_records,
        request_records=phase0_result.request_records,
        metrics={
            **phase0_result.metrics,
            "production_legacy_artifact_dir": _relative_artifact_path(legacy_dir, context.output_dir),
            "production_method_row_count": _frame_len(method_df),
            "production_route_stop_row_count": _frame_len(route_stops_df),
        },
        dashboard_records=phase0_result.dashboard_records,
    )


def run_phase0_generation_executor(
    context: PipelineRunContext,
    *,
    method_df: Any | None = None,
    route_stops_df: Any | None = None,
    legacy_subdir: str | Path = "phase0_legacy",
) -> PipelineExecutionResult:
    """Run Phase 0 generation exports and return canonical pipeline artifacts."""

    legacy_path = Path(legacy_subdir)
    legacy_dir = (legacy_path if legacy_path.is_absolute() else context.output_dir / legacy_path).resolve()
    export_result = write_phase0_research_artifacts(
        output_dir=legacy_dir,
        config=context.config,
        method_df=method_df,
        route_stops_df=route_stops_df,
    )
    planner_records = _dataframe_records(export_result.get("planner_runs_df"))
    evaluation_records = _dataframe_records(export_result.get("evaluation_reports_df"))
    summary_records = _dataframe_records(export_result.get("evidence_summary_df"))
    route_records = _phase0_route_records(_dataframe_records(export_result.get("route_audit_df")))
    plan_records = _read_jsonl_records(legacy_dir / "production_phase0_plan_artifacts.jsonl")
    relative_legacy_dir = _relative_artifact_path(legacy_dir, context.output_dir)
    return PipelineExecutionResult(
        planner_runs=planner_records,
        output_plans=plan_records,
        evaluations=evaluation_records,
        route_records=route_records,
        request_records=_phase0_request_records(planner_records, context=context, legacy_artifact_dir=relative_legacy_dir),
        metrics={
            "phase0_legacy_artifact_dir": relative_legacy_dir,
            "phase0_plan_artifact_count": int(export_result.get("plan_artifact_count", len(plan_records))),
            "phase0_planner_run_count": len(planner_records),
            "phase0_evaluation_count": len(evaluation_records),
            "phase0_route_record_count": len(route_records),
        },
        dashboard_records=tuple(
            {
                **summary,
                "dashboard_id": _record_id(summary, "plan_id", "run_id", prefix=f"phase0_summary_{index}"),
                "legacy_artifact_dir": relative_legacy_dir,
                "schema_version": "phase0-dashboard-summary-v1",
            }
            for index, summary in enumerate(summary_records, start=1)
        ),
    )


def build_context_blind_solver_executor(
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> PipelineExecutor:
    """Build the exact context-blind finite-universe solver executor."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_context_blind_solver_executor(
            context,
            parent_plan=parent_plan,
            repair_request=repair_request,
            route_matrix=route_matrix,
            day_route_config=day_route_config,
            evaluator_config=evaluator_config,
            publication_mode=publication_mode,
            max_complete_candidates=max_complete_candidates,
        )

    return executor


def run_context_blind_solver_executor(
    context: PipelineRunContext,
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> PipelineExecutionResult:
    """Run static exact optimization and evaluate its output with the common context-aware evaluator."""

    config = day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode)
    result = plan_context_blind_solver(
        parent_plan,
        repair_request,
        route_matrix,
        day_route_config=config,
        publication_mode=publication_mode,
        max_complete_candidates=max_complete_candidates,
    )
    return _exact_baseline_pipeline_result(
        context,
        result=result,
        parent_plan=parent_plan,
        repair_request=repair_request,
        route_matrix=route_matrix,
        day_route_config=config,
        evaluator_config=evaluator_config,
    )


def build_full_reoptimization_executor(
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> PipelineExecutor:
    """Build exact context-aware full reoptimization without preservation objectives."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_full_reoptimization_executor(
            context,
            parent_plan=parent_plan,
            repair_request=repair_request,
            route_matrix=route_matrix,
            day_route_config=day_route_config,
            evaluator_config=evaluator_config,
            publication_mode=publication_mode,
            max_complete_candidates=max_complete_candidates,
        )

    return executor


def run_full_reoptimization_executor(
    context: PipelineRunContext,
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    publication_mode: bool = False,
    max_complete_candidates: int = 50_000,
) -> PipelineExecutionResult:
    """Run exact context-aware reoptimization over the complete declared finite universe."""

    config = _day_route_config_for_repair(
        day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode),
        repair_request,
    )
    result = plan_full_reoptimization(
        parent_plan,
        repair_request,
        route_matrix,
        day_route_config=config,
        publication_mode=publication_mode,
        max_complete_candidates=max_complete_candidates,
    )
    return _exact_baseline_pipeline_result(
        context,
        result=result,
        parent_plan=parent_plan,
        repair_request=repair_request,
        route_matrix=route_matrix,
        day_route_config=config,
        evaluator_config=evaluator_config,
    )


def _exact_baseline_pipeline_result(
    context: PipelineRunContext,
    *,
    result: ExactBaselinePlanningResult,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig,
    evaluator_config: PlanEvaluatorConfig | None,
) -> PipelineExecutionResult:
    request_id = _repair_request_id(repair_request)
    if context.parent_plan_id and context.parent_plan_id != parent_plan.plan_id:
        raise ValueError("pipeline parent_plan_id does not match exact baseline parent_plan")
    if context.repair_request_id and context.repair_request_id != request_id:
        raise ValueError("pipeline repair_request_id does not match exact baseline request")
    evaluations: tuple[PlanEvaluationCertificate | dict[str, Any], ...] = ()
    explanations: tuple[dict[str, Any], ...] = ()
    if result.child_plan is not None:
        resolved_evaluator_config = _evaluator_config_for_repair(
            evaluator_config or _evaluator_config_from_day_route_config(day_route_config, parent_plan=parent_plan),
            repair_request,
        )
        evaluator = PlanEvaluator(
            route_matrix=route_matrix,
            planner_runs={result.planner_run.run_id: result.planner_run},
            config=resolved_evaluator_config,
            reference_plan=contextualize_plan_for_evaluation(parent_plan, repair_request),
        )
        evaluations = (evaluator.evaluate_final_plan(result.child_plan, planner_run=result.planner_run),)
        selected_ids = [record.candidate_id for record in result.decision_records if record.selected]
        explanations = (
            {
                "evidence_id": f"explain_{result.child_plan.plan_id}",
                "evidence_type": "complete_finite_search_trace",
                "parent_plan_id": parent_plan.plan_id,
                "child_plan_id": result.child_plan.plan_id,
                "method_id": result.method_id,
                "search_complete": result.search_complete,
                "candidate_count": result.candidate_count,
                "claims": [
                    {
                        "claim_id": "selected_from_complete_finite_search",
                        "evidence_refs": selected_ids,
                    }
                ],
                "schema_version": "baseline-explanation-v1",
            },
        )
    return PipelineExecutionResult(
        planner_runs=(result.planner_run,),
        output_plans=(result.child_plan,) if result.child_plan is not None else (),
        evaluations=evaluations,
        parent_plan=parent_plan,
        diff_records=(result.diff_record,) if result.diff_record is not None else (),
        route_records=(route_matrix,),
        explanation_records=explanations,
        request_records=(_repair_request_record(repair_request, context=context),),
        metrics={
            "baseline_status": result.status,
            "baseline_search_complete": result.search_complete,
            "baseline_candidate_count": result.candidate_count,
            "baseline_feasible_candidate_count": sum(record.feasible for record in result.decision_records),
            "baseline_failure_count": len(result.failure_reasons),
        },
        dashboard_records=(
            {
                **result.to_record(),
                "dashboard_id": f"baseline_{stable_content_hash({'request': request_id, 'method': result.method_id})[:12]}",
            },
        ),
    )


def build_deterministic_context_aware_heuristic_executor(
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    ownership_policy: Any | None = None,
    publication_mode: bool = False,
) -> PipelineExecutor:
    """Build the deterministic affected-day context-aware heuristic executor."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_deterministic_context_aware_heuristic_executor(
            context,
            parent_plan=parent_plan,
            repair_request=repair_request,
            route_matrix=route_matrix,
            day_route_config=day_route_config,
            evaluator_config=evaluator_config,
            ownership_policy=ownership_policy,
            publication_mode=publication_mode,
        )

    return executor


def run_deterministic_context_aware_heuristic_executor(
    context: PipelineRunContext,
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    ownership_policy: Any | None = None,
    publication_mode: bool = False,
) -> PipelineExecutionResult:
    """Run the deterministic heuristic and emit canonical independently evaluated artifacts."""

    request_id = _repair_request_id(repair_request)
    if context.parent_plan_id and context.parent_plan_id != parent_plan.plan_id:
        raise ValueError("pipeline parent_plan_id does not match heuristic parent_plan")
    if context.repair_request_id and context.repair_request_id != request_id:
        raise ValueError("pipeline repair_request_id does not match heuristic request")
    resolved_day_config = _day_route_config_for_repair(
        day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode),
        repair_request,
    )
    result = plan_deterministic_context_aware_heuristic(
        parent_plan,
        repair_request,
        route_matrix,
        day_route_config=resolved_day_config,
        ownership_policy=ownership_policy,
        publication_mode=publication_mode,
    )
    evaluations: tuple[PlanEvaluationCertificate | dict[str, Any], ...] = ()
    explanations: tuple[dict[str, Any], ...] = ()
    if result.child_plan is not None:
        resolved_evaluator_config = _evaluator_config_for_repair(
            evaluator_config
            or _evaluator_config_from_day_route_config(resolved_day_config, parent_plan=parent_plan),
            repair_request,
        )
        evaluator = PlanEvaluator(
            route_matrix=route_matrix,
            planner_runs={result.planner_run.run_id: result.planner_run},
            config=resolved_evaluator_config,
            reference_plan=contextualize_plan_for_evaluation(parent_plan, repair_request),
        )
        certificate = evaluator.evaluate_final_plan(result.child_plan, planner_run=result.planner_run)
        evaluations = (certificate,)
        explanations = (
            {
                "evidence_id": f"explain_{result.child_plan.plan_id}",
                "evidence_type": "deterministic_candidate_trace",
                "parent_plan_id": parent_plan.plan_id,
                "child_plan_id": result.child_plan.plan_id,
                "method_id": DETERMINISTIC_CONTEXT_AWARE_HEURISTIC,
                "decision_record_ids": [record.candidate_id for record in result.decision_records],
                "claims": [
                    {
                        "claim_id": "selected_from_deterministic_trace",
                        "evidence_refs": [
                            record.candidate_id for record in result.decision_records if record.selected
                        ],
                    }
                ],
                "schema_version": "baseline-explanation-v1",
            },
        )
    return PipelineExecutionResult(
        planner_runs=(result.planner_run,),
        output_plans=(result.child_plan,) if result.child_plan is not None else (),
        evaluations=evaluations,
        parent_plan=parent_plan,
        diff_records=(result.diff_record,) if result.diff_record is not None else (),
        route_records=(route_matrix,),
        explanation_records=explanations,
        request_records=(_repair_request_record(repair_request, context=context),),
        metrics={
            "baseline_status": result.status,
            "baseline_candidate_count": len(result.decision_records),
            "baseline_feasible_candidate_count": sum(record.feasible for record in result.decision_records),
            "baseline_failure_count": len(result.failure_reasons),
        },
        dashboard_records=(
            {
                **result.to_record(),
                "dashboard_id": f"baseline_{stable_content_hash({'request': request_id, 'method': DETERMINISTIC_CONTEXT_AWARE_HEURISTIC})[:12]}",
            },
        ),
    )


def build_progressive_repair_executor(
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    evaluator: Any | None = None,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    tolerances: ObjectiveTolerances | None = None,
    ownership_policy: Any | None = None,
    publication_mode: bool = False,
    method_id: str = "progressive_sequential_lexicographic_repair",
    plan_repository: PlanRepository | None = None,
    repository_subdir: str | Path = "repair_workspace/plans",
) -> PipelineExecutor:
    """Build a package-level executor for the progressive repair controller."""

    def executor(context: PipelineRunContext) -> PipelineExecutionResult:
        return run_progressive_repair_executor(
            context,
            parent_plan=parent_plan,
            repair_request=repair_request,
            route_matrix=route_matrix,
            evaluator=evaluator,
            day_route_config=day_route_config,
            evaluator_config=evaluator_config,
            tolerances=tolerances,
            ownership_policy=ownership_policy,
            publication_mode=publication_mode,
            method_id=method_id,
            plan_repository=plan_repository,
            repository_subdir=repository_subdir,
        )

    return executor


def run_progressive_repair_executor(
    context: PipelineRunContext,
    *,
    parent_plan: PlanArtifactV2,
    repair_request: Any,
    route_matrix: RouteMatrix,
    evaluator: Any | None = None,
    day_route_config: DayRouteSolverConfig | None = None,
    evaluator_config: PlanEvaluatorConfig | None = None,
    tolerances: ObjectiveTolerances | None = None,
    ownership_policy: Any | None = None,
    publication_mode: bool = False,
    method_id: str = "progressive_sequential_lexicographic_repair",
    plan_repository: PlanRepository | None = None,
    repository_subdir: str | Path = "repair_workspace/plans",
) -> PipelineExecutionResult:
    """Run progressive repair and translate the outcome to canonical pipeline artifacts."""

    method_id = str(method_id).strip()
    if not method_id:
        raise ValueError("progressive repair method_id must not be empty")
    request_id = _repair_request_id(repair_request)
    if context.parent_plan_id and context.parent_plan_id != parent_plan.plan_id:
        raise ValueError("pipeline parent_plan_id does not match progressive repair parent_plan")
    if context.repair_request_id and context.repair_request_id != request_id:
        raise ValueError("pipeline repair_request_id does not match progressive repair request")
    repository = plan_repository or _run_plan_repository(context.output_dir, repository_subdir)
    repository.save(parent_plan)
    resolved_day_config = _day_route_config_for_repair(
        day_route_config or DayRouteSolverConfig(strict_route_matrix=publication_mode),
        repair_request,
    )
    default_evaluator = None
    resolved_evaluator = evaluator
    if resolved_evaluator is None:
        default_evaluator = _PipelineRepairEvaluator(
            route_matrix=route_matrix,
            config=_evaluator_config_for_repair(
                evaluator_config
            or _evaluator_config_from_day_route_config(resolved_day_config, parent_plan=parent_plan),
                repair_request,
            ),
            method_id=method_id,
            reference_plan=contextualize_plan_for_evaluation(parent_plan, repair_request),
        )
        resolved_evaluator = default_evaluator
    outcome = repair_progressively(
        repair_request,
        repository,
        resolved_evaluator,
        route_matrix=route_matrix,
        day_route_config=resolved_day_config,
        tolerances=tolerances,
        ownership_policy=ownership_policy,
        publication_mode=publication_mode,
    )
    planner_runs = list(outcome.planner_runs)
    if default_evaluator is not None and outcome.child_plan is not None:
        matching_run = default_evaluator.planner_run_for_plan(outcome.child_plan.plan_id)
        if matching_run is not None:
            planner_runs.append(matching_run)
    has_canonical_method_run = any(
        run.method_requested == method_id and run.method_executed == method_id
        for run in planner_runs
    )
    if not has_canonical_method_run:
        diagnosis = getattr(outcome, "diagnosis", None)
        failure_reasons = tuple(getattr(diagnosis, "failure_reasons", ()))
        planner_runs.append(
            PlannerRun(
                run_id=f"{request_id}:{method_id}",
                planning_request_id=request_id,
                catalog_snapshot_id=parent_plan.catalog_snapshot_id,
                context_snapshot_id=parent_plan.context_snapshot_id,
                planner_specification_id=f"{method_id}-v1",
                method_requested=method_id,
                method_executed=method_id,
                execution_status="COMPLETED" if outcome.child_plan is not None else "FAILED",
                solver_certification="HEURISTIC_ONLY" if outcome.child_plan is not None else "NO_CERTIFICATE",
                solver_backend="progressive_sequential_lexicographic_repair",
                solver_status_raw=str(getattr(outcome, "status", "failed")),
                result_plan_id=outcome.child_plan.plan_id if outcome.child_plan is not None else None,
                error_summary=";".join(str(reason) for reason in failure_reasons),
            )
        )
    diff_records = (outcome.diff_record,) if outcome.diff_record else ()
    evaluation_records = (outcome.evaluation_record,) if outcome.evaluation_record else ()
    explanation_records = _repair_explanation_records(outcome, route_matrix=route_matrix)
    dashboard_record = {**outcome.to_record(), "dashboard_id": outcome.repair_outcome_id, "schema_version": "repair-outcome-dashboard-v1"}
    return PipelineExecutionResult(
        planner_runs=tuple(planner_runs),
        output_plans=(outcome.child_plan,) if outcome.child_plan is not None else (),
        evaluations=evaluation_records,
        parent_plan=parent_plan,
        diff_records=diff_records,
        route_records=(route_matrix,),
        explanation_records=explanation_records,
        request_records=(_repair_request_record(repair_request, context=context),),
        metrics=_repair_metrics(outcome),
        dashboard_records=(dashboard_record,),
    )


def run_research_pipeline(
    *,
    config_path: str | Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan_id: str | None = None,
    repair_request_id: str | None = None,
    refresh_policy: RefreshPolicy | str = RefreshPolicy.NEVER,
    run_id: str | None = None,
    output_root: str | Path = "runs",
    executor: PipelineExecutor | None = None,
    strict: bool = True,
    config_overrides: dict[str, Any] | None = None,
    data_bundle: DatasetBundle | None = None,
) -> PipelineRun:
    """Run one generation/repair pipeline through an injected executor and export artifacts."""

    policy = _coerce_refresh_policy(refresh_policy)
    resolved_run_id = run_id or _default_run_id(
        config_path=config_path,
        catalog_snapshot_id=catalog_snapshot_id,
        context_snapshot_id=context_snapshot_id,
        parent_plan_id=parent_plan_id,
        repair_request_id=repair_request_id,
    )
    mode = "repair" if repair_request_id else "generation"
    output_dir = _create_run_directory(Path(output_root), resolved_run_id)
    _create_layout(output_dir)
    config = _load_pipeline_config(
        config_path,
        catalog_snapshot_id=catalog_snapshot_id,
        context_snapshot_id=context_snapshot_id,
        refresh_policy=policy,
        overrides=config_overrides,
    )
    _write_json(output_dir / "resolved_config.redacted.json", _redacted_config_record(config))
    _write_json(output_dir / "dataset_manifest.json", _dataset_manifest(data_bundle, catalog_snapshot_id, context_snapshot_id))

    context = PipelineRunContext(
        run_id=resolved_run_id,
        mode=mode,
        config=config,
        catalog_snapshot_id=catalog_snapshot_id,
        context_snapshot_id=context_snapshot_id,
        output_dir=output_dir,
        refresh_policy=policy,
        parent_plan_id=parent_plan_id,
        repair_request_id=repair_request_id,
        data_bundle=data_bundle,
    )
    try:
        if executor is None:
            raise PipelineExecutionMissing("run_research_pipeline requires an executor for generation or repair mode")
        result = _coerce_execution_result(executor(context))
    except Exception as exc:
        _write_failed_pipeline_run(
            context=context,
            data_bundle=data_bundle,
            parent_plan=None,
            error=exc,
        )
        raise

    artifact_paths = _write_execution_artifacts(output_dir, result)
    strict_failures = _strict_failure_count(result.evaluations)
    planner_failures = _planner_failure_count(result.planner_runs)
    repair_output_missing = mode == "repair" and not result.output_plans
    status = "completed"
    if repair_output_missing:
        status = "failed"
    elif strict_failures:
        status = "failed_strict" if strict else "completed_with_warnings"
    metrics_path = _write_json(
        output_dir / "metrics" / "metrics.json",
        {
            "run_id": resolved_run_id,
            "mode": mode,
            "strict": strict,
            "strict_failure_count": strict_failures,
            "planner_failure_count": planner_failures,
            "repair_output_missing": repair_output_missing,
            **result.metrics,
        },
    )
    manifest_path = _write_manifest(
        output_dir=output_dir,
        context=context,
        result=result,
        status=status,
        strict=strict,
        strict_failures=strict_failures,
        planner_failures=planner_failures,
        artifact_paths={**artifact_paths, "metrics": [metrics_path]},
    )
    pipeline_run = PipelineRun(
        run_id=resolved_run_id,
        mode=mode,
        data_bundle=data_bundle,
        parent_plan=result.parent_plan,
        planner_runs=result.planner_runs,
        output_plans=result.output_plans,
        evaluations=result.evaluations,
        output_dir=output_dir,
        status=status,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
    )
    if strict and strict_failures:
        raise PipelineStrictModeError(
            f"strict pipeline run blocked {strict_failures} ineligible evaluation(s)",
            pipeline_run=pipeline_run,
        )
    return pipeline_run


def _write_failed_pipeline_run(
    *,
    context: PipelineRunContext,
    data_bundle: DatasetBundle | None,
    parent_plan: PlanArtifactV2 | dict[str, Any] | None,
    error: Exception,
) -> PipelineRun:
    metrics_path = _write_json(
        context.output_dir / "metrics" / "metrics.json",
        {"run_id": context.run_id, "mode": context.mode, "status": "failed", "error_class": type(error).__name__},
    )
    manifest_path = _write_json(
        context.output_dir / "manifest.json",
        {
            "run_id": context.run_id,
            "mode": context.mode,
            "status": "failed",
            "error_class": type(error).__name__,
            "error_summary": str(error),
            "schema_version": "pipeline-manifest-v1",
        },
    )
    return PipelineRun(
        run_id=context.run_id,
        mode=context.mode,
        data_bundle=data_bundle,
        parent_plan=parent_plan,
        planner_runs=(),
        output_plans=(),
        evaluations=(),
        output_dir=context.output_dir,
        status="failed",
        manifest_path=manifest_path,
        metrics_path=metrics_path,
    )


def _write_execution_artifacts(output_dir: Path, result: PipelineExecutionResult) -> dict[str, list[Path]]:
    artifacts: dict[str, list[Path]] = {
        "requests": [],
        "planner_runs": [],
        "plans": [],
        "diffs": [],
        "routing": [],
        "evaluations": [],
        "explanations": [],
        "dashboard": [],
    }
    for request in result.request_records:
        request_id = _record_id(request, "request_id", prefix="request")
        artifacts["requests"].append(_write_json(output_dir / "requests" / f"{request_id}.json", request))
    planner_records = [_record_from_artifact(run) for run in result.planner_runs]
    planner_path = output_dir / "planner_runs.jsonl"
    planner_path.write_text(
        "".join(json.dumps(_jsonish(record), sort_keys=True, default=str) + "\n" for record in planner_records),
        encoding="utf-8",
    )
    artifacts["planner_runs"].append(planner_path)
    plan_artifacts: list[PlanArtifactV2 | dict[str, Any]] = []
    if result.parent_plan is not None:
        plan_artifacts.append(result.parent_plan)
    plan_artifacts.extend(result.output_plans)
    seen_plan_ids: set[str] = set()
    for plan in plan_artifacts:
        record = _record_from_artifact(plan)
        plan_id = _record_id(record, "plan_id", prefix="plan")
        if plan_id in seen_plan_ids:
            continue
        seen_plan_ids.add(plan_id)
        artifacts["plans"].append(_write_json(output_dir / "plans" / f"{plan_id}.json", record))
    for diff in result.diff_records:
        diff_id = _record_id(diff, "diff_id", prefix="diff")
        artifacts["diffs"].append(_write_json(output_dir / "diffs" / f"{diff_id}.json", diff))
    for route in result.route_records:
        record = _record_from_artifact(route)
        route_id = _record_id(record, "matrix_id", "route_id", "route_result_id", prefix="route")
        artifacts["routing"].append(_write_json(output_dir / "routing" / f"{route_id}.json", record))
    for evaluation in result.evaluations:
        record = _record_from_artifact(evaluation)
        evaluation_id = _record_id(record, "certificate_id", "evaluation_id", prefix="evaluation")
        artifacts["evaluations"].append(_write_json(output_dir / "evaluations" / f"{evaluation_id}.json", record))
    for explanation in result.explanation_records:
        record = _record_from_artifact(explanation)
        explanation_id = _record_id(record, "evidence_id", "explanation_id", prefix="explanation")
        artifacts["explanations"].append(_write_json(output_dir / "explanations" / f"{explanation_id}.json", record))
    for index, dashboard_record in enumerate(result.dashboard_records, start=1):
        dashboard_id = _record_id(dashboard_record, "dashboard_id", prefix=f"dashboard_{index}")
        artifacts["dashboard"].append(_write_json(output_dir / "dashboard" / f"{dashboard_id}.json", dashboard_record))
    if not artifacts["dashboard"]:
        artifacts["dashboard"].append(
            _write_json(output_dir / "dashboard" / "manifest.json", {"status": "not_generated", "schema_version": "dashboard-manifest-v1"})
        )
    return artifacts


def _write_manifest(
    *,
    output_dir: Path,
    context: PipelineRunContext,
    result: PipelineExecutionResult,
    status: str,
    strict: bool,
    strict_failures: int,
    planner_failures: int,
    artifact_paths: dict[str, list[Path]],
) -> Path:
    relative_artifacts = {
        key: [path.relative_to(output_dir).as_posix() for path in paths] for key, paths in artifact_paths.items()
    }
    manifest = {
        "run_id": context.run_id,
        "mode": context.mode,
        "status": status,
        "strict": strict,
        "strict_failure_count": strict_failures,
        "planner_failure_count": planner_failures,
        "repair_output_missing": context.mode == "repair" and not result.output_plans,
        "catalog_snapshot_id": context.catalog_snapshot_id,
        "context_snapshot_id": context.context_snapshot_id,
        "parent_plan_id": context.parent_plan_id,
        "repair_request_id": context.repair_request_id,
        "refresh_policy": context.refresh_policy.value,
        "artifact_counts": {
            "requests": len(result.request_records),
            "planner_runs": len(result.planner_runs),
            "plans": len(result.output_plans),
            "evaluations": len(result.evaluations),
            "diffs": len(result.diff_records),
            "routing": len(result.route_records),
            "explanations": len(result.explanation_records),
        },
        "artifacts": relative_artifacts,
        "pipeline_run": {
            "output_plan_ids": [_record_id(_record_from_artifact(plan), "plan_id", prefix="plan") for plan in result.output_plans],
            "planner_run_ids": [_record_id(_record_from_artifact(run), "run_id", prefix="planner_run") for run in result.planner_runs],
            "evaluation_ids": [
                _record_id(_record_from_artifact(evaluation), "certificate_id", "evaluation_id", prefix="evaluation")
                for evaluation in result.evaluations
            ],
        },
        "schema_version": "pipeline-manifest-v1",
    }
    return _write_json(output_dir / "manifest.json", manifest)


def _create_run_directory(output_root: Path, run_id: str) -> Path:
    output_dir = output_root / run_id
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunDirectoryExists(f"run directory already exists: {output_dir}") from exc
    return output_dir


def _create_layout(output_dir: Path) -> None:
    for name in ("requests", "plans", "diffs", "routing", "evaluations", "explanations", "metrics", "dashboard"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _load_pipeline_config(
    config_path: str | Path,
    *,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    refresh_policy: RefreshPolicy,
    overrides: dict[str, Any] | None,
) -> TripConfig:
    pipeline_overrides = {
        "data": {
            "catalog_snapshot_id": catalog_snapshot_id,
            "context_snapshot_id": context_snapshot_id,
            "refresh_policy": refresh_policy.value,
        }
    }
    if refresh_policy == RefreshPolicy.NEVER:
        pipeline_overrides["enrichment"] = {
            "run_live_apis": False,
            "use_yelp_live_api": False,
            "use_nps_api": False,
        }
    merged_overrides = _deep_merge_dicts(pipeline_overrides, overrides or {})
    config = load_trip_config(config_path, overrides=merged_overrides)
    if refresh_policy == RefreshPolicy.NEVER and config.run_live_apis:
        config = TripConfig(data=_deep_merge_dicts(config.to_dict(), pipeline_overrides), source_path=config.source_path)
    return config


def _redacted_config_record(config: TripConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload["_source_path"] = config.source_path
    return _redact(payload)


def _dataset_manifest(
    data_bundle: DatasetBundle | None,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
) -> dict[str, Any]:
    if data_bundle is None:
        return {
            "catalog_snapshot_id": catalog_snapshot_id,
            "context_snapshot_id": context_snapshot_id,
            "file_hashes": {},
            "tables": {},
            "schema_version": "pipeline-dataset-manifest-v1",
        }
    return {
        "catalog_snapshot_id": data_bundle.catalog_snapshot_id,
        "context_snapshot_id": data_bundle.context_snapshot_id,
        "catalog_manifest": _jsonish(data_bundle.manifest),
        "context_manifest": _jsonish(data_bundle.context_manifest),
        "file_hashes": dict(data_bundle.file_hashes),
        "tables": {name: int(len(frame)) for name, frame in data_bundle.tables.items()},
        "schema_version": "pipeline-dataset-manifest-v1",
    }


def _strict_failure_count(evaluations: Iterable[PlanEvaluationCertificate | dict[str, Any]]) -> int:
    count = 0
    for evaluation in evaluations:
        record = _record_from_artifact(evaluation)
        eligible = record.get("eligible")
        comparison_eligibility = str(record.get("comparison_eligibility", "")).lower()
        evaluation_status = str(record.get("evaluation_status", "")).lower()
        if eligible is False or comparison_eligibility == "ineligible" or evaluation_status == "failed":
            count += 1
    return count


def _planner_failure_count(planner_runs: Iterable[PlannerRun | dict[str, Any]]) -> int:
    return sum(
        str(_record_from_artifact(planner_run).get("execution_status", "")).upper()
        in {"FAILED", "ERROR"}
        for planner_run in planner_runs
    )


def _coerce_execution_result(raw: PipelineExecutionResult | Mapping[str, Any]) -> PipelineExecutionResult:
    if isinstance(raw, PipelineExecutionResult):
        return raw
    if isinstance(raw, Mapping):
        return PipelineExecutionResult(
            planner_runs=tuple(raw.get("planner_runs", ())),
            output_plans=tuple(raw.get("output_plans", ())),
            evaluations=tuple(raw.get("evaluations", ())),
            parent_plan=raw.get("parent_plan"),
            diff_records=tuple(raw.get("diff_records", ())),
            route_records=tuple(raw.get("route_records", ())),
            explanation_records=tuple(raw.get("explanation_records", ())),
            request_records=tuple(raw.get("request_records", ())),
            metrics=dict(raw.get("metrics", {})),
            dashboard_records=tuple(raw.get("dashboard_records", ())),
        )
    raise TypeError("pipeline executor must return PipelineExecutionResult or a mapping")


def _dataframe_records(frame: Any) -> tuple[dict[str, Any], ...]:
    if frame is None or not hasattr(frame, "to_dict"):
        return ()
    return tuple(_phase0_jsonish(record) for record in frame.to_dict("records"))


def _default_production_generation_runner(**kwargs: Any) -> Mapping[str, Any]:
    from .experiment_runner import run_configurable_blueprint_pipeline

    return run_configurable_blueprint_pipeline(**kwargs)


def _production_output_frame(
    outputs: Mapping[str, Any],
    *,
    legacy_dir: Path,
    key: str,
    filename: str,
) -> Any:
    frame = outputs.get(key)
    if frame is not None:
        return frame
    csv_path = legacy_dir / filename
    if csv_path.exists():
        import pandas as pd

        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"production generation did not provide {key!r} or write {csv_path}")


def _frame_len(frame: Any) -> int:
    try:
        return int(len(frame))
    except TypeError:
        return 0


def _read_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(_phase0_jsonish(json.loads(line)))
    return tuple(records)


def _phase0_route_records(route_rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in route_rows:
        record = _phase0_jsonish(dict(row))
        route_id = str(record.get("route_id") or f"route_{record.get('plan_id') or stable_content_hash(record)}")
        grouped.setdefault(route_id, []).append(record)
    route_records: list[dict[str, Any]] = []
    for route_id, legs in grouped.items():
        first = legs[0] if legs else {}
        road_validated = bool(legs) and all(_truthy(leg.get("road_validated")) for leg in legs)
        fallback_used = any(_truthy(leg.get("fallback_used")) for leg in legs)
        route_records.append(
            {
                "route_id": route_id,
                "run_id": first.get("run_id"),
                "plan_id": first.get("plan_id"),
                "method": first.get("method"),
                "leg_count": len(legs),
                "road_validated": road_validated,
                "fallback_used": fallback_used,
                "route_evaluation_eligible": bool(legs)
                and all(_truthy(leg.get("route_evaluation_eligible")) for leg in legs),
                "legs": legs,
                "schema_version": "phase0-route-audit-v1",
            }
        )
    return tuple(route_records)


class _PipelineRepairEvaluator:
    """Independent evaluator wrapper that creates planner-run evidence for generated child plans."""

    def __init__(
        self,
        *,
        route_matrix: RouteMatrix,
        config: PlanEvaluatorConfig,
        method_id: str,
        reference_plan: PlanArtifactV2,
    ) -> None:
        self.route_matrix = route_matrix
        self.config = config
        self.method_id = method_id
        self.reference_plan = reference_plan
        self._planner_runs_by_plan_id: dict[str, PlannerRun] = {}

    def evaluate(self, child_plan: PlanArtifactV2) -> PlanEvaluationCertificate:
        planner_run = _planner_run_for_child_plan(child_plan, method_id=self.method_id)
        self._planner_runs_by_plan_id[child_plan.plan_id] = planner_run
        evaluator = PlanEvaluator(
            route_matrix=self.route_matrix,
            planner_runs={planner_run.run_id: planner_run},
            config=self.config,
            reference_plan=self.reference_plan,
        )
        return evaluator.evaluate_final_plan(child_plan, planner_run=planner_run)

    def planner_run_for_plan(self, plan_id: str) -> PlannerRun | None:
        return self._planner_runs_by_plan_id.get(plan_id)


def _run_plan_repository(output_dir: Path, repository_subdir: str | Path) -> PlanRepository:
    path = Path(repository_subdir)
    return PlanRepository(path if path.is_absolute() else output_dir / path)


def _evaluator_config_from_day_route_config(
    config: DayRouteSolverConfig,
    *,
    parent_plan: PlanArtifactV2,
) -> PlanEvaluatorConfig:
    return PlanEvaluatorConfig(
        strict_routes=config.strict_route_matrix,
        max_day_minutes=config.max_day_minutes,
        day_start_time=config.day_start_time,
        default_visit_minutes=config.default_visit_minutes,
        enforce_opening_windows=config.enforce_opening_windows,
        start_anchor_by_day=dict(config.start_anchor_by_day),
        end_anchor_by_day=dict(config.end_anchor_by_day),
        require_lodging_assignments=bool(parent_plan.lodging_assignments),
    )


def _day_route_config_for_repair(config: DayRouteSolverConfig, request: Any) -> DayRouteSolverConfig:
    requirements = context_evaluation_requirements(request)
    if requirements.max_day_minutes is None:
        return config
    return replace(config, max_day_minutes=min(config.max_day_minutes, requirements.max_day_minutes))


def _evaluator_config_for_repair(config: PlanEvaluatorConfig, request: Any) -> PlanEvaluatorConfig:
    requirements = context_evaluation_requirements(request)
    max_day_minutes = config.max_day_minutes
    if requirements.max_day_minutes is not None:
        max_day_minutes = min(max_day_minutes, requirements.max_day_minutes)
    return replace(
        config,
        max_day_minutes=max_day_minutes,
        required_stop_ids=requirements.required_stop_ids,
        excluded_stop_ids=requirements.excluded_stop_ids,
        closed_route_ids=requirements.closed_route_ids,
        unavailable_lodging_ids=requirements.unavailable_lodging_ids,
    )


def _planner_run_for_child_plan(child_plan: PlanArtifactV2, *, method_id: str) -> PlannerRun:
    return PlannerRun(
        run_id=child_plan.source_run_id,
        planning_request_id=child_plan.planning_request_id,
        catalog_snapshot_id=child_plan.catalog_snapshot_id,
        context_snapshot_id=child_plan.context_snapshot_id,
        planner_specification_id=f"pipeline-{method_id}-adapter-v1",
        method_requested=method_id,
        method_executed=method_id,
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        result_plan_id=child_plan.plan_id,
    )


def _repair_request_id(request: Any) -> str:
    return str(getattr(request, "request_id", "") or getattr(request, "repair_request_id", "") or "repair_request")


def _repair_request_record(request: Any, *, context: PipelineRunContext) -> dict[str, Any]:
    raw: dict[str, Any]
    if isinstance(request, Mapping):
        raw = dict(request)
    elif hasattr(request, "to_record"):
        raw = dict(request.to_record())
    elif hasattr(request, "__dict__"):
        raw = dict(vars(request))
    else:
        raw = {"request_repr": repr(request)}
    request_id = str(raw.get("request_id") or raw.get("repair_request_id") or context.repair_request_id or "repair_request")
    return {
        **_jsonish(raw),
        "request_id": request_id,
        "kind": "progressive_repair",
        "parent_plan_id": context.parent_plan_id,
        "repair_request_id": context.repair_request_id or request_id,
        "schema_version": "progressive-repair-request-v1",
    }


def _repair_metrics(outcome: Any) -> dict[str, float | int | str | bool | None]:
    accepted_radius = getattr(outcome, "accepted_radius", None)
    attempts = tuple(getattr(outcome, "attempts", ()))
    candidate_evaluation_count = sum(len(getattr(attempt, "candidate_evaluations", ())) for attempt in attempts)
    return {
        "repair_outcome_id": getattr(outcome, "repair_outcome_id", ""),
        "repair_outcome_status": getattr(outcome, "status", ""),
        "accepted_radius": getattr(accepted_radius, "value", None) if accepted_radius is not None else None,
        "repair_attempt_count": len(attempts),
        "repair_candidate_evaluation_count": int(candidate_evaluation_count),
        "repair_output_plan_count": 1 if getattr(outcome, "child_plan", None) is not None else 0,
    }


def _repair_explanation_records(outcome: Any, *, route_matrix: RouteMatrix) -> tuple[Any, ...]:
    if getattr(outcome, "child_plan", None) is not None and getattr(outcome, "diff_record", None):
        return (_accepted_repair_explanation(outcome, route_matrix=route_matrix),)
    if getattr(outcome, "diagnosis", None) is not None:
        return (_failed_repair_explanation(outcome),)
    return ()


def _accepted_repair_explanation(outcome: Any, *, route_matrix: RouteMatrix) -> ContrastiveEvidence:
    child_plan = outcome.child_plan
    diff_record = dict(outcome.diff_record or {})
    evaluation_record = dict(outcome.evaluation_record or {})
    diff_id = _record_id(diff_record, "diff_id", prefix="diff")
    evaluation_id = _record_id(evaluation_record, "certificate_id", "evaluation_id", prefix="evaluation")
    diff_ref = f"plan_diff:{diff_id}:weighted_edit_cost"
    evaluation_ref = f"evaluation:{evaluation_id}:comparison_eligibility"
    route_ref = f"route:{route_matrix.matrix_id}:matrix"
    accepted_radius = getattr(getattr(outcome, "accepted_radius", None), "value", "")
    weighted_edit_cost = diff_record.get("weighted_edit_cost", 0.0)
    evidence_records = (
        EvidenceRecord(
            ref_id=diff_ref,
            source_type="plan_diff",
            source_id=diff_id,
            field_path="weighted_edit_cost",
            payload={"value": weighted_edit_cost},
        ),
        EvidenceRecord(
            ref_id=evaluation_ref,
            source_type="evaluation",
            source_id=evaluation_id,
            field_path="comparison_eligibility",
            payload={"value": evaluation_record.get("comparison_eligibility")},
        ),
        EvidenceRecord(
            ref_id=route_ref,
            source_type="route",
            source_id=route_matrix.matrix_id,
            field_path="matrix_id",
            payload={"context_snapshot_id": route_matrix.context_snapshot_id},
        ),
    )
    claims = (
        ExplanationClaim(
            claim_id=f"claim_acceptance_{child_plan.plan_id}",
            claim_type="causal",
            text_template="The repair accepted {accepted_radius} after the child plan passed independent evaluation.",
            values={"accepted_radius": accepted_radius},
            evidence_refs=(evaluation_ref, diff_ref, route_ref),
        ),
        ExplanationClaim(
            claim_id=f"claim_weighted_edit_cost_{child_plan.plan_id}",
            claim_type="numeric",
            text_template="The ownership-weighted edit cost is {weighted_edit_cost}.",
            values={"weighted_edit_cost": weighted_edit_cost},
            evidence_refs=(diff_ref,),
        ),
    )
    return ContrastiveEvidence(
        evidence_id=f"repair_explanation_{stable_content_hash({'child': child_plan.plan_id, 'diff': diff_id})}",
        parent_plan_id=outcome.parent_plan_id,
        child_plan_id=child_plan.plan_id,
        diff_id=diff_id,
        claims=claims,
        evidence_records=evidence_records,
    )


def _failed_repair_explanation(outcome: Any) -> WhyNotEvidence:
    failed_attempt_ids = tuple(str(getattr(attempt, "attempt_id", "")) for attempt in getattr(outcome, "attempts", ()))
    evidence_records = []
    for attempt in getattr(outcome, "attempts", ()):
        for record in getattr(attempt, "candidate_evaluations", ()):
            candidate_record = record.to_record() if hasattr(record, "to_record") else dict(record)
            evaluation_id = str(candidate_record.get("evaluation_id") or "")
            if not evaluation_id:
                continue
            evidence_records.append(
                EvidenceRecord(
                    ref_id=f"evaluation:{evaluation_id}:comparison_eligibility",
                    source_type="evaluation",
                    source_id=evaluation_id,
                    field_path="comparison_eligibility",
                    payload=candidate_record,
                )
            )
    evidence_refs = tuple(record.ref_id for record in evidence_records)
    claims = (
        ExplanationClaim(
            claim_id=f"claim_no_repair_{outcome.repair_outcome_id}",
            claim_type="causal",
            text_template="No configured repair radius produced an independently eligible child plan.",
            values={},
            evidence_refs=evidence_refs,
            unsupported=not evidence_refs,
        ),
    )
    return WhyNotEvidence(
        evidence_id=f"repair_failure_{outcome.repair_outcome_id}",
        target_id=outcome.parent_plan_id,
        failed_attempt_ids=failed_attempt_ids,
        outcome_status=getattr(outcome, "status", ""),
        claims=claims,
        evidence_records=tuple(evidence_records),
    )


def _phase0_request_records(
    planner_records: Iterable[Mapping[str, Any]],
    *,
    context: PipelineRunContext,
    legacy_artifact_dir: str,
) -> tuple[dict[str, Any], ...]:
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in planner_records:
        request_id = str(record.get("planning_request_id") or "").strip()
        if not request_id:
            request_id = "request_" + stable_content_hash(record)
        if request_id in seen:
            continue
        seen.add(request_id)
        requests.append(
            {
                "request_id": request_id,
                "kind": "phase0_generation",
                "method_requested": record.get("method_requested"),
                "catalog_snapshot_id": context.catalog_snapshot_id,
                "context_snapshot_id": context.context_snapshot_id,
                "legacy_artifact_dir": legacy_artifact_dir,
                "schema_version": "phase0-pipeline-request-v1",
            }
        )
    if requests:
        return tuple(requests)
    return (
        {
            "request_id": f"request_{context.run_id}",
            "kind": "phase0_generation",
            "catalog_snapshot_id": context.catalog_snapshot_id,
            "context_snapshot_id": context.context_snapshot_id,
            "legacy_artifact_dir": legacy_artifact_dir,
            "schema_version": "phase0-pipeline-request-v1",
        },
    )


def _phase0_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _phase0_jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_phase0_jsonish(item) for item in value]
    if isinstance(value, list):
        return [_phase0_jsonish(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _phase0_jsonish(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "validated", "road_validated", "eligible"}


def _relative_artifact_path(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _record_from_artifact(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, RouteMatrix):
        return {
            "matrix_id": artifact.matrix_id,
            "context_snapshot_id": artifact.context_snapshot_id,
            "source_bundle_id": artifact.source_bundle_id,
            "source_content_sha256": artifact.source_content_sha256,
            "entity_ids": list(artifact.entity_ids),
            "cells": [_jsonish(asdict(cell)) for cell in artifact.cells.values()],
            "schema_version": "route-matrix-v1",
        }
    if hasattr(artifact, "to_record"):
        return dict(artifact.to_record())
    if isinstance(artifact, Mapping):
        return dict(artifact)
    raise TypeError(f"unsupported pipeline artifact type: {type(artifact).__name__}")


def _record_id(record: Mapping[str, Any], *keys: str, prefix: str) -> str:
    for key in keys:
        value = str(record.get(key) or "").strip()
        if value:
            return _safe_filename(value)
    return f"{prefix}_{stable_content_hash(record)}"


def _write_json(path: Path, record: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writable_path = _windows_extended_path(path)
    writable_path.write_text(
        json.dumps(_jsonish(record), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _windows_extended_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    backslash = chr(92)
    extended_prefix = backslash * 2 + "?" + backslash
    unc_prefix = backslash * 2
    if resolved.startswith(extended_prefix):
        return Path(resolved)
    if resolved.startswith(unc_prefix):
        return Path(extended_prefix + "UNC" + backslash + resolved.lstrip(backslash))
    return Path(extended_prefix + resolved)


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonish(item) for item in value]
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    if hasattr(value, "to_record"):
        return _jsonish(value.to_record())
    return value


def _redact(value: Any, *, key_name: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact(item, key_name=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, key_name=key_name) for item in value]
    if _is_sensitive_key(key_name):
        return "***REDACTED***"
    return value


def _is_sensitive_key(key_name: str) -> bool:
    lowered = key_name.lower()
    return any(token in lowered for token in ("api_key", "access_token", "secret", "password", "credential"))


def _coerce_refresh_policy(value: RefreshPolicy | str) -> RefreshPolicy:
    if isinstance(value, RefreshPolicy):
        return value
    return RefreshPolicy(str(value))


def _default_run_id(
    *,
    config_path: str | Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan_id: str | None,
    repair_request_id: str | None,
) -> str:
    return "run_" + stable_content_hash(
        {
            "config_path": str(config_path),
            "catalog_snapshot_id": catalog_snapshot_id,
            "context_snapshot_id": context_snapshot_id,
            "parent_plan_id": parent_plan_id,
            "repair_request_id": repair_request_id,
        }
    )


def _safe_filename(value: str, *, max_length: int = 40) -> str:
    clean = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    if len(clean) <= max_length:
        return clean
    suffix = stable_content_hash({"filename": clean})[:12]
    prefix_length = max(1, max_length - len(suffix) - 1)
    return f"{clean[:prefix_length]}_{suffix}"


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base, default=str))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
