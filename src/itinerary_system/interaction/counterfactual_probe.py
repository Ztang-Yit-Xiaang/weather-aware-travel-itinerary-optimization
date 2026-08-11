"""Bounded non-executable probes that reuse the progressive repair controller."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..plans.repository import PlanRepository
from ..repair.progressive import ProgressiveRepairController, RepairOutcome
from ..repair_planner import RepairRequest
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import (
    CounterfactualProbeRequest,
    CounterfactualProbeResult,
    ModelPatch,
    ProbeStatus,
    SemanticInterpretationCandidate,
)
from .patch_compiler import AllowListedPatchCompiler
from .permission_policy import PatchPermissionAssessment

ProbeRepairExecutor = Callable[[RepairRequest], RepairOutcome]


class CounterfactualProbeExecutor:
    """Execute one test-only request and discard execution eligibility/certification.

    Production callers should use :meth:`from_progressive_controller`, which
    clones the controller around a dedicated hypothetical plan repository.
    The callable form exists for deterministic unit fixtures.
    """

    def __init__(
        self,
        run_repair: ProbeRepairExecutor | None = None,
        *,
        compiler: AllowListedPatchCompiler | None = None,
        controller_template: ProgressiveRepairController | None = None,
        hypothetical_repository_root: str | Path | None = None,
    ) -> None:
        self.run_repair = run_repair
        self.compiler = compiler or AllowListedPatchCompiler()
        self.controller_template = controller_template
        self.hypothetical_repository_root = (
            Path(hypothetical_repository_root) if hypothetical_repository_root is not None else None
        )

    @classmethod
    def from_progressive_controller(
        cls,
        controller: ProgressiveRepairController,
        *,
        hypothetical_repository_root: str | Path,
        compiler: AllowListedPatchCompiler | None = None,
    ) -> CounterfactualProbeExecutor:
        return cls(
            compiler=compiler,
            controller_template=controller,
            hypothetical_repository_root=hypothetical_repository_root,
        )

    def execute(
        self,
        *,
        parent: PlanArtifactV2,
        candidate: SemanticInterpretationCandidate,
        patch: ModelPatch,
        request: CounterfactualProbeRequest,
        assessment: PatchPermissionAssessment,
    ) -> CounterfactualProbeResult:
        if request.parent_plan_id != parent.plan_id:
            raise ValueError("probe parent plan does not match the supplied parent artifact")
        if not patch.is_valid:
            return _blocked_result(request, ProbeStatus.INVALID_PATCH, patch.reason_codes)
        if not assessment.allowed_for_probe:
            return _blocked_result(request, ProbeStatus.BLOCKED_BY_POLICY, assessment.reason_codes)

        repair_request = self.compiler.build_repair_request(
            parent=parent,
            candidate=candidate,
            patch=patch,
            request_id=request.probe_request_id,
            test_only=True,
            granted_constraint_ids=request.allowed_probe_constraint_ids,
            time_limit_seconds=request.time_limit_seconds,
        )
        run_repair = self._isolated_executor(parent, request)
        before_hash = parent.content_hash
        started = time.perf_counter()
        try:
            outcome = run_repair(repair_request)
            runtime = time.perf_counter() - started
        except Exception as exc:
            runtime = time.perf_counter() - started
            return _failed_result(request, runtime, f"{type(exc).__name__}:{exc}")
        if parent.content_hash != before_hash:
            return _failed_result(request, runtime, "parent_plan_mutated")

        status = _probe_status(outcome)
        child_record = _hypothetical_plan_record(outcome.child_plan)
        diff_record = dict(outcome.diff_record) if outcome.diff_record else None
        planner_records = tuple(run.to_record() for run in outcome.planner_runs)
        solver_run_ids = tuple(str(record.get("run_id", "")) for record in planner_records if record.get("run_id"))
        bound = _first_numeric(planner_records, "best_bound", "solver_bound", "bound")
        gap = _first_numeric(planner_records, "mip_gap", "solver_gap", "gap")
        failure_reasons = _outcome_failure_reasons(outcome)
        result_seed = {
            "probe_request_id": request.probe_request_id,
            "status": status.value,
            "hypothetical_plan_id": outcome.child_plan_id,
            "diff_id": diff_record.get("diff_id") if diff_record else None,
            "solver_runs": solver_run_ids,
        }
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    *patch.evidence_refs,
                    f"probe_request:{request.probe_request_id}",
                    *((f"hypothetical_plan:{outcome.child_plan_id}",) if outcome.child_plan_id else ()),
                    *(f"planner_run:{run_id}" for run_id in solver_run_ids),
                    *((f"plan_diff:{diff_record['diff_id']}",) if diff_record and diff_record.get("diff_id") else ()),
                )
            )
        )
        return CounterfactualProbeResult(
            probe_result_id=f"probe_result_{stable_content_hash(result_seed)}",
            probe_request_id=request.probe_request_id,
            parent_plan_id=parent.plan_id,
            hypothetical_plan_id=outcome.child_plan_id,
            status=status,
            diff_id=str(diff_record.get("diff_id")) if diff_record and diff_record.get("diff_id") else None,
            solver_run_ids=solver_run_ids,
            requires_user_permission=assessment.requires_user_permission,
            permission_constraint_ids=assessment.permission_constraint_ids,
            eligible_for_execution=False,
            evidence_refs=evidence_refs,
            interpretation_id=request.interpretation_id,
            affected_constraint_ids=patch.affected_constraint_ids,
            runtime_seconds=runtime,
            solver_bound=bound,
            solver_gap=gap,
            accepted_repair_radius=outcome.accepted_radius.value if outcome.accepted_radius else None,
            failure_reasons=failure_reasons,
            diff_record=diff_record,
            hypothetical_plan_record=child_record,
            diagnostic_evaluation=_diagnostic_evaluation(outcome.evaluation_record),
        )

    def _isolated_executor(
        self,
        parent: PlanArtifactV2,
        request: CounterfactualProbeRequest,
    ) -> ProbeRepairExecutor:
        if self.controller_template is not None and self.hypothetical_repository_root is not None:
            repository = PlanRepository(self.hypothetical_repository_root / request.probe_request_id)
            repository.save(parent)
            template = self.controller_template
            controller = ProgressiveRepairController(
                plan_repository=repository,
                route_matrix=template.route_matrix,
                evaluator=template.evaluator,
                day_route_config=template.day_route_config,
                tolerances=template.tolerances,
                ownership_policy=template.ownership_policy,
                publication_mode=template.publication_mode,
            )
            return controller.repair_progressively
        if self.run_repair is None:
            raise ValueError("counterfactual probe executor requires an isolated repair executor")
        return self.run_repair


def _blocked_result(
    request: CounterfactualProbeRequest,
    status: ProbeStatus,
    reasons: tuple[str, ...],
) -> CounterfactualProbeResult:
    seed = {"request": request.probe_request_id, "status": status.value, "reasons": reasons}
    return CounterfactualProbeResult(
        probe_result_id=f"probe_result_{stable_content_hash(seed)}",
        probe_request_id=request.probe_request_id,
        parent_plan_id=request.parent_plan_id,
        hypothetical_plan_id=None,
        status=status,
        diff_id=None,
        solver_run_ids=(),
        requires_user_permission=False,
        permission_constraint_ids=(),
        eligible_for_execution=False,
        evidence_refs=(f"probe_request:{request.probe_request_id}",),
        interpretation_id=request.interpretation_id,
        failure_reasons=reasons,
    )


def _failed_result(
    request: CounterfactualProbeRequest,
    runtime: float,
    reason: str,
) -> CounterfactualProbeResult:
    result = _blocked_result(request, ProbeStatus.FAILED, (reason,))
    return CounterfactualProbeResult(**{**result.__dict__, "runtime_seconds": runtime})


def _probe_status(outcome: RepairOutcome) -> ProbeStatus:
    statuses = " ".join(
        str(value).lower()
        for run in outcome.planner_runs
        for value in (run.to_record().get("status"), run.to_record().get("solver_status"))
        if value
    )
    if "time" in statuses and "limit" in statuses:
        return (
            ProbeStatus.TIME_LIMIT_WITH_INCUMBENT
            if outcome.child_plan is not None
            else ProbeStatus.TIME_LIMIT_NO_INCUMBENT
        )
    if outcome.child_plan is not None:
        return ProbeStatus.FEASIBLE_BOUNDED
    if outcome.status == "infeasible":
        return ProbeStatus.INFEASIBLE
    return ProbeStatus.FAILED


def _hypothetical_plan_record(plan: PlanArtifactV2 | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    record = plan.to_record()
    record["artifact_role"] = "hypothetical"
    record["test_only"] = True
    record["eligible_for_execution"] = False
    record["certificate_id"] = None
    return record


def _diagnostic_evaluation(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    diagnostic = dict(record)
    diagnostic.pop("certificate_id", None)
    diagnostic["evaluation_role"] = "probe_diagnostic"
    diagnostic["eligible_for_execution"] = False
    return diagnostic


def _outcome_failure_reasons(outcome: RepairOutcome) -> tuple[str, ...]:
    reasons: list[str] = []
    for attempt in outcome.attempts:
        reasons.extend(attempt.failure_reasons)
    if outcome.diagnosis is not None:
        reasons.extend(outcome.diagnosis.failure_reasons)
    return tuple(dict.fromkeys(reasons))


def _first_numeric(records: tuple[dict[str, Any], ...], *keys: str) -> float | None:
    for record in records:
        sources = (record, record.get("metrics", {}), record.get("metadata", {}))
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
    return None
