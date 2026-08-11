"""End-to-end rule-based interaction controller over existing repair components."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..repair_planner import RepairRequest
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .clarification_policy import decide_clarification
from .consequence import build_consequence_vector
from .counterfactual_probe import CounterfactualProbeExecutor
from .models import (
    ClarificationAction,
    ClarificationDecision,
    ConsequenceThresholds,
    CounterfactualProbeRequest,
    InteractionArtifacts,
    InteractionOptions,
    InteractionRequest,
    ProbeStatus,
    UserPermissionDecision,
)
from .patch_compiler import AllowListedPatchCompiler, confirmed_candidate
from .permission_policy import PermissionPolicy, granted_and_denied_constraint_ids
from .semantic_candidates import SemanticCandidateProvider
from .tradeoff_selector import CriticalTradeoffSelector


@dataclass(frozen=True)
class InteractionSessionResult:
    status: str
    decision: ClarificationDecision
    artifacts: InteractionArtifacts
    authorized_repair_request: RepairRequest | None
    metrics: dict[str, float | int | str | bool | None]


class PermissionAwareClarificationController:
    """Generate candidates, probe safely, and decide one inspectable next action."""

    def __init__(
        self,
        *,
        candidate_provider: SemanticCandidateProvider,
        probe_executor: CounterfactualProbeExecutor,
        compiler: AllowListedPatchCompiler | None = None,
        permission_policy: PermissionPolicy | None = None,
        thresholds: ConsequenceThresholds | None = None,
        tradeoff_selector: CriticalTradeoffSelector | None = None,
    ) -> None:
        self.candidate_provider = candidate_provider
        self.probe_executor = probe_executor
        self.compiler = compiler or AllowListedPatchCompiler()
        self.permission_policy = permission_policy or PermissionPolicy()
        self.thresholds = thresholds or ConsequenceThresholds()
        self.tradeoff_selector = tradeoff_selector or CriticalTradeoffSelector()

    def run(
        self,
        *,
        parent: PlanArtifactV2,
        request: InteractionRequest,
        options: InteractionOptions,
        permission_decisions: tuple[UserPermissionDecision, ...] = (),
    ) -> InteractionSessionResult:
        if options.clarification_mode.value == "disabled":
            raise ValueError("permission-aware controller cannot run when clarification mode is disabled")
        if request.parent_plan_id != parent.plan_id:
            raise ValueError("interaction request parent does not match parent artifact")
        candidates = self.candidate_provider.candidates(
            parent=parent,
            user_edit=request.user_edit,
            repair_session_id=request.repair_session_id,
            evidence_refs=request.evidence_refs,
        )
        if request.selected_interpretation_id:
            candidates = tuple(
                confirmed_candidate(candidate)
                for candidate in candidates
                if candidate.interpretation_id == request.selected_interpretation_id
            )
        patches = tuple(self.compiler.compile(parent, candidate) for candidate in candidates)
        assessments = {
            patch.interpretation_id: self.permission_policy.assess_patch(
                parent,
                patch,
                permission_decisions=permission_decisions,
                repair_session_id=request.repair_session_id,
            )
            for patch in patches
        }
        probe_requests: list[CounterfactualProbeRequest] = []
        probe_results = []
        candidate_by_id = {candidate.interpretation_id: candidate for candidate in candidates}
        for patch in patches:
            if not patch.is_valid:
                radius = next(iter(_safe_default_radius()))
            else:
                radius = self.compiler.repair_radius(patch)
            assessment = assessments[patch.interpretation_id]
            seed = {
                "session": request.repair_session_id,
                "patch": patch.patch_id,
                "radius": radius.value,
                "time_limit": options.probe_time_limit_seconds,
            }
            probe_request = CounterfactualProbeRequest(
                probe_request_id=f"probe_request_{stable_content_hash(seed)}",
                repair_session_id=request.repair_session_id,
                parent_plan_id=parent.plan_id,
                interpretation_id=patch.interpretation_id,
                model_patch_id=patch.patch_id,
                allowed_probe_constraint_ids=assessment.permission_constraint_ids,
                repair_radius=radius,
                time_limit_seconds=options.probe_time_limit_seconds,
            )
            probe_requests.append(probe_request)
            probe_results.append(
                self.probe_executor.execute(
                    parent=parent,
                    candidate=candidate_by_id[patch.interpretation_id],
                    patch=patch,
                    request=probe_request,
                    assessment=assessment,
                )
            )

        consequences = tuple(
            build_consequence_vector(
                parent=parent,
                interpretation_id=probe_request.interpretation_id,
                probe_result=probe_result,
            )
            for probe_request, probe_result in zip(probe_requests, probe_results, strict=True)
            if probe_result.status not in {ProbeStatus.INVALID_PATCH, ProbeStatus.BLOCKED_BY_POLICY}
        )
        tradeoff = self.tradeoff_selector.select(
            repair_session_id=request.repair_session_id,
            consequences=consequences,
        )
        decision = decide_clarification(
            candidates=candidates,
            probe_results=tuple(probe_results),
            consequences=consequences,
            thresholds=self.thresholds,
            question_count=request.question_count,
            max_questions=options.max_questions,
            tradeoff=tradeoff,
        )
        decision = self._apply_explicit_user_decision(
            decision=decision,
            request=request,
            patches=patches,
            assessments=assessments,
            probe_requests=tuple(probe_requests),
            probe_results=tuple(probe_results),
            permission_decisions=permission_decisions,
        )
        authorized_request = self._authorized_request(
            parent=parent,
            interaction_request=request,
            decision=decision,
            candidates=candidates,
            patches=patches,
            assessments=assessments,
            permission_decisions=permission_decisions,
        )
        status = _interaction_status(decision)
        artifacts = InteractionArtifacts(
            semantic_candidates=candidates,
            model_patches=patches,
            probe_requests=tuple(probe_requests),
            probe_results=tuple(probe_results),
            consequence_vectors=consequences,
            critical_tradeoffs=(tradeoff,) if tradeoff else (),
            clarification_decisions=(decision,),
            permission_decisions=permission_decisions,
            selected_authorized_repair_request_id=authorized_request.request_id if authorized_request else None,
            selected_interpretation_id=decision.selected_interpretation_id,
            semantic_candidate_source=str(
                getattr(self.candidate_provider, "source_name", type(self.candidate_provider).__name__)
            ),
            continuation_of_session_id=request.continuation_of_session_id,
        )
        metrics = _interaction_metrics(
            candidates=candidates,
            patches=patches,
            probe_results=tuple(probe_results),
            consequences=consequences,
            decision=decision,
            question_count=request.question_count,
            authorized_request=authorized_request,
        )
        return InteractionSessionResult(
            status=status,
            decision=decision,
            artifacts=artifacts,
            authorized_repair_request=authorized_request,
            metrics=metrics,
        )

    def _apply_explicit_user_decision(
        self,
        *,
        decision: ClarificationDecision,
        request: InteractionRequest,
        patches: tuple,
        assessments: dict,
        probe_requests: tuple[CounterfactualProbeRequest, ...],
        probe_results: tuple,
        permission_decisions: tuple[UserPermissionDecision, ...],
    ) -> ClarificationDecision:
        permission_interpretation = next(
            (
                probe_request.interpretation_id
                for probe_request, result in zip(probe_requests, probe_results, strict=True)
                if result.requires_user_permission
            ),
            None,
        )
        if decision.action == ClarificationAction.ASK_PERMISSION and decision.selected_interpretation_id is None:
            decision = replace(decision, selected_interpretation_id=permission_interpretation)
        selected = request.selected_interpretation_id or next(
            (
                item.selected_interpretation_id
                for item in reversed(permission_decisions)
                if item.repair_session_id == request.repair_session_id and item.selected_interpretation_id
            ),
            None,
        )
        if not selected or selected not in assessments:
            return decision
        assessment = assessments[selected]
        selected_result = next(
            (
                result
                for probe_request, result in zip(probe_requests, probe_results, strict=True)
                if probe_request.interpretation_id == selected
            ),
            None,
        )
        if selected_result is None or selected_result.status not in {
            ProbeStatus.FEASIBLE_BOUNDED,
            ProbeStatus.TIME_LIMIT_WITH_INCUMBENT,
        }:
            return decision
        if not assessment.allowed_for_authorized_repair:
            return decision
        granted, _ = granted_and_denied_constraint_ids(
            permission_decisions,
            repair_session_id=request.repair_session_id,
            interpretation_id=selected,
        )
        reason = "permission_granted_for_session" if granted else "user_confirmed_interpretation"
        return replace(
            decision,
            action=ClarificationAction.COMMIT,
            selected_interpretation_id=selected,
            question_text=None,
            reason_codes=(reason,),
        )

    def _authorized_request(
        self,
        *,
        parent: PlanArtifactV2,
        interaction_request: InteractionRequest,
        decision: ClarificationDecision,
        candidates: tuple,
        patches: tuple,
        assessments: dict,
        permission_decisions: tuple[UserPermissionDecision, ...],
    ) -> RepairRequest | None:
        if decision.action != ClarificationAction.COMMIT or not decision.selected_interpretation_id:
            return None
        selected_id = decision.selected_interpretation_id
        assessment = assessments.get(selected_id)
        if assessment is None or not assessment.allowed_for_authorized_repair:
            return None
        candidate = next(item for item in candidates if item.interpretation_id == selected_id)
        patch = next(item for item in patches if item.interpretation_id == selected_id)
        granted, _ = granted_and_denied_constraint_ids(
            permission_decisions,
            repair_session_id=interaction_request.repair_session_id,
            interpretation_id=selected_id,
        )
        seed = {
            "session": interaction_request.repair_session_id,
            "interpretation": selected_id,
            "patch": patch.patch_id,
            "granted": sorted(granted),
            "continuation": interaction_request.continuation_of_session_id,
        }
        return self.compiler.build_repair_request(
            parent=parent,
            candidate=confirmed_candidate(candidate),
            patch=patch,
            request_id=f"authorized_repair_{stable_content_hash(seed)}",
            test_only=False,
            granted_constraint_ids=tuple(sorted(granted)),
        )


def _interaction_status(decision: ClarificationDecision) -> str:
    if decision.action in {ClarificationAction.ASK_PERMISSION, ClarificationAction.ASK_SEMANTIC}:
        return "confirmation_required"
    if decision.action == ClarificationAction.PROBE_MORE:
        return "probe_required"
    if decision.action == ClarificationAction.DEFER:
        return "deferred"
    return "committed"


def _interaction_metrics(
    *,
    candidates: tuple,
    patches: tuple,
    probe_results: tuple,
    consequences: tuple,
    decision: ClarificationDecision,
    question_count: int,
    authorized_request: RepairRequest | None,
) -> dict[str, float | int | str | bool | None]:
    runtime = sum(result.runtime_seconds or 0.0 for result in probe_results)
    final_vector = next(
        (item for item in consequences if item.interpretation_id == decision.selected_interpretation_id),
        None,
    )
    asks = decision.action in {ClarificationAction.ASK_PERMISSION, ClarificationAction.ASK_SEMANTIC}
    return {
        "interaction_semantic_candidate_count": len(candidates),
        "interaction_valid_candidate_count": sum(patch.is_valid for patch in patches),
        "interaction_solver_probe_count": len(probe_results),
        "interaction_probe_runtime_seconds": runtime,
        "interaction_clarification_action": decision.action.value,
        "interaction_question_count": question_count + int(asks),
        "interaction_unnecessary_question_indicator": None,
        "interaction_unsafe_auto_commit_indicator": None,
        "interaction_permission_gated_automatic_change_count": None,
        "interaction_ownership_weighted_semantic_regret": None,
        "interaction_authorized_request_ready": authorized_request is not None,
        "interaction_parent_plan_weighted_edit_cost": final_vector.weighted_edit_cost if final_vector else None,
        "interaction_unchanged_day_ratio": None,
        "interaction_accepted_repair_radius": final_vector.accepted_repair_radius if final_vector else None,
        "interaction_explanation_evidence_coverage": None,
    }


def _safe_default_radius():
    from ..repair.neighborhood import RepairRadius

    return (RepairRadius.SAME_DAY_REPLACEMENT,)
