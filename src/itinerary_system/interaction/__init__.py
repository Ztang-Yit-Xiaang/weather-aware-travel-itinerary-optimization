"""Permission-aware counterfactual clarification research extension."""

from .clarification_policy import decide_clarification
from .consequence import build_consequence_vector, consequences_materially_different, equivalent_typed_repairs
from .controller import InteractionSessionResult, PermissionAwareClarificationController
from .counterfactual_probe import CounterfactualProbeExecutor
from .frozen_probe import FrozenCounterfactualProbeExecutor
from .models import (
    ClarificationAction,
    ClarificationDecision,
    ClarificationMode,
    ConsequenceThresholds,
    ConsequenceVector,
    CounterfactualProbeRequest,
    CounterfactualProbeResult,
    CriticalTradeoff,
    InteractionArtifacts,
    InteractionOptions,
    InteractionRequest,
    ModelPatch,
    PermissionDecisionAction,
    ProbeStatus,
    SemanticInterpretationCandidate,
    UserPermissionDecision,
)
from .patch_compiler import AllowListedPatchCompiler, PatchType
from .permission_policy import ConstraintPermissionClass, PatchPermissionAssessment, PermissionPolicy
from .pipeline import PermissionAwarePipelineRun, run_permission_aware_research_pipeline
from .semantic_candidates import (
    FrozenSemanticCandidateProvider,
    RuleBasedSemanticCandidateProvider,
    SemanticCandidateProvider,
)
from .tradeoff_selector import CriticalTradeoffSelector

__all__ = [
    "AllowListedPatchCompiler",
    "ClarificationAction",
    "ClarificationDecision",
    "ClarificationMode",
    "ConsequenceThresholds",
    "ConsequenceVector",
    "ConstraintPermissionClass",
    "CounterfactualProbeExecutor",
    "CounterfactualProbeRequest",
    "CounterfactualProbeResult",
    "CriticalTradeoff",
    "CriticalTradeoffSelector",
    "FrozenCounterfactualProbeExecutor",
    "FrozenSemanticCandidateProvider",
    "InteractionArtifacts",
    "InteractionOptions",
    "InteractionRequest",
    "InteractionSessionResult",
    "ModelPatch",
    "PatchPermissionAssessment",
    "PatchType",
    "PermissionAwareClarificationController",
    "PermissionAwarePipelineRun",
    "PermissionDecisionAction",
    "PermissionPolicy",
    "ProbeStatus",
    "RuleBasedSemanticCandidateProvider",
    "SemanticCandidateProvider",
    "SemanticInterpretationCandidate",
    "UserPermissionDecision",
    "build_consequence_vector",
    "consequences_materially_different",
    "decide_clarification",
    "equivalent_typed_repairs",
    "run_permission_aware_research_pipeline",
]
