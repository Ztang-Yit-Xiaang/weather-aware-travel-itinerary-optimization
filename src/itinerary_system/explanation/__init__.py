"""Explanation evidence containers and validation helpers."""

from .counterfactual import (
    CounterfactualRequest,
    CounterfactualRunner,
    CounterfactualRunRecord,
    build_counterfactual_request,
)
from .evidence import (
    ContrastiveEvidence,
    EvidenceRecord,
    ExplanationClaim,
    ExplanationEvidence,
    ExplanationEvidenceBuilder,
    ExplanationFinding,
    WhatIfEvidence,
    WhyEvidence,
    WhyNotEvidence,
    build_explanation_evidence,
    validate_explanation_claims,
)
from .verbalizer import DeterministicTemplateVerbalizer, RenderedExplanation, validate_claim_evidence_map

__all__ = [
    "ContrastiveEvidence",
    "CounterfactualRequest",
    "CounterfactualRunRecord",
    "CounterfactualRunner",
    "DeterministicTemplateVerbalizer",
    "EvidenceRecord",
    "ExplanationClaim",
    "ExplanationEvidence",
    "ExplanationEvidenceBuilder",
    "ExplanationFinding",
    "RenderedExplanation",
    "WhatIfEvidence",
    "WhyEvidence",
    "WhyNotEvidence",
    "build_counterfactual_request",
    "build_explanation_evidence",
    "validate_claim_evidence_map",
    "validate_explanation_claims",
]
