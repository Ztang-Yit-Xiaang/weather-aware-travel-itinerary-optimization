"""Structured explanation evidence with claim-to-artifact validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..research_artifacts import stable_content_hash

ClaimType = Literal["numeric", "causal", "descriptive"]
EvidenceSourceType = Literal[
    "constraint",
    "route",
    "plan_diff",
    "evaluation",
    "certificate",
    "counterfactual_run",
    "planner_run",
]

ALLOWED_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "constraint",
        "route",
        "plan_diff",
        "evaluation",
        "certificate",
        "counterfactual_run",
        "planner_run",
    }
)
NUMERIC_SOURCE_TYPES: frozenset[str] = frozenset(
    {"route", "plan_diff", "evaluation", "certificate", "counterfactual_run", "planner_run"}
)
CAUSAL_SOURCE_TYPES: frozenset[str] = ALLOWED_SOURCE_TYPES


@dataclass(frozen=True)
class EvidenceRecord:
    """One structured source record that can support explanation claims."""

    ref_id: str
    source_type: EvidenceSourceType | str
    source_id: str
    field_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def __post_init__(self) -> None:
        if not self.record_hash:
            object.__setattr__(
                self,
                "record_hash",
                stable_content_hash(
                    {
                        "ref_id": self.ref_id,
                        "source_type": self.source_type,
                        "source_id": self.source_id,
                        "field_path": self.field_path,
                        "payload": self.payload,
                    }
                ),
            )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationFinding:
    """Validation finding for unsupported or malformed explanation claims."""

    code: str
    claim_id: str
    message: str
    evidence_ref: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationClaim:
    """A user-facing claim that must cite structured evidence when needed."""

    claim_id: str
    claim_type: ClaimType | str
    text_template: str
    evidence_refs: tuple[str, ...]
    values: dict[str, Any] = field(default_factory=dict)
    confidence: str = "derived"
    unsupported: bool = False

    def validate(self, evidence_records: tuple[EvidenceRecord, ...]) -> tuple[ExplanationFinding, ...]:
        return validate_explanation_claims((self,), evidence_records)

    def render(self) -> str:
        try:
            return self.text_template.format(**self.values)
        except Exception:
            return self.text_template

    def to_record(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "text_template": self.text_template,
            "rendered_text": self.render(),
            "values": self.values,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "unsupported": self.unsupported,
        }


@dataclass(frozen=True, kw_only=True)
class ExplanationEvidence:
    """Base container for validated explanation evidence bundles."""

    evidence_id: str
    evidence_type: str
    claims: tuple[ExplanationClaim, ...]
    evidence_records: tuple[EvidenceRecord, ...] = ()
    schema_version: str = "explanation-evidence-v1"

    def validate(self) -> tuple[ExplanationFinding, ...]:
        return validate_explanation_claims(self.claims, self.evidence_records)

    def evidence_refs(self) -> tuple[str, ...]:
        refs: list[str] = []
        for claim in self.claims:
            for ref in claim.evidence_refs:
                if ref not in refs:
                    refs.append(ref)
        return tuple(refs)

    def supported_claims(self) -> tuple[ExplanationClaim, ...]:
        unsupported_ids = {finding.claim_id for finding in self.validate()}
        return tuple(claim for claim in self.claims if not claim.unsupported and claim.claim_id not in unsupported_ids)

    def to_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "claims": [claim.to_record() for claim in self.claims],
            "evidence_records": [record.to_record() for record in self.evidence_records],
            "findings": [finding.to_record() for finding in self.validate()],
            "schema_version": self.schema_version,
        }

    def to_publication_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "claims": [claim.to_record() for claim in self.supported_claims()],
            "evidence_refs": list(self.evidence_refs()),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, kw_only=True)
class WhyEvidence(ExplanationEvidence):
    """Evidence explaining why a plan, stop, route, or choice was selected or kept."""

    plan_id: str = ""
    target_id: str = ""
    evidence_type: str = "why"

    def to_record(self) -> dict[str, Any]:
        return {**super().to_record(), "plan_id": self.plan_id, "target_id": self.target_id}

    def to_publication_record(self) -> dict[str, Any]:
        return {**super().to_publication_record(), "plan_id": self.plan_id, "target_id": self.target_id}


@dataclass(frozen=True, kw_only=True)
class WhyNotEvidence(ExplanationEvidence):
    """Evidence explaining why an option was omitted, failed, or deferred."""

    target_id: str = ""
    failed_attempt_ids: tuple[str, ...] = ()
    counterfactual_run_ids: tuple[str, ...] = ()
    outcome_status: str = ""
    evidence_type: str = "why_not"

    def to_record(self) -> dict[str, Any]:
        return {
            **super().to_record(),
            "target_id": self.target_id,
            "failed_attempt_ids": list(self.failed_attempt_ids),
            "counterfactual_run_ids": list(self.counterfactual_run_ids),
            "outcome_status": self.outcome_status,
        }

    def to_publication_record(self) -> dict[str, Any]:
        return {
            **super().to_publication_record(),
            "target_id": self.target_id,
            "failed_attempt_ids": list(self.failed_attempt_ids),
            "counterfactual_run_ids": list(self.counterfactual_run_ids),
            "outcome_status": self.outcome_status,
        }


@dataclass(frozen=True, kw_only=True)
class WhatIfEvidence(ExplanationEvidence):
    """Evidence for a sandbox counterfactual result."""

    sandbox_request_id: str = ""
    counterfactual_run_ids: tuple[str, ...] = ()
    outcome_status: str = ""
    evidence_type: str = "what_if"

    def to_record(self) -> dict[str, Any]:
        return {
            **super().to_record(),
            "sandbox_request_id": self.sandbox_request_id,
            "counterfactual_run_ids": list(self.counterfactual_run_ids),
            "outcome_status": self.outcome_status,
        }

    def to_publication_record(self) -> dict[str, Any]:
        return {
            **super().to_publication_record(),
            "sandbox_request_id": self.sandbox_request_id,
            "counterfactual_run_ids": list(self.counterfactual_run_ids),
            "outcome_status": self.outcome_status,
        }


@dataclass(frozen=True, kw_only=True)
class ContrastiveEvidence(ExplanationEvidence):
    """Evidence explaining a parent/child or method/method contrast."""

    parent_plan_id: str = ""
    child_plan_id: str = ""
    diff_id: str = ""
    evidence_type: str = "contrastive"

    def to_record(self) -> dict[str, Any]:
        return {
            **super().to_record(),
            "parent_plan_id": self.parent_plan_id,
            "child_plan_id": self.child_plan_id,
            "diff_id": self.diff_id,
        }

    def to_publication_record(self) -> dict[str, Any]:
        return {
            **super().to_publication_record(),
            "parent_plan_id": self.parent_plan_id,
            "child_plan_id": self.child_plan_id,
            "diff_id": self.diff_id,
        }


class ExplanationEvidenceBuilder:
    """Build structured evidence bundles from existing plan, diff, and evaluation artifacts."""

    def why(self, plan: Any, evaluation: Any) -> WhyEvidence:
        """Explain why a final plan is eligible or blocked using evaluator-owned evidence."""

        plan_record = _artifact_record(plan)
        evaluation_record = _artifact_record(evaluation)
        plan_id = _artifact_id(plan, plan_record, "plan_id", prefix="plan")
        evaluation_id = _artifact_id(evaluation, evaluation_record, "certificate_id", "evaluation_id", prefix="evaluation")
        evaluation_source_type = _evaluation_source_type(evaluation_record)
        eligibility = str(evaluation_record.get("comparison_eligibility", "unknown"))
        evaluation_ref = f"{evaluation_source_type}:{evaluation_id}:comparison_eligibility"
        evidence_records = (
            EvidenceRecord(
                ref_id=evaluation_ref,
                source_type=evaluation_source_type,
                source_id=evaluation_id,
                field_path="comparison_eligibility",
                payload={"value": eligibility},
            ),
        )
        claims = (
            ExplanationClaim(
                claim_id=f"claim_plan_eligibility_{plan_id}",
                claim_type="causal",
                text_template="Plan {plan_id} is {comparison_eligibility} after independent evaluation.",
                values={"plan_id": plan_id, "comparison_eligibility": eligibility},
                evidence_refs=(evaluation_ref,),
            ),
        )
        return WhyEvidence(
            evidence_id=f"why_{stable_content_hash({'plan_id': plan_id, 'evaluation_id': evaluation_id})}",
            plan_id=plan_id,
            target_id=plan_id,
            claims=claims,
            evidence_records=evidence_records,
        )

    def changed(self, plan: Any, diff: Any, evaluation: Any) -> ContrastiveEvidence:
        """Explain parent-child changes using diff cost and independent evaluation evidence."""

        plan_record = _artifact_record(plan)
        diff_record = _artifact_record(diff)
        evaluation_record = _artifact_record(evaluation)
        plan_id = _artifact_id(plan, plan_record, "plan_id", prefix="plan")
        diff_id = _artifact_id(diff, diff_record, "diff_id", prefix="diff")
        parent_plan_id = str(diff_record.get("parent_plan_id") or plan_record.get("parent_plan_id") or "")
        child_plan_id = str(diff_record.get("child_plan_id") or plan_id)
        evaluation_id = _artifact_id(evaluation, evaluation_record, "certificate_id", "evaluation_id", prefix="evaluation")
        evaluation_source_type = _evaluation_source_type(evaluation_record)
        weighted_edit_cost = float(diff_record.get("weighted_edit_cost", 0.0) or 0.0)
        eligibility = str(evaluation_record.get("comparison_eligibility", "unknown"))
        diff_ref = f"plan_diff:{diff_id}:weighted_edit_cost"
        evaluation_ref = f"{evaluation_source_type}:{evaluation_id}:comparison_eligibility"
        certificate_route_ref = f"{evaluation_source_type}:{evaluation_id}:route_validation"
        route_validation = evaluation_record.get("route_validation")
        route_record_id = _route_validation_id(route_validation)
        route_ref = f"route:{route_record_id}:route_validation" if route_record_id else ""
        records = [
            EvidenceRecord(
                ref_id=diff_ref,
                source_type="plan_diff",
                source_id=diff_id,
                field_path="weighted_edit_cost",
                payload={"value": weighted_edit_cost},
            ),
            EvidenceRecord(
                ref_id=evaluation_ref,
                source_type=evaluation_source_type,
                source_id=evaluation_id,
                field_path="comparison_eligibility",
                payload={"value": eligibility},
            ),
        ]
        if "route_validation" in evaluation_record:
            records.append(
                EvidenceRecord(
                    ref_id=certificate_route_ref,
                    source_type=evaluation_source_type,
                    source_id=evaluation_id,
                    field_path="route_validation",
                    payload={"value": route_validation},
                )
            )
        if route_ref:
            records.append(
                EvidenceRecord(
                    ref_id=route_ref,
                    source_type="route",
                    source_id=route_record_id,
                    field_path="route_validation",
                    payload=route_validation if isinstance(route_validation, dict) else {"value": route_validation},
                )
            )
        evaluation_refs = tuple(
            ref
            for ref in (
                evaluation_ref,
                certificate_route_ref if "route_validation" in evaluation_record else "",
                route_ref,
            )
            if ref
        )
        claims = (
            ExplanationClaim(
                claim_id=f"claim_independent_evaluation_{child_plan_id}",
                claim_type="causal",
                text_template=(
                    "The child plan {child_plan_id} is {comparison_eligibility} after independent evaluation."
                ),
                values={"child_plan_id": child_plan_id, "comparison_eligibility": eligibility},
                evidence_refs=evaluation_refs,
            ),
            ExplanationClaim(
                claim_id=f"claim_weighted_edit_cost_{child_plan_id}",
                claim_type="numeric",
                text_template="The ownership-weighted edit cost is {weighted_edit_cost}.",
                values={"weighted_edit_cost": weighted_edit_cost},
                evidence_refs=(diff_ref,),
            ),
        )
        return ContrastiveEvidence(
            evidence_id=f"contrastive_{stable_content_hash({'child': child_plan_id, 'diff': diff_id})}",
            parent_plan_id=parent_plan_id,
            child_plan_id=child_plan_id,
            diff_id=diff_id,
            claims=claims,
            evidence_records=tuple(records),
        )


def build_explanation_evidence(plan: Any, diff: Any, evaluation: Any) -> tuple[WhyEvidence, ContrastiveEvidence]:
    """Build the default EXPLAIN-001 why and changed evidence bundles."""

    builder = ExplanationEvidenceBuilder()
    return (builder.why(plan, evaluation), builder.changed(plan, diff, evaluation))


def validate_explanation_claims(
    claims: ExplanationClaim | Iterable[ExplanationClaim],
    evidence_records: EvidenceRecord | Iterable[EvidenceRecord],
) -> tuple[ExplanationFinding, ...]:
    """Validate that explanation claims cite valid structured evidence."""

    claim_tuple = (claims,) if isinstance(claims, ExplanationClaim) else tuple(claims)
    evidence_record_tuple = (
        (evidence_records,) if isinstance(evidence_records, EvidenceRecord) else tuple(evidence_records)
    )
    index = {record.ref_id: record for record in evidence_record_tuple}
    findings: list[ExplanationFinding] = []
    for claim in claim_tuple:
        claim_type = str(claim.claim_type)
        if claim.unsupported:
            findings.append(
                ExplanationFinding(
                    code="claim_marked_unsupported",
                    claim_id=claim.claim_id,
                    message="Claim is explicitly marked unsupported.",
                )
            )
            continue
        if claim_type not in {"numeric", "causal", "descriptive"}:
            findings.append(
                ExplanationFinding(
                    code="unknown_claim_type",
                    claim_id=claim.claim_id,
                    message=f"Unknown claim type: {claim.claim_type!r}.",
                )
            )
            continue
        if claim_type in {"numeric", "causal"} and not claim.evidence_refs:
            findings.append(
                ExplanationFinding(
                    code="missing_evidence_refs",
                    claim_id=claim.claim_id,
                    message="Numerical and causal claims require at least one evidence reference.",
                )
            )
            continue
        for evidence_ref in claim.evidence_refs:
            record = index.get(evidence_ref)
            if record is None:
                findings.append(
                    ExplanationFinding(
                        code="missing_evidence_ref",
                        claim_id=claim.claim_id,
                        message=f"Evidence reference {evidence_ref!r} was not found.",
                        evidence_ref=evidence_ref,
                    )
                )
                continue
            source_type = str(record.source_type)
            if source_type not in ALLOWED_SOURCE_TYPES:
                findings.append(
                    ExplanationFinding(
                        code="invalid_evidence_type",
                        claim_id=claim.claim_id,
                        message=f"Evidence type {source_type!r} is not allowed for explanations.",
                        evidence_ref=evidence_ref,
                    )
                )
                continue
            if claim_type == "numeric" and source_type not in NUMERIC_SOURCE_TYPES:
                findings.append(
                    ExplanationFinding(
                        code="invalid_evidence_type",
                        claim_id=claim.claim_id,
                        message=f"Numerical claim cannot be supported by evidence type {source_type!r}.",
                        evidence_ref=evidence_ref,
                    )
                )
            if claim_type == "causal" and source_type not in CAUSAL_SOURCE_TYPES:
                findings.append(
                    ExplanationFinding(
                        code="invalid_evidence_type",
                        claim_id=claim.claim_id,
                        message=f"Causal claim cannot be supported by evidence type {source_type!r}.",
                        evidence_ref=evidence_ref,
                    )
                )
    return tuple(findings)


def _artifact_record(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        return dict(artifact)
    if hasattr(artifact, "to_record"):
        return dict(artifact.to_record())
    if hasattr(artifact, "__dict__"):
        return dict(vars(artifact))
    return {}


def _artifact_id(artifact: Any, record: dict[str, Any], *keys: str, prefix: str) -> str:
    for key in keys:
        value = record.get(key, None)
        if value:
            return str(value)
        value = getattr(artifact, key, None)
        if value:
            return str(value)
    return f"{prefix}_{stable_content_hash(record)}"


def _evaluation_source_type(evaluation_record: dict[str, Any]) -> str:
    if evaluation_record.get("certificate_id"):
        return "certificate"
    return "evaluation"


def _route_validation_id(route_validation: Any) -> str:
    if not isinstance(route_validation, dict):
        return ""
    for key in ("matrix_id", "route_matrix_id", "route_result_id", "route_id"):
        value = route_validation.get(key)
        if value:
            return str(value)
    return ""
