"""Evidence-bounded explanation verbalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import ExplanationEvidence, ExplanationFinding


@dataclass(frozen=True)
class RenderedExplanation:
    """Rendered explanation text with claim-to-evidence mappings."""

    evidence_id: str
    evidence_type: str
    texts: tuple[str, ...]
    claim_evidence_map: dict[str, tuple[str, ...]]
    findings: tuple[ExplanationFinding, ...] = ()
    schema_version: str = "rendered-explanation-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "texts": list(self.texts),
            "claim_evidence_map": {
                claim_id: list(evidence_refs) for claim_id, evidence_refs in self.claim_evidence_map.items()
            },
            "findings": [finding.to_record() for finding in self.findings],
            "schema_version": self.schema_version,
        }


class DeterministicTemplateVerbalizer:
    """Benchmark-default verbalizer that renders only supported structured claims."""

    def render(self, evidence: ExplanationEvidence) -> RenderedExplanation:
        findings = evidence.validate()
        claims = evidence.supported_claims()
        return RenderedExplanation(
            evidence_id=evidence.evidence_id,
            evidence_type=evidence.evidence_type,
            texts=tuple(claim.render() for claim in claims),
            claim_evidence_map={claim.claim_id: claim.evidence_refs for claim in claims},
            findings=findings,
        )


def validate_claim_evidence_map(
    evidence: ExplanationEvidence,
    claim_evidence_map: dict[str, tuple[str, ...] | list[str]],
) -> tuple[ExplanationFinding, ...]:
    """Validate that an external verbalizer mapped every supported claim to real refs."""

    findings: list[ExplanationFinding] = list(evidence.validate())
    supported_claims = {claim.claim_id: claim for claim in evidence.supported_claims()}
    evidence_refs = {record.ref_id for record in evidence.evidence_records}
    for claim_id, claim in supported_claims.items():
        mapped_refs = tuple(claim_evidence_map.get(claim_id, ()))
        if not mapped_refs:
            findings.append(
                ExplanationFinding(
                    code="missing_claim_evidence_mapping",
                    claim_id=claim_id,
                    message="Supported verbalized claim must map to one or more evidence references.",
                )
            )
            continue
        for evidence_ref in mapped_refs:
            if evidence_ref not in claim.evidence_refs or evidence_ref not in evidence_refs:
                findings.append(
                    ExplanationFinding(
                        code="invalid_claim_evidence_mapping",
                        claim_id=claim_id,
                        message=f"Mapped evidence reference {evidence_ref!r} is not valid for this claim.",
                        evidence_ref=evidence_ref,
                    )
                )
    for claim_id, mapped_refs in claim_evidence_map.items():
        if claim_id in supported_claims:
            continue
        findings.append(
            ExplanationFinding(
                code="unknown_claim_mapping",
                claim_id=claim_id,
                message="Verbalized output references a claim not present in supported evidence.",
                evidence_ref=",".join(mapped_refs),
            )
        )
    return tuple(findings)
