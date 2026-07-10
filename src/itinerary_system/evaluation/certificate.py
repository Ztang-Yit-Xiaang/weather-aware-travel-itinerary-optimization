"""Independent final-plan evaluation certificate records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..research_artifacts import PlanArtifactV2, stable_content_hash

EvaluationStatus = Literal["PASSED", "PASSED_WITH_WARNINGS", "FAILED"]
EligibilityStatus = Literal["eligible", "ineligible"]
FindingSeverity = Literal["failure", "warning"]


@dataclass(frozen=True)
class EvaluationFinding:
    """One independent evaluator finding, separated by severity."""

    code: str
    message: str
    severity: FindingSeverity
    category: str
    evidence_refs: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanEvaluationCertificate:
    """Certificate tying evaluation eligibility to one exact plan content hash."""

    certificate_id: str
    plan_id: str
    source_run_id: str
    plan_content_hash: str
    evaluator_version: str
    artifact_grounding_status: EvaluationStatus
    hard_feasibility_status: EvaluationStatus
    evaluation_status: EvaluationStatus
    comparison_eligibility: EligibilityStatus
    failures: tuple[EvaluationFinding, ...] = ()
    warnings: tuple[EvaluationFinding, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    route_validation: dict[str, Any] | None = None
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "plan-evaluation-certificate-v1"

    @property
    def eligible(self) -> bool:
        return self.comparison_eligibility == "eligible"

    @property
    def reward_eligibility(self) -> bool:
        return self.eligible

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_record(include_content_hash=False))

    def valid_for_plan(self, plan: PlanArtifactV2) -> bool:
        """Return whether this certificate still applies to the exact plan content."""

        return self.plan_id == plan.plan_id and self.plan_content_hash == plan.content_hash

    def to_record(self, *, include_content_hash: bool = True) -> dict[str, Any]:
        record = {
            "certificate_id": self.certificate_id,
            "evaluation_id": self.certificate_id,
            "plan_id": self.plan_id,
            "source_run_id": self.source_run_id,
            "plan_content_hash": self.plan_content_hash,
            "evaluator_version": self.evaluator_version,
            "artifact_grounding_status": self.artifact_grounding_status,
            "hard_feasibility_status": self.hard_feasibility_status,
            "evaluation_status": self.evaluation_status,
            "comparison_eligibility": self.comparison_eligibility,
            "eligible": self.eligible,
            "reward_eligibility": self.reward_eligibility,
            "failures": [finding.to_record() for finding in self.failures],
            "warnings": [finding.to_record() for finding in self.warnings],
            "failure_count": len(self.failures),
            "warning_count": len(self.warnings),
            "blocking_warning_count": len(self.failures),
            "nonblocking_warning_count": len(self.warnings),
            "metrics": self.metrics,
            "route_validation": self.route_validation,
            "evaluated_at": self.evaluated_at,
            "schema_version": self.schema_version,
        }
        if include_content_hash:
            record["content_hash"] = self.content_hash
        return record
