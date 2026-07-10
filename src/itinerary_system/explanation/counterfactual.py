"""Sandbox counterfactual requests and explanation evidence builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .evidence import EvidenceRecord, ExplanationClaim, WhatIfEvidence, WhyNotEvidence

CounterfactualQuestionType = Literal["why_not", "what_if"]
CounterfactualExecutor = Callable[["CounterfactualRequest"], Any]


@dataclass(frozen=True)
class CounterfactualRequest:
    """A solver-safe sandbox request derived from an explanation question."""

    request_id: str
    parent_plan_id: str
    parent_content_hash: str
    question_type: CounterfactualQuestionType | str
    target_id: str
    requested_change: str
    source_request_id: str = ""
    force_constraints: dict[str, Any] = field(default_factory=dict)
    sandbox_overrides: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "counterfactual-request-v1"

    def to_solver_payload(self) -> SimpleNamespace:
        confirmed_constraints = {
            "parent_plan_id": self.parent_plan_id,
            "counterfactual": True,
            "counterfactual_request_id": self.request_id,
            "counterfactual_question_type": self.question_type,
            "counterfactual_target_id": self.target_id,
            "counterfactual_requested_change": self.requested_change,
            **self.force_constraints,
            **self.sandbox_overrides,
        }
        return SimpleNamespace(
            request_id=self.request_id,
            parent_plan_id=self.parent_plan_id,
            confirmed_constraints=confirmed_constraints,
            source_request_id=self.source_request_id,
            evidence_refs=self.evidence_refs,
            metadata=self.metadata,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "parent_plan_id": self.parent_plan_id,
            "parent_content_hash": self.parent_content_hash,
            "question_type": self.question_type,
            "target_id": self.target_id,
            "requested_change": self.requested_change,
            "source_request_id": self.source_request_id,
            "force_constraints": _jsonish(self.force_constraints),
            "sandbox_overrides": _jsonish(self.sandbox_overrides),
            "evidence_refs": list(self.evidence_refs),
            "metadata": _jsonish(self.metadata),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class CounterfactualRunRecord:
    """Stored result of one sandbox counterfactual attempt."""

    run_id: str
    sandbox_request_id: str
    parent_plan_id: str
    parent_content_hash: str
    question_type: str
    target_id: str
    requested_change: str
    status: str
    outcome_id: str = ""
    child_plan_id: str = ""
    diff_id: str = ""
    evaluation_id: str = ""
    failure_reasons: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "counterfactual-run-v1"

    @property
    def evidence_ref(self) -> str:
        return f"counterfactual:{self.run_id}"

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sandbox_request_id": self.sandbox_request_id,
            "parent_plan_id": self.parent_plan_id,
            "parent_content_hash": self.parent_content_hash,
            "question_type": self.question_type,
            "target_id": self.target_id,
            "requested_change": self.requested_change,
            "status": self.status,
            "outcome_id": self.outcome_id,
            "child_plan_id": self.child_plan_id,
            "diff_id": self.diff_id,
            "evaluation_id": self.evaluation_id,
            "failure_reasons": list(self.failure_reasons),
            "payload": _jsonish(self.payload),
            "schema_version": self.schema_version,
        }

    def to_evidence_record(self) -> EvidenceRecord:
        return EvidenceRecord(
            ref_id=self.evidence_ref,
            source_type="counterfactual_run",
            source_id=self.run_id,
            field_path="status",
            payload={
                "status": self.status,
                "sandbox_request_id": self.sandbox_request_id,
                "child_plan_id": self.child_plan_id,
                "diff_id": self.diff_id,
                "evaluation_id": self.evaluation_id,
                "failure_reasons": list(self.failure_reasons),
                "record": self.to_record(),
            },
        )


class CounterfactualRunner:
    """Build and execute sandbox explanation counterfactuals through an injected executor."""

    def __init__(self, executor: CounterfactualExecutor | None = None) -> None:
        self.executor = executor

    def build_why_not_request(
        self,
        parent_plan: PlanArtifactV2,
        *,
        target_id: str,
        source_request_id: str = "",
        requested_change: str = "force_include",
        evidence_refs: tuple[str, ...] = (),
    ) -> CounterfactualRequest:
        return build_counterfactual_request(
            parent_plan,
            question_type="why_not",
            target_id=target_id,
            requested_change=requested_change,
            source_request_id=source_request_id,
            force_constraints={
                "must_include": (target_id,),
                "strengthen_target": target_id,
                "strengthened_relation": requested_change,
            },
            evidence_refs=evidence_refs,
        )

    def build_what_if_request(
        self,
        parent_plan: PlanArtifactV2,
        *,
        target_id: str,
        requested_change: str,
        source_request_id: str = "",
        sandbox_overrides: dict[str, Any] | None = None,
        evidence_refs: tuple[str, ...] = (),
    ) -> CounterfactualRequest:
        return build_counterfactual_request(
            parent_plan,
            question_type="what_if",
            target_id=target_id,
            requested_change=requested_change,
            source_request_id=source_request_id,
            force_constraints={
                "what_if_target": target_id,
                "what_if_change": requested_change,
                **(sandbox_overrides or {}),
            },
            sandbox_overrides=sandbox_overrides or {},
            evidence_refs=evidence_refs,
        )

    def run(self, parent_plan: PlanArtifactV2, request: CounterfactualRequest) -> CounterfactualRunRecord:
        before_hash = parent_plan.content_hash
        if self.executor is None:
            return _build_run_record(
                request,
                status="not_evaluated",
                failure_reasons=("counterfactual_executor_missing",),
            )
        try:
            raw_outcome = self.executor(request)
            outcome_record = _record_from_outcome(raw_outcome)
            status = _status_from_record(outcome_record)
            failure_reasons = _failure_reasons_from_record(outcome_record)
        except Exception as exc:
            outcome_record = {"error": f"{type(exc).__name__}: {exc}"}
            status = "failed"
            failure_reasons = (str(outcome_record["error"]),)
        if parent_plan.content_hash != before_hash:
            status = "parent_mutated"
            failure_reasons = tuple(dict.fromkeys((*failure_reasons, "parent_plan_mutated")))
        return _build_run_record(
            request,
            status=status,
            outcome_record=outcome_record,
            failure_reasons=failure_reasons,
        )

    def explain_why_not(
        self,
        parent_plan: PlanArtifactV2,
        *,
        target_id: str,
        source_request_id: str = "",
        requested_change: str = "force_include",
        evidence_refs: tuple[str, ...] = (),
    ) -> WhyNotEvidence:
        request = self.build_why_not_request(
            parent_plan,
            target_id=target_id,
            source_request_id=source_request_id,
            requested_change=requested_change,
            evidence_refs=evidence_refs,
        )
        run = self.run(parent_plan, request)
        evidence_record = run.to_evidence_record()
        claim = ExplanationClaim(
            claim_id=f"claim_{run.run_id}_why_not",
            claim_type="causal",
            text_template="Forcing {target_id} in a sandbox counterfactual ended with status {status}.",
            values={"target_id": target_id, "status": run.status},
            evidence_refs=(evidence_record.ref_id,),
        )
        return WhyNotEvidence(
            evidence_id=f"why_not_{run.run_id}",
            target_id=target_id,
            failed_attempt_ids=(run.run_id,),
            counterfactual_run_ids=(run.run_id,),
            outcome_status=run.status,
            claims=(claim,),
            evidence_records=(evidence_record,),
        )

    def explain_what_if(
        self,
        parent_plan: PlanArtifactV2,
        *,
        target_id: str,
        requested_change: str,
        source_request_id: str = "",
        sandbox_overrides: dict[str, Any] | None = None,
        evidence_refs: tuple[str, ...] = (),
    ) -> WhatIfEvidence:
        request = self.build_what_if_request(
            parent_plan,
            target_id=target_id,
            requested_change=requested_change,
            source_request_id=source_request_id,
            sandbox_overrides=sandbox_overrides,
            evidence_refs=evidence_refs,
        )
        run = self.run(parent_plan, request)
        evidence_record = run.to_evidence_record()
        claim = ExplanationClaim(
            claim_id=f"claim_{run.run_id}_what_if",
            claim_type="causal",
            text_template="The sandbox change {requested_change} for {target_id} ended with status {status}.",
            values={"requested_change": requested_change, "target_id": target_id, "status": run.status},
            evidence_refs=(evidence_record.ref_id,),
        )
        return WhatIfEvidence(
            evidence_id=f"what_if_{run.run_id}",
            sandbox_request_id=request.request_id,
            counterfactual_run_ids=(run.run_id,),
            outcome_status=run.status,
            claims=(claim,),
            evidence_records=(evidence_record,),
        )


def build_counterfactual_request(
    parent_plan: PlanArtifactV2,
    *,
    question_type: CounterfactualQuestionType | str,
    target_id: str,
    requested_change: str,
    source_request_id: str = "",
    force_constraints: dict[str, Any] | None = None,
    sandbox_overrides: dict[str, Any] | None = None,
    evidence_refs: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> CounterfactualRequest:
    request_seed = {
        "parent_plan_id": parent_plan.plan_id,
        "parent_content_hash": parent_plan.content_hash,
        "question_type": question_type,
        "target_id": target_id,
        "requested_change": requested_change,
        "source_request_id": source_request_id,
        "force_constraints": _jsonish(force_constraints or {}),
        "sandbox_overrides": _jsonish(sandbox_overrides or {}),
    }
    return CounterfactualRequest(
        request_id=f"sandbox_cf_{stable_content_hash(request_seed)}",
        parent_plan_id=parent_plan.plan_id,
        parent_content_hash=parent_plan.content_hash,
        question_type=question_type,
        target_id=target_id,
        requested_change=requested_change,
        source_request_id=source_request_id,
        force_constraints=force_constraints or {},
        sandbox_overrides=sandbox_overrides or {},
        evidence_refs=evidence_refs,
        metadata=metadata or {},
    )


def _build_run_record(
    request: CounterfactualRequest,
    *,
    status: str,
    outcome_record: dict[str, Any] | None = None,
    failure_reasons: tuple[str, ...] = (),
) -> CounterfactualRunRecord:
    outcome = outcome_record or {}
    run_seed = {
        "request": request.to_record(),
        "status": status,
        "outcome_id": outcome.get("repair_outcome_id") or outcome.get("outcome_id") or "",
        "child_plan_id": outcome.get("child_plan_id") or "",
        "failure_reasons": list(failure_reasons),
    }
    return CounterfactualRunRecord(
        run_id=f"cf_run_{stable_content_hash(run_seed)}",
        sandbox_request_id=request.request_id,
        parent_plan_id=request.parent_plan_id,
        parent_content_hash=request.parent_content_hash,
        question_type=str(request.question_type),
        target_id=request.target_id,
        requested_change=request.requested_change,
        status=status,
        outcome_id=str(outcome.get("repair_outcome_id") or outcome.get("outcome_id") or ""),
        child_plan_id=str(outcome.get("child_plan_id") or ""),
        diff_id=str(outcome.get("diff_id") or ""),
        evaluation_id=str(outcome.get("evaluation_id") or ""),
        failure_reasons=failure_reasons,
        payload={"request": request.to_record(), "outcome": _jsonish(outcome)},
    )


def _record_from_outcome(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_record"):
        return dict(raw.to_record())
    record: dict[str, Any] = {}
    for key in (
        "repair_outcome_id",
        "outcome_id",
        "status",
        "child_plan_id",
        "diff_id",
        "evaluation_id",
        "diagnosis",
    ):
        if hasattr(raw, key):
            value = getattr(raw, key)
            record[key] = value.to_record() if hasattr(value, "to_record") else value
    child_plan = getattr(raw, "child_plan", None)
    if child_plan is not None and hasattr(child_plan, "plan_id"):
        record.setdefault("child_plan_id", child_plan.plan_id)
    diff_record = getattr(raw, "diff_record", None)
    if isinstance(diff_record, dict):
        record.setdefault("diff_id", diff_record.get("diff_id", ""))
    evaluation_record = getattr(raw, "evaluation_record", None)
    if isinstance(evaluation_record, dict):
        record.setdefault("evaluation_id", evaluation_record.get("evaluation_id", ""))
    return record


def _status_from_record(record: dict[str, Any]) -> str:
    status = str(record.get("status") or record.get("comparison_eligibility") or "").lower()
    if status in {"accepted", "eligible", "passed", "success"}:
        return "accepted"
    if status in {"infeasible", "ineligible", "failed", "failure"}:
        return "infeasible" if status in {"infeasible", "ineligible"} else "failed"
    if record.get("child_plan_id"):
        return "accepted"
    return "unknown"


def _failure_reasons_from_record(record: dict[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    raw_reasons = record.get("failure_reasons")
    if isinstance(raw_reasons, (list, tuple)):
        reasons.extend(str(reason) for reason in raw_reasons)
    diagnosis = record.get("diagnosis")
    if isinstance(diagnosis, dict):
        raw_diagnosis_reasons = diagnosis.get("failure_reasons")
        if isinstance(raw_diagnosis_reasons, (list, tuple)):
            reasons.extend(str(reason) for reason in raw_diagnosis_reasons)
    attempts = record.get("attempts")
    if isinstance(attempts, (list, tuple)):
        for attempt in attempts:
            if isinstance(attempt, dict) and isinstance(attempt.get("failure_reasons"), (list, tuple)):
                reasons.extend(str(reason) for reason in attempt["failure_reasons"])
    if record.get("error"):
        reasons.append(str(record["error"]))
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonish(item) for item in value]
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    if hasattr(value, "to_record"):
        return _jsonish(value.to_record())
    return value
