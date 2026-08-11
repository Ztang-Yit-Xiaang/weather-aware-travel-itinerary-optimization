"""Replay counterfactual probe results from deterministic JSON fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..research_artifacts import stable_content_hash
from .models import CounterfactualProbeResult, ProbeStatus


class FrozenCounterfactualProbeExecutor:
    source_name = "frozen_probe_fixture"

    def __init__(self, fixture_path: str | Path) -> None:
        self.records = _read_records(Path(fixture_path))

    def execute(self, *, parent, candidate, patch, request, assessment) -> CounterfactualProbeResult:
        del parent
        record = next(
            (
                item
                for item in self.records
                if str(item.get("interpretation_id", "")) == candidate.interpretation_id
                or str(item.get("probe_request_id", "")) == request.probe_request_id
            ),
            None,
        )
        if record is None:
            raise ValueError(f"no frozen probe result for interpretation {candidate.interpretation_id}")
        status = ProbeStatus(str(record.get("status", "failed")))
        diff = dict(record["diff_record"]) if isinstance(record.get("diff_record"), dict) else None
        hypothetical = (
            dict(record["hypothetical_plan_record"])
            if isinstance(record.get("hypothetical_plan_record"), dict)
            else None
        )
        if hypothetical is not None:
            hypothetical["artifact_role"] = "hypothetical"
            hypothetical["test_only"] = True
            hypothetical["eligible_for_execution"] = False
            hypothetical["certificate_id"] = None
        hypothetical_id = str(record.get("hypothetical_plan_id") or (hypothetical or {}).get("plan_id") or "") or None
        result_seed = {
            "request": request.probe_request_id,
            "interpretation": candidate.interpretation_id,
            "status": status.value,
            "diff": diff,
        }
        evidence_refs = tuple(
            dict.fromkeys(
                (*record.get("evidence_refs", (f"fixture:{request.probe_request_id}",)),)
                + ((f"hypothetical_plan:{hypothetical_id}",) if hypothetical_id else ())
                + ((f"plan_diff:{diff.get('diff_id')}",) if diff and diff.get("diff_id") else ())
            )
        )
        return CounterfactualProbeResult(
            probe_result_id=str(record.get("probe_result_id") or f"probe_result_{stable_content_hash(result_seed)}"),
            probe_request_id=request.probe_request_id,
            parent_plan_id=request.parent_plan_id,
            hypothetical_plan_id=hypothetical_id,
            status=status,
            diff_id=str(record.get("diff_id") or (diff or {}).get("diff_id") or "") or None,
            solver_run_ids=tuple(str(item) for item in record.get("solver_run_ids", ())),
            requires_user_permission=bool(record.get("requires_user_permission", assessment.requires_user_permission)),
            permission_constraint_ids=tuple(
                str(item) for item in record.get("permission_constraint_ids", assessment.permission_constraint_ids)
            ),
            eligible_for_execution=False,
            evidence_refs=tuple(str(item) for item in evidence_refs),
            interpretation_id=candidate.interpretation_id,
            affected_constraint_ids=tuple(
                str(item) for item in record.get("affected_constraint_ids", patch.affected_constraint_ids)
            ),
            runtime_seconds=_optional_float(record.get("runtime_seconds")),
            solver_bound=_optional_float(record.get("solver_bound")),
            solver_gap=_optional_float(record.get("solver_gap")),
            accepted_repair_radius=record.get("accepted_repair_radius"),
            failure_reasons=tuple(str(item) for item in record.get("failure_reasons", ())),
            diff_record=diff,
            hypothetical_plan_record=hypothetical,
            diagnostic_evaluation=(
                dict(record["diagnostic_evaluation"]) if isinstance(record.get("diagnostic_evaluation"), dict) else None
            ),
        )


def _read_records(path: Path) -> tuple[dict[str, Any], ...]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("probe_results", ())
    if not isinstance(raw, list):
        raise ValueError("frozen probe fixture must contain a list")
    return tuple(dict(item) for item in raw)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
