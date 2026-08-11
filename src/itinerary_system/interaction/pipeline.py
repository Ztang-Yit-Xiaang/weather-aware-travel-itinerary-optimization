"""Narrow immutable package entry point for permission-aware repair sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..pipeline_runner import PipelineExecutor, PipelineRun, run_research_pipeline
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .controller import PermissionAwareClarificationController
from .models import (
    ClarificationMode,
    InteractionArtifacts,
    InteractionOptions,
    InteractionRequest,
    UserPermissionDecision,
)


@dataclass(frozen=True)
class PermissionAwarePipelineRun:
    interaction_run_id: str
    status: str
    output_dir: Path
    manifest_path: Path
    metrics_path: Path
    authorized_run: PipelineRun | None = None
    schema_version: str = "permission-aware-pipeline-run-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "interaction_run_id": self.interaction_run_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "authorized_run": self.authorized_run.to_record() if self.authorized_run else None,
            "schema_version": self.schema_version,
        }


def run_permission_aware_research_pipeline(
    *,
    config_path: str | Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan: PlanArtifactV2,
    interaction_request: InteractionRequest,
    controller: PermissionAwareClarificationController | None,
    interaction_options: InteractionOptions | None = None,
    permission_decisions: tuple[UserPermissionDecision, ...] = (),
    authorized_executor_factory: Any | None = None,
    disabled_executor: PipelineExecutor | None = None,
    refresh_policy: str = "never",
    run_id: str | None = None,
    output_root: str | Path = "runs",
    strict: bool = True,
    config_overrides: dict[str, Any] | None = None,
    data_bundle: Any | None = None,
) -> PermissionAwarePipelineRun | PipelineRun:
    """Run a clarification session or delegate unchanged when the gate is off.

    The disabled branch calls the existing pipeline directly and creates no
    interaction directories or manifest fields.  Enabled sessions write an
    immutable interaction run.  If an interpretation is authorized, a second
    immutable continuation run calls the existing pipeline and repair executor.
    """

    resolved_options = interaction_options or InteractionOptions()
    if resolved_options.clarification_mode == ClarificationMode.DISABLED:
        if disabled_executor is None:
            raise ValueError("disabled clarification mode requires the existing pipeline executor")
        return run_research_pipeline(
            config_path=config_path,
            catalog_snapshot_id=catalog_snapshot_id,
            context_snapshot_id=context_snapshot_id,
            parent_plan_id=parent_plan.plan_id,
            repair_request_id=interaction_request.repair_session_id,
            refresh_policy=refresh_policy,
            run_id=run_id,
            output_root=output_root,
            executor=disabled_executor,
            strict=strict,
            config_overrides=config_overrides,
            data_bundle=data_bundle,
        )
    if controller is None:
        raise ValueError("enabled clarification mode requires a permission-aware controller")
    resolved_run_id = (
        run_id
        or f"interaction_{stable_content_hash({'session': interaction_request.repair_session_id, 'parent': parent_plan.content_hash})}"
    )
    output_dir = Path(output_root) / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _create_interaction_layout(output_dir)
    session = controller.run(
        parent=parent_plan,
        request=interaction_request,
        options=resolved_options,
        permission_decisions=permission_decisions,
    )
    artifact_paths = _write_interaction_artifacts(output_dir, session.artifacts)
    _write_json(output_dir / "plans" / f"{parent_plan.plan_id}.json", parent_plan.to_record())
    authorized_request = session.authorized_repair_request
    if authorized_request is not None:
        request_path = output_dir / "requests" / f"{authorized_request.request_id}.json"
        _write_json(request_path, _repair_request_record(authorized_request, parent_plan))
        artifact_paths.setdefault("requests", []).append(request_path)

    authorized_run: PipelineRun | None = None
    continuation_error: Exception | None = None
    if authorized_request is not None and authorized_executor_factory is not None:
        continuation_id = f"auth_{stable_content_hash({'request': authorized_request.request_id})}"
        try:
            authorized_run = run_research_pipeline(
                config_path=config_path,
                catalog_snapshot_id=catalog_snapshot_id,
                context_snapshot_id=context_snapshot_id,
                parent_plan_id=parent_plan.plan_id,
                repair_request_id=authorized_request.request_id,
                refresh_policy=refresh_policy,
                run_id=continuation_id,
                output_root=output_root,
                executor=authorized_executor_factory(authorized_request),
                strict=strict,
                config_overrides=config_overrides,
                data_bundle=data_bundle,
            )
        except Exception as exc:
            continuation_error = exc
            authorized_run = getattr(exc, "pipeline_run", None)
    status = session.status
    if authorized_request is not None and authorized_executor_factory is None:
        status = "authorized_repair_ready"
    elif continuation_error is not None:
        status = "authorized_continuation_failed"
    elif authorized_run is not None:
        status = authorized_run.status

    metrics_path = _write_json(output_dir / "metrics" / "interaction_metrics.json", session.metrics)
    artifact_paths.setdefault("metrics", []).append(metrics_path)
    manifest = _interaction_manifest(
        run_id=resolved_run_id,
        status=status,
        output_dir=output_dir,
        parent_plan=parent_plan,
        request=interaction_request,
        options=resolved_options,
        artifacts=session.artifacts,
        artifact_paths=artifact_paths,
        authorized_request_id=authorized_request.request_id if authorized_request else None,
        authorized_run=authorized_run,
        continuation_error=continuation_error,
        catalog_snapshot_id=catalog_snapshot_id,
        context_snapshot_id=context_snapshot_id,
    )
    manifest_path = _write_json(output_dir / "manifest.json", manifest)
    result = PermissionAwarePipelineRun(
        interaction_run_id=resolved_run_id,
        status=status,
        output_dir=output_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        authorized_run=authorized_run,
    )
    if continuation_error is not None:
        raise continuation_error
    return result


def _create_interaction_layout(output_dir: Path) -> None:
    for relative in (
        "requests",
        "plans",
        "metrics",
        "interpretations",
        "probes/hypothetical_plans",
        "clarifications",
        "permissions",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def _write_interaction_artifacts(output_dir: Path, artifacts: InteractionArtifacts) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {}
    groups = {
        "semantic_candidates": ("interpretations/semantic_candidates.jsonl", artifacts.semantic_candidates),
        "model_patches": ("interpretations/model_patches.jsonl", artifacts.model_patches),
        "probe_requests": ("probes/probe_requests.jsonl", artifacts.probe_requests),
        "probe_results": ("probes/probe_results.jsonl", artifacts.probe_results),
        "consequence_vectors": ("probes/consequence_vectors.jsonl", artifacts.consequence_vectors),
        "critical_tradeoffs": ("clarifications/critical_tradeoffs.jsonl", artifacts.critical_tradeoffs),
        "clarification_decisions": ("clarifications/clarification_decisions.jsonl", artifacts.clarification_decisions),
        "permission_decisions": ("permissions/user_permission_decisions.jsonl", artifacts.permission_decisions),
    }
    for key, (relative, records) in groups.items():
        if not records:
            continue
        path = _write_jsonl(output_dir / relative, records)
        paths[key] = [path]
    questions = tuple(
        {
            "decision_id": decision.decision_id,
            "repair_session_id": decision.repair_session_id,
            "question_text": decision.question_text,
            "evidence_refs": list(decision.evidence_refs),
            "schema_version": "clarification-question-v1",
        }
        for decision in artifacts.clarification_decisions
        if decision.question_text
    )
    if questions:
        paths["questions"] = [_write_jsonl(output_dir / "clarifications/questions.jsonl", questions)]
    hypothetical_paths: list[Path] = []
    for result in artifacts.probe_results:
        if not result.hypothetical_plan_record:
            continue
        path = output_dir / "probes" / "hypothetical_plans" / f"{result.hypothetical_plan_id}.json"
        hypothetical_paths.append(_write_json(path, result.hypothetical_plan_record))
    if hypothetical_paths:
        paths["hypothetical_plans"] = hypothetical_paths
    return paths


def _interaction_manifest(
    *,
    run_id: str,
    status: str,
    output_dir: Path,
    parent_plan: PlanArtifactV2,
    request: InteractionRequest,
    options: InteractionOptions,
    artifacts: InteractionArtifacts,
    artifact_paths: dict[str, list[Path]],
    authorized_request_id: str | None,
    authorized_run: PipelineRun | None,
    continuation_error: Exception | None,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
) -> dict[str, Any]:
    relative_paths = {
        key: [path.relative_to(output_dir).as_posix() for path in paths] for key, paths in artifact_paths.items()
    }
    hashes = {
        path.relative_to(output_dir).as_posix(): _file_sha256(path)
        for paths in artifact_paths.values()
        for path in paths
    }
    question_count = sum(bool(item.question_text) for item in artifacts.clarification_decisions)
    return {
        "run_id": run_id,
        "mode": "permission_aware_repair_interaction",
        "status": status,
        "parent_plan_id": parent_plan.plan_id,
        "parent_content_hash": parent_plan.content_hash,
        "catalog_snapshot_id": catalog_snapshot_id,
        "context_snapshot_id": context_snapshot_id,
        "repair_session_id": request.repair_session_id,
        "continuation_of_session_id": request.continuation_of_session_id,
        "clarification_mode": options.clarification_mode.value,
        "semantic_candidate_source": artifacts.semantic_candidate_source,
        "prompt_or_fixture_version": "rule-based-interaction-v1",
        "number_of_semantic_candidates": len(artifacts.semantic_candidates),
        "number_of_probes": len(artifacts.probe_results),
        "number_of_questions": question_count,
        "permission_decision_ids": [item.permission_decision_id for item in artifacts.permission_decisions],
        "selected_interpretation_id": artifacts.selected_interpretation_id,
        "selected_authorized_repair_request_id": authorized_request_id,
        "authorized_continuation_run_id": authorized_run.run_id if authorized_run else None,
        "continuation_error_class": type(continuation_error).__name__ if continuation_error else None,
        "artifacts": relative_paths,
        "artifact_sha256": hashes,
        "schema_version": "permission-aware-pipeline-manifest-v1",
    }


def _repair_request_record(request: Any, parent: PlanArtifactV2) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "parent_plan_id": parent.plan_id,
        "parent_content_hash": parent.content_hash,
        "user_intent": request.user_intent,
        "evidence_records": list(request.evidence_records),
        "tolerance_profile": dict(request.tolerance_profile),
        "confirmed_constraints": dict(request.confirmed_constraints),
        "candidate_pois": list(request.candidate_pois),
        "travel_graph": dict(request.travel_graph),
        "parsed_intent": asdict(request.parsed_intent) if request.parsed_intent else None,
        "schema_version": "authorized-repair-request-v1",
    }


def _write_json(path: Path, record: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonish(record), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: tuple) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonish(_record(item)), sort_keys=True, default=str) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _record(item: Any) -> Any:
    if hasattr(item, "to_record"):
        return item.to_record()
    return item


def _jsonish(value: Any) -> Any:
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonish(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
