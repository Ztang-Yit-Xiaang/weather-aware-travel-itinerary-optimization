"""Run or replay a frozen-fixture or rule-based permission-aware clarification session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..pipeline_runner import build_progressive_repair_executor
from ..plans.repository import load_plan
from ..routing.matrix import load_route_matrix_from_cache
from .controller import PermissionAwareClarificationController
from .frozen_probe import FrozenCounterfactualProbeExecutor
from .models import (
    ClarificationMode,
    InteractionOptions,
    InteractionRequest,
    UserPermissionDecision,
)
from .pipeline import run_permission_aware_research_pipeline
from .semantic_candidates import FrozenSemanticCandidateProvider, RuleBasedSemanticCandidateProvider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--user-edit", required=True)
    parser.add_argument("--repair-session-id", required=True)
    parser.add_argument("--semantic-candidates", default=None)
    parser.add_argument("--probe-results", required=True)
    parser.add_argument(
        "--clarification-mode",
        choices=(ClarificationMode.FROZEN_FIXTURE.value, ClarificationMode.RULE_BASED_CONSEQUENCE.value),
        default=ClarificationMode.FROZEN_FIXTURE.value,
    )
    parser.add_argument("--selected-interpretation-id", default=None)
    parser.add_argument("--permission-decisions", default=None)
    parser.add_argument("--continuation-of-session-id", default=None)
    parser.add_argument("--question-count", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--probe-time-limit-seconds", type=float, default=2.0)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--catalog-snapshot-id", default=None)
    parser.add_argument("--context-snapshot-id", default=None)
    parser.add_argument("--route-cache", default=None)
    parser.add_argument("--route-source-bundle-id", default="")
    parser.add_argument("--route-source-sha256", default="")
    parser.add_argument("--publication-mode", action="store_true")
    parser.add_argument("--permissive", action="store_true")
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace):
    parent = load_plan(args.parent_plan)
    catalog_snapshot_id = args.catalog_snapshot_id or parent.catalog_snapshot_id
    context_snapshot_id = args.context_snapshot_id or parent.context_snapshot_id
    if not catalog_snapshot_id or not context_snapshot_id:
        raise ValueError("catalog and context snapshot IDs must be supplied by the parent plan or CLI")
    clarification_mode = ClarificationMode(args.clarification_mode)
    if clarification_mode == ClarificationMode.FROZEN_FIXTURE:
        if not args.semantic_candidates:
            raise ValueError("frozen_fixture mode requires --semantic-candidates")
        provider = FrozenSemanticCandidateProvider(args.semantic_candidates)
    else:
        provider = RuleBasedSemanticCandidateProvider()
    probe_executor = FrozenCounterfactualProbeExecutor(args.probe_results)
    controller = PermissionAwareClarificationController(
        candidate_provider=provider,
        probe_executor=probe_executor,  # type: ignore[arg-type]
    )
    interaction_request = InteractionRequest(
        repair_session_id=args.repair_session_id,
        parent_plan_id=parent.plan_id,
        user_edit=args.user_edit,
        selected_interpretation_id=args.selected_interpretation_id,
        question_count=args.question_count,
        continuation_of_session_id=args.continuation_of_session_id,
    )
    permissions = _permission_decisions(args.permission_decisions)
    factory = _authorized_executor_factory(args, parent, str(context_snapshot_id))
    return run_permission_aware_research_pipeline(
        config_path=args.config,
        catalog_snapshot_id=str(catalog_snapshot_id),
        context_snapshot_id=str(context_snapshot_id),
        parent_plan=parent,
        interaction_request=interaction_request,
        controller=controller,
        interaction_options=InteractionOptions(
            clarification_mode=clarification_mode,
            max_questions=args.max_questions,
            probe_time_limit_seconds=args.probe_time_limit_seconds,
        ),
        permission_decisions=permissions,
        authorized_executor_factory=factory,
        run_id=args.run_id,
        output_root=args.output_root,
        strict=not args.permissive,
    )


def _authorized_executor_factory(args: argparse.Namespace, parent, context_snapshot_id: str):
    if not args.route_cache:
        return None
    matrix = load_route_matrix_from_cache(
        Path(args.route_cache),
        context_snapshot_id,
        source_bundle_id=args.route_source_bundle_id,
        expected_source_sha256=args.route_source_sha256,
    )

    def factory(repair_request):
        return build_progressive_repair_executor(
            parent_plan=parent,
            repair_request=repair_request,
            route_matrix=matrix,
            publication_mode=args.publication_mode,
        )

    return factory


def _permission_decisions(path: str | None) -> tuple[UserPermissionDecision, ...]:
    if not path:
        return ()
    fixture_path = Path(path)
    text = fixture_path.read_text(encoding="utf-8")
    if fixture_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        records = raw.get("permission_decisions", ()) if isinstance(raw, dict) else raw
    return tuple(UserPermissionDecision(**record) for record in records)


def main(argv: list[str] | None = None) -> int:
    try:
        run = run_from_args(parse_args(argv))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"FAILED permission-aware repair: {exc}")
        return 2
    print(json.dumps(run.to_record(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
