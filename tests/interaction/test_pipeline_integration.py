from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from itinerary_system.benchmark.publication import REQUIRED_PUBLICATION_METHOD_IDS
from itinerary_system.interaction.controller import PermissionAwareClarificationController
from itinerary_system.interaction.frozen_probe import FrozenCounterfactualProbeExecutor
from itinerary_system.interaction.models import (
    ClarificationMode,
    InteractionOptions,
    InteractionRequest,
    PermissionDecisionAction,
    UserPermissionDecision,
)
from itinerary_system.interaction.pipeline import (
    PermissionAwarePipelineRun,
    run_permission_aware_research_pipeline,
)
from itinerary_system.interaction.semantic_candidates import FrozenSemanticCandidateProvider
from itinerary_system.pipeline_runner import PipelineExecutionResult, PipelineRun
from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
)
from itinerary_system.research_artifacts import PlanArtifactV2

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default_trip_config.yaml"


def parent_plan() -> PlanArtifactV2:
    booked = OwnedConstraint(
        constraint_id="booked_hotel",
        origin=ConstraintOrigin.BOOKING,
        strength=ConstraintStrength.BOOKED,
        scope=ConstraintScope.LODGING,
        target_id="hotel_a",
        relation="preserve_booking",
        value=True,
        confirmed=True,
        relaxation_policy=RelaxationPolicy.EXPLICIT_ONLY,
        evidence_refs=("booking:hotel_a",),
    )
    return PlanArtifactV2(
        plan_id="parent_interaction",
        source_run_id="parent_run",
        planning_request_id="parent_request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=(
            {"stop_id": "view", "name": "Viewpoint", "day": 1},
            {"stop_id": "hotel_a", "name": "Booked Hotel", "day": 1},
        ),
        day_assignments={"view": 1, "hotel_a": 1},
        sequence=("view", "hotel_a"),
        lodging_assignments={"1": "hotel_a"},
        ordered_days=({"day": 1, "stop_ids": ("view", "hotel_a")},),
        owned_constraints=(booked.to_record(),),
        modeled_metrics={"travel_minutes": 180.0, "monetary_cost": 0.0},
    )


def write_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(
            {
                "semantic_candidates": [
                    {
                        "interpretation_id": "keep_hotel",
                        "interpretation_type": "reduce_driving_burden",
                        "target_ids": ["view"],
                        "normalized_parameters": {"target_minutes_reduction": 30.0},
                        "evidence_refs": ["user_edit:hash"],
                    },
                    {
                        "interpretation_id": "change_hotel",
                        "interpretation_type": "lodging_change",
                        "target_ids": ["hotel_a"],
                        "normalized_parameters": {
                            "replacement_lodging_id": "hotel_b",
                            "day": 1,
                            "monetary_cost_delta": 80.0,
                        },
                        "evidence_refs": ["booking:hotel_a", "user_edit:hash"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    keep_hypothetical = replace(
        parent_plan(),
        plan_id="hypothetical_keep",
        parent_plan_id="parent_interaction",
        source_run_id="probe_keep",
        certificate_id=None,
        modeled_metrics={"travel_minutes": 360.0, "monetary_cost": 0.0},
    ).to_record()
    change_hypothetical = replace(
        parent_plan(),
        plan_id="hypothetical_change",
        parent_plan_id="parent_interaction",
        source_run_id="probe_change",
        lodging_assignments={"1": "hotel_b"},
        certificate_id=None,
        modeled_metrics={"travel_minutes": 180.0, "monetary_cost": 80.0},
    ).to_record()
    probe_path = tmp_path / "probes.json"
    probe_path.write_text(
        json.dumps(
            {
                "probe_results": [
                    {
                        "interpretation_id": "keep_hotel",
                        "status": "feasible_bounded",
                        "hypothetical_plan_id": "hypothetical_keep",
                        "hypothetical_plan_record": keep_hypothetical,
                        "diff_record": {
                            "diff_id": "diff_keep",
                            "weighted_edit_cost": 2.0,
                            "deleted_stops": [{"stop_id": "view", "day": 1, "cost": 2.0}],
                        },
                        "solver_run_ids": ["solver_keep"],
                        "runtime_seconds": 0.1,
                        "accepted_repair_radius": "hotel_preserving_reroute",
                        "evidence_refs": ["probe:keep", "route:matrix", "plan_diff:diff_keep"],
                    },
                    {
                        "interpretation_id": "change_hotel",
                        "status": "feasible_bounded",
                        "hypothetical_plan_id": "hypothetical_change",
                        "hypothetical_plan_record": change_hypothetical,
                        "diff_record": {
                            "diff_id": "diff_change",
                            "weighted_edit_cost": 5.0,
                            "lodging_changes": [
                                {
                                    "day": 1,
                                    "from_lodging_id": "hotel_a",
                                    "to_lodging_id": "hotel_b",
                                    "owner_strength": "booked",
                                    "cost": 5.0,
                                }
                            ],
                        },
                        "requires_user_permission": True,
                        "permission_constraint_ids": ["booked_hotel"],
                        "solver_run_ids": ["solver_change"],
                        "runtime_seconds": 0.1,
                        "accepted_repair_radius": "hotel_changing_repair",
                        "evidence_refs": [
                            "probe:change",
                            "route:matrix",
                            "plan_diff:diff_change",
                            "cost:booking",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return semantic_path, probe_path


def controller(semantic_path: Path, probe_path: Path) -> PermissionAwareClarificationController:
    return PermissionAwareClarificationController(
        candidate_provider=FrozenSemanticCandidateProvider(semantic_path),
        probe_executor=FrozenCounterfactualProbeExecutor(probe_path),  # type: ignore[arg-type]
    )


def test_initial_permission_run_pauses_and_keeps_hypothetical_out_of_plan_store(tmp_path: Path) -> None:
    semantic_path, probe_path = write_fixtures(tmp_path)
    run = run_permission_aware_research_pipeline(
        config_path=CONFIG,
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        parent_plan=parent_plan(),
        interaction_request=InteractionRequest(
            repair_session_id="session_initial",
            parent_plan_id="parent_interaction",
            user_edit="Make the Yosemite day easier",
        ),
        controller=controller(semantic_path, probe_path),
        interaction_options=InteractionOptions(clarification_mode=ClarificationMode.FROZEN_FIXTURE),
        output_root=tmp_path / "runs",
        run_id="interaction_initial",
    )
    assert isinstance(run, PermissionAwarePipelineRun)
    assert run.status == "confirmation_required"
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["number_of_probes"] == 2
    assert manifest["number_of_questions"] == 1
    assert manifest["selected_authorized_repair_request_id"] is None
    assert not (run.output_dir / "evaluations").exists()
    assert {path.name for path in (run.output_dir / "plans").glob("*.json")} == {"parent_interaction.json"}
    hypothetical = json.loads(
        (run.output_dir / "probes" / "hypothetical_plans" / "hypothetical_change.json").read_text(encoding="utf-8")
    )
    assert hypothetical["eligible_for_execution"] is False
    assert hypothetical.get("certificate_id") is None


def test_grant_once_creates_authorized_continuation_and_preserves_parent(tmp_path: Path) -> None:
    semantic_path, probe_path = write_fixtures(tmp_path)
    parent = parent_plan()
    parent_hash = parent.content_hash
    permission = UserPermissionDecision(
        permission_decision_id="permission_once",
        repair_session_id="session_continue",
        constraint_ids=("booked_hotel",),
        action=PermissionDecisionAction.GRANT_ONCE,
        selected_interpretation_id="change_hotel",
        created_at="2026-07-22T12:00:00+00:00",
        evidence_refs=("user_decision:permission_once",),
    )

    def factory(repair_request):
        child = replace(
            parent,
            plan_id="authorized_child",
            parent_plan_id=parent.plan_id,
            source_run_id="authorized_planner_run",
            planning_request_id=repair_request.request_id,
            lodging_assignments={"1": "hotel_b"},
            certificate_id="certificate_authorized",
        )

        def execute(_context):
            return PipelineExecutionResult(
                parent_plan=parent,
                output_plans=(child,),
                evaluations=(
                    {
                        "evaluation_id": "evaluation_authorized",
                        "certificate_id": "certificate_authorized",
                        "plan_id": child.plan_id,
                        "eligible": True,
                        "comparison_eligibility": "eligible",
                        "evaluation_status": "passed",
                    },
                ),
                request_records=(
                    {
                        "request_id": repair_request.request_id,
                        "parent_plan_id": parent.plan_id,
                        "test_only": False,
                    },
                ),
                metrics={"authorized_repair": True},
            )

        return execute

    run = run_permission_aware_research_pipeline(
        config_path=CONFIG,
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        parent_plan=parent,
        interaction_request=InteractionRequest(
            repair_session_id="session_continue",
            parent_plan_id=parent.plan_id,
            user_edit="Allow the hotel change",
            selected_interpretation_id="change_hotel",
            question_count=1,
            continuation_of_session_id="session_initial",
        ),
        controller=controller(semantic_path, probe_path),
        interaction_options=InteractionOptions(clarification_mode=ClarificationMode.FROZEN_FIXTURE),
        permission_decisions=(permission,),
        authorized_executor_factory=factory,
        output_root=tmp_path / "runs",
        run_id="interaction_continue",
    )
    assert isinstance(run, PermissionAwarePipelineRun)
    assert run.authorized_run is not None
    assert run.authorized_run.status == "completed"
    assert run.authorized_run.output_plans[0].parent_plan_id == parent.plan_id
    assert run.authorized_run.output_plans[0].certificate_id == "certificate_authorized"
    assert parent.content_hash == parent_hash
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_interpretation_id"] == "change_hotel"
    assert manifest["selected_authorized_repair_request_id"].startswith("authorized_repair_")
    assert manifest["authorized_continuation_run_id"].startswith("auth_")


def test_disabled_mode_delegates_without_interaction_artifacts_or_e3_changes(tmp_path: Path) -> None:
    parent = parent_plan()

    def existing_executor(_context):
        return PipelineExecutionResult(parent_plan=parent)

    run = run_permission_aware_research_pipeline(
        config_path=CONFIG,
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        parent_plan=parent,
        interaction_request=InteractionRequest(
            repair_session_id="disabled_session",
            parent_plan_id=parent.plan_id,
            user_edit="ignored while disabled",
        ),
        controller=None,
        interaction_options=InteractionOptions(),
        disabled_executor=existing_executor,
        output_root=tmp_path / "runs",
        run_id="disabled_run",
    )
    assert isinstance(run, PipelineRun)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert "clarification_mode" not in manifest
    assert not (run.output_dir / "interpretations").exists()
    assert REQUIRED_PUBLICATION_METHOD_IDS == (
        "context_blind_solver",
        "deterministic_context_aware_heuristic",
        "progressive_sequential_lexicographic_repair",
        "full_reoptimization",
    )


def test_permission_denied_continuation_keeps_hotel_and_receives_certificate(tmp_path: Path) -> None:
    semantic_path, probe_path = write_fixtures(tmp_path)
    parent = parent_plan()
    denial = UserPermissionDecision(
        permission_decision_id="permission_denied",
        repair_session_id="session_denied",
        constraint_ids=("booked_hotel",),
        action=PermissionDecisionAction.DENY,
        selected_interpretation_id="change_hotel",
        created_at="2026-07-22T12:00:00+00:00",
        evidence_refs=("user_decision:permission_denied",),
    )

    def factory(repair_request):
        child = replace(
            parent,
            plan_id="authorized_keep_hotel",
            parent_plan_id=parent.plan_id,
            source_run_id="authorized_keep_run",
            planning_request_id=repair_request.request_id,
            certificate_id="certificate_keep_hotel",
        )

        def execute(_context):
            return PipelineExecutionResult(
                parent_plan=parent,
                output_plans=(child,),
                evaluations=(
                    {
                        "evaluation_id": "evaluation_keep_hotel",
                        "certificate_id": "certificate_keep_hotel",
                        "plan_id": child.plan_id,
                        "eligible": True,
                        "comparison_eligibility": "eligible",
                        "evaluation_status": "passed",
                    },
                ),
                request_records=(
                    {
                        "request_id": repair_request.request_id,
                        "parent_plan_id": parent.plan_id,
                        "test_only": False,
                    },
                ),
            )

        return execute

    run = run_permission_aware_research_pipeline(
        config_path=CONFIG,
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        parent_plan=parent,
        interaction_request=InteractionRequest(
            repair_session_id="session_denied",
            parent_plan_id=parent.plan_id,
            user_edit="Keep the booked hotel",
            selected_interpretation_id="keep_hotel",
            question_count=1,
            continuation_of_session_id="session_initial",
        ),
        controller=controller(semantic_path, probe_path),
        interaction_options=InteractionOptions(clarification_mode=ClarificationMode.FROZEN_FIXTURE),
        permission_decisions=(denial,),
        authorized_executor_factory=factory,
        output_root=tmp_path / "runs_denied",
        run_id="interaction_denied",
    )
    assert isinstance(run, PermissionAwarePipelineRun)
    assert run.authorized_run is not None
    child = run.authorized_run.output_plans[0]
    assert child.lodging_assignments == parent.lodging_assignments
    assert child.certificate_id == "certificate_keep_hotel"
    request_path = next((run.output_dir / "requests").glob("authorized_repair_*.json"))
    request_record = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_record["confirmed_constraints"].get("allow_booked_relaxation") is not True
