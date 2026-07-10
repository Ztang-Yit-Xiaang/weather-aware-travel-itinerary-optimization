from __future__ import annotations

from itinerary_system.evaluation import PlanEvaluationCertificate
from itinerary_system.explanation import ExplanationEvidenceBuilder, build_explanation_evidence
from itinerary_system.plans import PlanDiff, StopChange
from itinerary_system.research_artifacts import PlanArtifactV2


def plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="child_explain",
        parent_plan_id="parent_explain",
        source_run_id="run_explain",
        planning_request_id="request_explain",
        catalog_snapshot_id="catalog_explain",
        context_snapshot_id="context_explain",
        selected_stops=({"stop_id": "poi_a", "day": 1, "stop_order": 1},),
        day_assignments={"poi_a": 1},
        sequence=("poi_a",),
        created_at="2026-07-09T00:00:00+00:00",
    )


def diff() -> PlanDiff:
    return PlanDiff(
        diff_id="diff_explain",
        parent_plan_id="parent_explain",
        child_plan_id="child_explain",
        added_stops=(StopChange(stop_id="poi_a", day=1, order=1, owner_strength="soft", cost=10.0),),
        deleted_stops=(),
        day_moves=(),
        time_shifts=(),
        reorder_changes=(),
        lodging_changes=(),
        road_changes=(),
        unchanged_days=(),
        weighted_edit_cost=10.0,
    )


def certificate(target_plan: PlanArtifactV2) -> PlanEvaluationCertificate:
    return PlanEvaluationCertificate(
        certificate_id="cert_explain",
        plan_id=target_plan.plan_id,
        source_run_id=target_plan.source_run_id,
        plan_content_hash=target_plan.content_hash,
        evaluator_version="verify-001-test",
        artifact_grounding_status="PASSED",
        hard_feasibility_status="PASSED",
        evaluation_status="PASSED",
        comparison_eligibility="eligible",
        metrics={"route_required_leg_count": 2.0},
        route_validation={"matrix_id": "matrix_explain", "road_validated": True},
        evaluated_at="2026-07-09T00:00:00+00:00",
    )


def test_build_explanation_evidence_links_diff_and_certificate_records():
    child = plan()
    evidence = build_explanation_evidence(child, diff(), certificate(child))

    assert [bundle.evidence_type for bundle in evidence] == ["why", "contrastive"]
    assert all(bundle.validate() == () for bundle in evidence)

    why, contrastive = evidence
    assert why.plan_id == "child_explain"
    assert contrastive.parent_plan_id == "parent_explain"
    assert contrastive.child_plan_id == "child_explain"
    assert contrastive.diff_id == "diff_explain"

    refs = {record.ref_id for record in contrastive.evidence_records}
    assert "plan_diff:diff_explain:weighted_edit_cost" in refs
    assert "certificate:cert_explain:comparison_eligibility" in refs
    assert "certificate:cert_explain:route_validation" in refs
    assert "route:matrix_explain:route_validation" in refs
    assert contrastive.to_publication_record()["claims"]


def test_explanation_evidence_builder_changed_claims_are_publication_safe():
    child = plan()
    builder = ExplanationEvidenceBuilder()

    contrastive = builder.changed(child, diff(), certificate(child))

    claim_ids = {claim.claim_id for claim in contrastive.supported_claims()}
    assert "claim_weighted_edit_cost_child_explain" in claim_ids
    assert "claim_independent_evaluation_child_explain" in claim_ids
    assert contrastive.validate() == ()
