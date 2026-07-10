from __future__ import annotations

from itinerary_system.explanation import (
    CounterfactualRunner,
    DeterministicTemplateVerbalizer,
    EvidenceRecord,
    ExplanationClaim,
    WhyEvidence,
    validate_claim_evidence_map,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_explain",
        source_run_id="run_parent",
        planning_request_id="request_demo",
        catalog_snapshot_id="catalog_demo",
        context_snapshot_id="context_demo",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Museum", "day": 1, "stop_order": 1},
            {"stop_id": "poi_c", "name": "Garden", "day": 1, "stop_order": 2},
        ),
        day_assignments={"poi_a": 1, "poi_c": 1},
        sequence=("poi_a", "poi_c"),
        ordered_days=({"day": 1, "stop_ids": ("poi_a", "poi_c")},),
        created_at="2026-07-08T00:00:00+00:00",
    )


def test_why_not_counterfactual_forces_omitted_option_and_records_failed_run():
    parent = parent_plan()
    original_hash = parent.content_hash
    received = []

    def executor(request):
        received.append(request)
        return {
            "status": "infeasible",
            "repair_outcome_id": "outcome_failed",
            "diagnosis": {"failure_reasons": ["time_window_blocked"]},
        }

    evidence = CounterfactualRunner(executor).explain_why_not(
        parent,
        target_id="poi_b",
        source_request_id="repair_request_1",
    )

    assert parent.content_hash == original_hash
    assert evidence.validate() == ()
    assert evidence.target_id == "poi_b"
    assert len(received) == 1
    assert received[0].question_type == "why_not"
    assert received[0].requested_change == "force_include"
    assert received[0].force_constraints["must_include"] == ("poi_b",)
    assert received[0].request_id.startswith("sandbox_cf_")
    assert evidence.failed_attempt_ids == evidence.counterfactual_run_ids
    assert evidence.outcome_status == "infeasible"
    assert evidence.evidence_records[0].source_type == "counterfactual_run"
    assert evidence.evidence_records[0].payload["failure_reasons"] == ["time_window_blocked"]


def test_what_if_counterfactual_creates_sandbox_request_without_parent_mutation():
    parent = parent_plan()
    received = []

    def executor(request):
        received.append(request)
        return {
            "status": "accepted",
            "repair_outcome_id": "outcome_accepted",
            "child_plan_id": "child_counterfactual",
            "diff_id": "diff_counterfactual",
            "evaluation_id": "eval_counterfactual",
        }

    evidence = CounterfactualRunner(executor).explain_what_if(
        parent,
        target_id="poi_c",
        requested_change="move_to_day_2",
        sandbox_overrides={"target_day": 2},
        source_request_id="repair_request_1",
    )

    assert received[0].question_type == "what_if"
    assert received[0].force_constraints["target_day"] == 2
    assert evidence.sandbox_request_id == received[0].request_id
    assert evidence.outcome_status == "accepted"
    assert evidence.to_record()["counterfactual_run_ids"] == list(evidence.counterfactual_run_ids)
    assert evidence.evidence_records[0].payload["child_plan_id"] == "child_counterfactual"
    assert parent.sequence == ("poi_a", "poi_c")


def test_counterfactual_solver_failure_becomes_failure_evidence_not_success_text():
    def executor(_request):
        raise RuntimeError("solver unavailable")

    evidence = CounterfactualRunner(executor).explain_what_if(
        parent_plan(),
        target_id="poi_b",
        requested_change="force_include",
    )

    assert evidence.validate() == ()
    assert evidence.outcome_status == "failed"
    assert "solver unavailable" in evidence.evidence_records[0].payload["failure_reasons"][0]
    assert "succeeded" not in evidence.claims[0].render().lower()


def test_counterfactual_parent_mutation_is_detected_and_marked_failed():
    parent = parent_plan()

    def mutating_executor(_request):
        parent.selected_stops[0]["name"] = "Mutated Museum"
        return {"status": "accepted", "child_plan_id": "child_mutated"}

    evidence = CounterfactualRunner(mutating_executor).explain_why_not(parent, target_id="poi_b")

    assert evidence.outcome_status == "parent_mutated"
    assert "parent_plan_mutated" in evidence.evidence_records[0].payload["failure_reasons"]
    assert evidence.validate() == ()


def test_deterministic_verbalizer_hides_unsupported_claims_and_maps_evidence_refs():
    supported_record = EvidenceRecord(
        ref_id="evaluation:score",
        source_type="evaluation",
        source_id="eval_1",
        field_path="soft_metrics.score_delta",
        payload={"score_delta": 2.5},
    )
    supported = ExplanationClaim(
        claim_id="claim_supported",
        claim_type="numeric",
        text_template="The score changed by {delta}.",
        values={"delta": 2.5},
        evidence_refs=("evaluation:score",),
    )
    unsupported = ExplanationClaim(
        claim_id="claim_unsupported",
        claim_type="causal",
        text_template="This hidden claim has no evidence.",
        evidence_refs=(),
    )
    evidence = WhyEvidence(
        evidence_id="why_score",
        plan_id="child",
        target_id="child",
        claims=(supported, unsupported),
        evidence_records=(supported_record,),
    )

    rendered = DeterministicTemplateVerbalizer().render(evidence)

    assert rendered.texts == ("The score changed by 2.5.",)
    assert rendered.claim_evidence_map == {"claim_supported": ("evaluation:score",)}
    assert [finding.code for finding in rendered.findings] == ["missing_evidence_refs"]


def test_llm_claim_evidence_map_rejects_unmapped_and_unknown_claims():
    evidence = WhyEvidence(
        evidence_id="why_score",
        plan_id="child",
        target_id="child",
        claims=(
            ExplanationClaim(
                claim_id="claim_supported",
                claim_type="numeric",
                text_template="The score changed by {delta}.",
                values={"delta": 2.5},
                evidence_refs=("evaluation:score",),
            ),
        ),
        evidence_records=(
            EvidenceRecord(
                ref_id="evaluation:score",
                source_type="evaluation",
                source_id="eval_1",
                field_path="soft_metrics.score_delta",
            ),
        ),
    )

    findings = validate_claim_evidence_map(
        evidence,
        {
            "claim_supported": (),
            "claim_extra": ("evaluation:score",),
        },
    )

    assert [finding.code for finding in findings] == [
        "missing_claim_evidence_mapping",
        "unknown_claim_mapping",
    ]
