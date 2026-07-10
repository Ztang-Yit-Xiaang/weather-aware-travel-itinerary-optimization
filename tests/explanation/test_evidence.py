from __future__ import annotations

from itinerary_system.explanation import (
    ContrastiveEvidence,
    EvidenceRecord,
    ExplanationClaim,
    WhatIfEvidence,
    WhyEvidence,
    WhyNotEvidence,
    validate_explanation_claims,
)


def record(ref_id: str, source_type: str, source_id: str, field_path: str = "") -> EvidenceRecord:
    return EvidenceRecord(
        ref_id=ref_id,
        source_type=source_type,
        source_id=source_id,
        field_path=field_path,
        payload={"value": 3.0},
    )


def test_numerical_claim_without_evidence_fails_closed_and_is_hidden_from_publication():
    claim = ExplanationClaim(
        claim_id="claim_weighted_cost",
        claim_type="numeric",
        text_template="The repair changed the plan by {weighted_cost} weighted edit units.",
        values={"weighted_cost": 3.0},
        evidence_refs=(),
    )
    evidence = WhyEvidence(
        evidence_id="why_plan",
        plan_id="plan_child",
        target_id="plan_child",
        claims=(claim,),
        evidence_records=(),
    )

    findings = evidence.validate()

    assert [finding.code for finding in findings] == ["missing_evidence_refs"]
    assert evidence.to_publication_record()["claims"] == []


def test_numerical_claim_with_plan_diff_reference_is_supported():
    claim = ExplanationClaim(
        claim_id="claim_weighted_cost",
        claim_type="numeric",
        text_template="The repair changed the plan by {weighted_cost} weighted edit units.",
        values={"weighted_cost": 3.0},
        evidence_refs=("diff:weighted_edit_cost",),
    )
    evidence_record = record(
        "diff:weighted_edit_cost",
        "plan_diff",
        "diff_123",
        "weighted_edit_cost",
    )

    findings = validate_explanation_claims((claim,), (evidence_record,))

    assert findings == ()
    assert claim.render() == "The repair changed the plan by 3.0 weighted edit units."


def test_causal_claim_with_missing_reference_is_unsupported():
    claim = ExplanationClaim(
        claim_id="claim_route_reason",
        claim_type="causal",
        text_template="The stop was omitted because the route evidence failed validation.",
        evidence_refs=("route:missing",),
    )

    findings = validate_explanation_claims((claim,), (record("route:present", "route", "leg_1"),))

    assert [finding.code for finding in findings] == ["missing_evidence_ref"]
    assert findings[0].claim_id == "claim_route_reason"


def test_evidence_containers_validate_allowed_references_and_serialize_records():
    why_not = WhyNotEvidence(
        evidence_id="why_not_stop",
        target_id="poi_closed",
        failed_attempt_ids=("attempt_1",),
        claims=(
            ExplanationClaim(
                claim_id="claim_constraint",
                claim_type="causal",
                text_template="The stop was omitted because a confirmed closure constraint blocked it.",
                evidence_refs=("constraint:closure",),
            ),
        ),
        evidence_records=(record("constraint:closure", "constraint", "closure_1"),),
    )
    what_if = WhatIfEvidence(
        evidence_id="what_if_stop",
        sandbox_request_id="sandbox_1",
        counterfactual_run_ids=("run_cf_1",),
        outcome_status="infeasible",
        claims=(
            ExplanationClaim(
                claim_id="claim_counterfactual",
                claim_type="causal",
                text_template="Forcing that stop required a failed counterfactual run.",
                evidence_refs=("counterfactual:run",),
            ),
        ),
        evidence_records=(record("counterfactual:run", "counterfactual_run", "run_cf_1"),),
    )
    contrast = ContrastiveEvidence(
        evidence_id="contrast_parent_child",
        parent_plan_id="plan_parent",
        child_plan_id="plan_child",
        diff_id="diff_123",
        claims=(
            ExplanationClaim(
                claim_id="claim_added_stop",
                claim_type="causal",
                text_template="The child plan differs because the repair added a replacement stop.",
                evidence_refs=("diff:added_stop", "evaluation:certificate"),
            ),
        ),
        evidence_records=(
            record("diff:added_stop", "plan_diff", "diff_123", "added_stops[0]"),
            record("evaluation:certificate", "evaluation", "cert_1", "comparison_eligibility"),
        ),
    )

    assert why_not.validate() == ()
    assert what_if.validate() == ()
    assert contrast.validate() == ()
    assert why_not.evidence_refs() == ("constraint:closure",)
    assert what_if.to_record()["counterfactual_run_ids"] == ["run_cf_1"]
    assert contrast.to_publication_record()["claims"][0]["claim_id"] == "claim_added_stop"


def test_claim_rejects_disallowed_evidence_type_for_numeric_claim():
    claim = ExplanationClaim(
        claim_id="claim_numeric_bad_ref",
        claim_type="numeric",
        text_template="The score changed by {delta}.",
        values={"delta": 2.0},
        evidence_refs=("constraint:soft_preference",),
    )

    findings = validate_explanation_claims((claim,), (record("constraint:soft_preference", "constraint", "constraint_1")))

    assert [finding.code for finding in findings] == ["invalid_evidence_type"]
