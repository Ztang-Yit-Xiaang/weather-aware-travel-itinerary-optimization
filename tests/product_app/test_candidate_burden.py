from __future__ import annotations

from dataclasses import replace

import pytest

from itinerary_system.product_app.candidate_burden import (
    CandidateBurdenError,
    EvaluatorCandidateEvidenceV1,
    FastFeasibilityPrecheckV1,
    assess_candidate_insertion,
    assess_candidate_replacement,
    select_bounded_candidate_top_k,
)
from itinerary_system.routing.matrix import RouteMatrix, RouteMatrixCell


def _cell(origin: str, destination: str, minutes: float, meters: float) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=meters,
        duration_s=minutes * 60,
        route_leg_id=f"leg_{origin}_{destination}",
        road_validated=True,
        provider="frozen_osrm",
        query_hash=f"query_{origin}_{destination}",
        context_snapshot_id="context_1",
        routing_status="road_validated",
        geometry_source="osrm",
        distance_source="osrm",
        duration_source="osrm",
    )


def _matrix(*, valid_candidate_legs: bool = True) -> RouteMatrix:
    cells = {
        ("i", "j"): _cell("i", "j", 15, 12_000),
    }
    if valid_candidate_legs:
        cells.update(
            {
                ("i", "c"): _cell("i", "c", 10, 8_000),
                ("c", "j"): _cell("c", "j", 20, 16_000),
            }
        )
    return RouteMatrix(
        matrix_id="matrix_1",
        context_snapshot_id="context_1",
        entity_ids=("i", "c", "j"),
        cells=cells,
    )


def _assess(**overrides: object):
    arguments: dict[str, object] = {
        "candidate_id": "candidate_c",
        "place_id": "c",
        "predecessor_id": "i",
        "successor_id": "j",
        "route_matrix": _matrix(),
        "geographic_distance_m": 300,
        "visit_minutes": 60,
        "parking_minutes": 5,
        "walking_minutes": 10,
        "waiting_minutes": 0,
        "nearby_radius_m": 1_000,
        "maximum_detour_minutes": 20,
    }
    arguments.update(overrides)
    return assess_candidate_insertion(**arguments)


def test_insertion_burden_uses_directed_delta_and_all_time_components() -> None:
    result = _assess()

    assert result.predecessor_to_candidate_minutes == 10
    assert result.candidate_to_successor_minutes == 20
    assert result.predecessor_to_successor_minutes == 15
    assert result.marginal_travel_minutes == 15
    assert result.marginal_travel_distance_m == 12_000
    assert result.total_insertion_minutes == 90
    assert result.nearby is True
    assert result.route_near is True
    assert result.candidate_state == "route_near"
    assert result.ranking_eligible is False
    assert result.recommended is False


def test_replacement_burden_uses_two_leg_target_baseline_not_direct_shortcut() -> None:
    matrix = RouteMatrix(
        matrix_id="matrix_replacement",
        context_snapshot_id="context_1",
        entity_ids=("i", "target", "c", "j"),
        cells={
            ("i", "target"): _cell("i", "target", 12, 9_000),
            ("target", "j"): _cell("target", "j", 18, 13_000),
            ("i", "c"): _cell("i", "c", 10, 8_000),
            ("c", "j"): _cell("c", "j", 15, 11_000),
            # Deliberately much shorter: replacement must not use this shortcut.
            ("i", "j"): _cell("i", "j", 5, 4_000),
        },
    )

    result = assess_candidate_replacement(
        candidate_id="candidate_c",
        place_id="c",
        replacement_target_id="target",
        predecessor_id="i",
        successor_id="j",
        route_matrix=matrix,
        geographic_distance_m=100,
        visit_minutes=30,
        parking_minutes=None,
        walking_minutes=None,
        waiting_minutes=None,
        nearby_radius_m=1_000,
        maximum_detour_minutes=20,
    )

    assert result.context_kind == "replacement"
    assert result.replacement_target_id == "target"
    assert result.baseline_route_leg_ids == ("leg_i_target", "leg_target_j")
    assert result.baseline_travel_minutes == 30
    assert result.baseline_travel_distance_m == 22_000
    assert result.marginal_travel_minutes == -5
    assert result.marginal_travel_distance_m == -3_000
    assert result.predecessor_to_successor_minutes is None
    assert result.total_insertion_minutes is None


@pytest.mark.parametrize(
    ("field", "blocking_code"),
    [
        ("visit_minutes", "visit_duration_unavailable"),
        ("parking_minutes", "parking_time_unavailable"),
        ("walking_minutes", "walking_time_unavailable"),
        ("waiting_minutes", "waiting_time_unavailable"),
    ],
)
def test_missing_time_component_remains_missing(field: str, blocking_code: str) -> None:
    result = _assess(**{field: None})

    assert result.marginal_travel_minutes == 15
    assert result.total_insertion_minutes is None
    assert blocking_code in result.blocking_codes


def test_euclidean_near_does_not_imply_route_near() -> None:
    result = _assess(
        geographic_distance_m=5,
        route_matrix=_matrix(valid_candidate_legs=False),
    )

    assert result.nearby is True
    assert result.route_near is False
    assert result.candidate_state == "nearby"
    assert result.marginal_travel_minutes is None
    assert "route_leg_predecessor_candidate_unavailable" in result.blocking_codes
    assert "route_leg_candidate_successor_unavailable" in result.blocking_codes


def test_precheck_can_only_establish_likely_feasible() -> None:
    result = _assess(
        precheck=FastFeasibilityPrecheckV1(
            status="passed", evidence_refs=("precheck_1",)
        )
    )

    assert result.likely_feasible is True
    assert result.evaluated_feasible is False
    assert result.candidate_state == "likely_feasible"
    assert result.recommended is False


def test_independent_evaluator_can_establish_evaluated_feasible_without_ranking() -> None:
    result = _assess(
        evaluator=EvaluatorCandidateEvidenceV1(
            owner="independent_evaluator",
            decision_eligible=True,
            ranking_eligible=False,
            evaluator_rank=None,
            recommended=False,
            evidence_refs=("evaluation_1",),
        )
    )

    assert result.evaluated_feasible is True
    assert result.ranking_eligible is False
    assert result.candidate_state == "evaluated_feasible"
    assert "evaluator_ranking_ineligible" in result.blocking_codes


def test_only_independent_evaluator_can_make_candidate_recommended() -> None:
    evaluator = EvaluatorCandidateEvidenceV1(
        owner="independent_evaluator",
        decision_eligible=True,
        ranking_eligible=True,
        evaluator_rank=1,
        recommended=True,
        evidence_refs=("evaluation_1", "ranking_1"),
    )
    result = _assess(evaluator=evaluator)

    assert result.evaluated_feasible is True
    assert result.ranking_eligible is True
    assert result.recommended is True
    assert result.candidate_state == "recommended"
    assert result.evaluator_rank == 1

    with pytest.raises(CandidateBurdenError, match="evaluator_owner_invalid"):
        EvaluatorCandidateEvidenceV1(
            owner="frontend",
            decision_eligible=True,
            ranking_eligible=True,
            evaluator_rank=1,
            recommended=True,
            evidence_refs=("claim_1",),
        )


def test_recommendation_requires_ranking_eligibility() -> None:
    with pytest.raises(
        CandidateBurdenError, match="recommendation_requires_ranking_eligibility"
    ):
        EvaluatorCandidateEvidenceV1(
            owner="independent_evaluator",
            decision_eligible=True,
            ranking_eligible=False,
            evaluator_rank=None,
            recommended=True,
            evidence_refs=("evaluation_1",),
        )


def test_final_burden_dto_cannot_be_forged_into_a_ranked_recommendation() -> None:
    baseline = _assess()

    with pytest.raises(CandidateBurdenError, match="evaluator_evidence_required"):
        replace(
            baseline,
            ranking_eligible=True,
            recommended=True,
            evaluated_feasible=True,
            candidate_state="recommended",
            evaluator_rank=1,
            evidence_refs=(),
        )

    with pytest.raises(CandidateBurdenError, match="candidate_state_inconsistent"):
        replace(baseline, candidate_state="recommended")

    with pytest.raises(CandidateBurdenError, match="candidate_state_flag_invalid"):
        replace(baseline, ranking_eligible=1)


def test_top_k_uses_evaluator_rank_then_stable_ids_without_route_rank_claim() -> None:
    evaluator_second = EvaluatorCandidateEvidenceV1(
        owner="independent_evaluator",
        decision_eligible=True,
        ranking_eligible=True,
        evaluator_rank=2,
        recommended=False,
        evidence_refs=("evaluation_2",),
    )
    ranked = _assess(candidate_id="ranked", evaluator=evaluator_second)
    zeta = _assess(candidate_id="zeta", geographic_distance_m=1)
    alpha = _assess(candidate_id="alpha", geographic_distance_m=900)

    selected = select_bounded_candidate_top_k((zeta, alpha, ranked), limit=2)

    assert [candidate.candidate_id for candidate in selected] == ["ranked", "alpha"]
    assert selected[1].ranking_eligible is False
    assert selected[1].recommended is False


@pytest.mark.parametrize("limit", [0, 51, True])
def test_top_k_enforces_resource_bound(limit: object) -> None:
    with pytest.raises(CandidateBurdenError, match="candidate_limit_invalid"):
        select_bounded_candidate_top_k((_assess(),), limit=limit)


def test_duplicate_candidate_ids_are_rejected() -> None:
    candidate = _assess()
    with pytest.raises(CandidateBurdenError, match="candidate_id_duplicate"):
        select_bounded_candidate_top_k((candidate, candidate), limit=2)
