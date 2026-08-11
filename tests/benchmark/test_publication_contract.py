from __future__ import annotations

import pytest

from itinerary_system.benchmark.publication import (
    REQUIRED_PUBLICATION_METHOD_IDS,
    publication_method_set_findings,
    publication_readiness,
    require_publication_method_set,
)


def test_publication_method_contract_requires_all_four_distinct_methods():
    with pytest.raises(ValueError, match="missing required methods"):
        require_publication_method_set(("full_reoptimization",))

    require_publication_method_set(REQUIRED_PUBLICATION_METHOD_IDS)
    findings = publication_method_set_findings((*REQUIRED_PUBLICATION_METHOD_IDS, "full_reoptimization"))

    assert findings["complete"] is False
    assert findings["duplicate_method_ids"] == ["full_reoptimization"]


def test_publication_readiness_requires_every_run_to_be_independently_rankable():
    records = [
        {"status": "completed", "benchmark_ranking_eligible": True},
        {"status": "failed", "benchmark_ranking_eligible": False},
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["complete"] is True
    assert readiness["ranking_eligible_run_count"] == 1
    assert readiness["failed_run_count"] == 1
    assert readiness["publication_ready"] is False


def test_publication_readiness_allows_evaluated_failure_but_requires_shared_route_input():
    records = [
        {
            "scenario_id": "scenario_1",
            "method_id": method_id,
            "status": "failed_strict" if method_id == "context_blind_solver" else "completed",
            "benchmark_ranking_eligible": method_id != "context_blind_solver",
            "benchmark_method_provenance_valid": True,
            "quality_route_validated": True,
            "benchmark_route_matrix_hash": "shared_hash",
            "benchmark_route_source_bundle_id": "shared_bundle",
        }
        for method_id in REQUIRED_PUBLICATION_METHOD_IDS
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["failed_run_count"] == 1
    assert readiness["ranking_eligible_run_count"] == 3
    assert readiness["all_runs_ranking_eligible"] is False
    assert readiness["evidence_complete"] is True
    assert readiness["publication_ready"] is True


def test_publication_readiness_accepts_input_route_evidence_for_failed_run_without_output():
    records = [
        {
            "scenario_id": "scenario_1",
            "method_id": method_id,
            "status": "failed" if method_id == "context_blind_solver" else "completed",
            "output_plan_id": "" if method_id == "context_blind_solver" else f"plan_{method_id}",
            "benchmark_ranking_eligible": method_id != "context_blind_solver",
            "benchmark_method_provenance_valid": True,
            "quality_route_validated": None if method_id == "context_blind_solver" else True,
            "benchmark_route_matrix_hash": "shared_hash",
            "benchmark_route_source_bundle_id": "shared_bundle",
        }
        for method_id in REQUIRED_PUBLICATION_METHOD_IDS
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["route_evidence_complete"] is True
    assert readiness["ranking_eligible_run_count"] == 3
    assert readiness["publication_ready"] is True


def test_publication_readiness_requires_output_route_certificate_for_failed_strict_plan():
    records = [
        {
            "scenario_id": "scenario_1",
            "method_id": method_id,
            "status": "failed_strict" if method_id == "context_blind_solver" else "completed",
            "output_plan_id": f"plan_{method_id}",
            "benchmark_ranking_eligible": method_id != "context_blind_solver",
            "benchmark_method_provenance_valid": True,
            "quality_route_validated": method_id != "context_blind_solver",
            "benchmark_route_matrix_hash": "shared_hash",
            "benchmark_route_source_bundle_id": "shared_bundle",
        }
        for method_id in REQUIRED_PUBLICATION_METHOD_IDS
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["route_evidence_complete"] is False
    assert readiness["publication_ready"] is False


def test_publication_readiness_rejects_mismatched_route_matrix_hashes():
    records = [
        {
            "scenario_id": "scenario_1",
            "method_id": method_id,
            "benchmark_ranking_eligible": True,
            "benchmark_method_provenance_valid": True,
            "quality_route_validated": True,
            "benchmark_route_matrix_hash": "different_hash" if index == 3 else "shared_hash",
            "benchmark_route_source_bundle_id": "shared_bundle",
        }
        for index, method_id in enumerate(REQUIRED_PUBLICATION_METHOD_IDS)
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["route_input_consistency_complete"] is False
    assert readiness["publication_ready"] is False


def test_publication_readiness_rejects_missing_route_source_bundle_lineage():
    records = [
        {
            "scenario_id": "scenario_1",
            "method_id": method_id,
            "benchmark_ranking_eligible": True,
            "benchmark_method_provenance_valid": True,
            "quality_route_validated": True,
            "benchmark_route_matrix_hash": "shared_hash",
            "benchmark_route_source_bundle_id": "",
        }
        for method_id in REQUIRED_PUBLICATION_METHOD_IDS
    ]

    readiness = publication_readiness(REQUIRED_PUBLICATION_METHOD_IDS, records)

    assert readiness["route_source_bundle_consistency_complete"] is False
    assert readiness["publication_ready"] is False
