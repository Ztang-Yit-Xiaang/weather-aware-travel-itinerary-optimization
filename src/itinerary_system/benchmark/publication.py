"""Publication method-set and eligibility contracts for paired benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

REQUIRED_PUBLICATION_METHOD_IDS = (
    "context_blind_solver",
    "deterministic_context_aware_heuristic",
    "progressive_sequential_lexicographic_repair",
    "full_reoptimization",
)


def publication_method_set_findings(method_ids: Iterable[str]) -> dict[str, Any]:
    ordered = tuple(str(method_id) for method_id in method_ids)
    duplicates = tuple(sorted({method_id for method_id in ordered if ordered.count(method_id) > 1}))
    missing = tuple(method_id for method_id in REQUIRED_PUBLICATION_METHOD_IDS if method_id not in ordered)
    unexpected = tuple(method_id for method_id in ordered if method_id not in REQUIRED_PUBLICATION_METHOD_IDS)
    return {
        "required_method_ids": list(REQUIRED_PUBLICATION_METHOD_IDS),
        "provided_method_ids": list(ordered),
        "missing_method_ids": list(missing),
        "unexpected_method_ids": list(unexpected),
        "duplicate_method_ids": list(duplicates),
        "complete": not missing and not unexpected and not duplicates,
    }


def require_publication_method_set(method_ids: Iterable[str]) -> None:
    findings = publication_method_set_findings(method_ids)
    if findings["duplicate_method_ids"]:
        raise ValueError(f"duplicate benchmark method IDs: {findings['duplicate_method_ids']}")
    if findings["missing_method_ids"]:
        raise ValueError(f"publication benchmark missing required methods: {findings['missing_method_ids']}")
    if findings["unexpected_method_ids"]:
        raise ValueError(f"publication benchmark has unexpected methods: {findings['unexpected_method_ids']}")


def publication_readiness(
    method_ids: Iterable[str],
    run_records: Iterable[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    method_findings = publication_method_set_findings(method_ids)
    records = tuple(_record(item) for item in run_records)
    ranking_eligible = tuple(record for record in records if bool(record.get("benchmark_ranking_eligible")))
    failed = tuple(record for record in records if str(record.get("status", "")).lower().startswith("fail"))
    provenance_complete = bool(records) and all(
        record.get("benchmark_method_provenance_valid") is True for record in records
    )
    route_evidence_complete = bool(records) and all(_route_evidence_valid(record) for record in records)
    represented_method_ids = {str(record.get("method_id") or "") for record in records}
    method_run_coverage_complete = set(REQUIRED_PUBLICATION_METHOD_IDS).issubset(represented_method_ids)
    run_keys = tuple(
        (str(record.get("scenario_id") or ""), str(record.get("method_id") or "")) for record in records
    )
    duplicate_run_keys = tuple(sorted({key for key in run_keys if run_keys.count(key) > 1}))
    scenario_ids = {str(record.get("scenario_id") or "") for record in records}
    route_hashes_by_scenario = {
        scenario_id: {
            str(record.get("benchmark_route_matrix_hash") or "")
            for record in records
            if str(record.get("scenario_id") or "") == scenario_id
        }
        for scenario_id in scenario_ids
    }
    route_input_consistency_complete = bool(route_hashes_by_scenario) and all(
        len(hashes) == 1 and "" not in hashes for hashes in route_hashes_by_scenario.values()
    )
    source_bundle_ids_by_scenario = {
        scenario_id: {
            str(record.get("benchmark_route_source_bundle_id") or "")
            for record in records
            if str(record.get("scenario_id") or "") == scenario_id
        }
        for scenario_id in scenario_ids
    }
    route_source_bundle_consistency_complete = bool(source_bundle_ids_by_scenario) and all(
        len(bundle_ids) == 1 and "" not in bundle_ids for bundle_ids in source_bundle_ids_by_scenario.values()
    )
    evidence_complete = (
        provenance_complete
        and route_evidence_complete
        and method_run_coverage_complete
        and route_input_consistency_complete
        and route_source_bundle_consistency_complete
        and not duplicate_run_keys
    )
    return {
        **method_findings,
        "run_count": len(records),
        "ranking_eligible_run_count": len(ranking_eligible),
        "failed_run_count": len(failed),
        "all_runs_ranking_eligible": bool(records) and len(ranking_eligible) == len(records),
        "method_provenance_complete": provenance_complete,
        "route_evidence_complete": route_evidence_complete,
        "method_run_coverage_complete": method_run_coverage_complete,
        "route_input_consistency_complete": route_input_consistency_complete,
        "route_source_bundle_consistency_complete": route_source_bundle_consistency_complete,
        "route_source_bundle_ids_by_scenario": {
            scenario_id: sorted(bundle_ids)
            for scenario_id, bundle_ids in sorted(source_bundle_ids_by_scenario.items())
        },
        "route_matrix_hashes_by_scenario": {
            scenario_id: sorted(hashes) for scenario_id, hashes in sorted(route_hashes_by_scenario.items())
        },
        "duplicate_scenario_method_pairs": [list(key) for key in duplicate_run_keys],
        "evidence_complete": evidence_complete,
        "publication_ready": bool(method_findings["complete"]) and evidence_complete,
    }


def _route_evidence_valid(record: Mapping[str, Any]) -> bool:
    if record.get("quality_route_validated") is True:
        return True
    failed_without_output = (
        str(record.get("status", "")).lower().startswith("fail")
        and not str(record.get("output_plan_id") or "").strip()
    )
    if not failed_without_output:
        return False
    return bool(
        str(record.get("benchmark_route_matrix_hash") or "").strip()
        and str(record.get("benchmark_route_source_bundle_id") or "").strip()
    )


def _record(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_record"):
        return dict(value.to_record())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}

