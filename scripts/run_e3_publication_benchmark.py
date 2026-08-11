"""Run the real four-method E3 publication benchmark from frozen E2 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.benchmark import (  # noqa: E402
    DisruptionFamily,
    build_benchmark_route_coverage,
    build_publication_benchmark_method_adapters,
    generate_disruption_scenarios,
    run_benchmark_suite,
)
from itinerary_system.plans import load_plan  # noqa: E402
from itinerary_system.repair import DayRouteSolverConfig  # noqa: E402
from itinerary_system.research_artifacts import stable_content_hash  # noqa: E402
from itinerary_system.routing import (  # noqa: E402
    load_route_matrix_from_cache,
    route_anchor_key,
    validate_route_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze real E3 inputs and run the canonical four-method paired benchmark."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase0-plans", required=True, help="Frozen production_phase0_plan_artifacts.jsonl.")
    parser.add_argument(
        "--candidate-catalog",
        help="Frozen production_enriched_poi_catalog.csv used for located disruption candidates.",
    )
    parser.add_argument(
        "--candidate-snap-audit",
        help="Frozen candidate_snap_audit.csv; when supplied, only <=threshold candidates are eligible.",
    )
    parser.add_argument("--parent-plan-id", required=True)
    parser.add_argument("--route-stops", required=True, help="Frozen production_method_route_stops.csv.")
    parser.add_argument("--parent-method-id", default="hierarchical_gurobi_pipeline")
    parser.add_argument("--route-cache", required=True, help="Frozen production_road_route_cache.csv.")
    parser.add_argument("--route-evidence-manifest", required=True)
    parser.add_argument("--route-matrix-artifact", required=True)
    parser.add_argument("--expected-route-matrix-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--family",
        action="append",
        choices=[family.value for family in DisruptionFamily],
        default=[],
        help="Run only selected disruption families. Repeat for multiple families; default is all six.",
    )
    parser.add_argument("--max-day-minutes", type=float, default=720.0)
    parser.add_argument("--default-visit-minutes", type=float, default=60.0)
    parser.add_argument("--max-complete-candidates", type=int, default=50_000)
    parser.add_argument(
        "--allow-incomplete-route-coverage",
        action="store_true",
        help="Diagnostic-only: run methods even when the conservative E3 route universe is incomplete.",
    )
    parser.add_argument("--require-publication-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if output_root.exists():
        raise ValueError(f"immutable E3 output already exists: {output_root}")

    config_path = Path(args.config)
    phase0_plans_path = Path(args.phase0_plans)
    candidate_catalog_path = Path(args.candidate_catalog) if args.candidate_catalog else None
    candidate_snap_audit_path = Path(args.candidate_snap_audit) if args.candidate_snap_audit else None
    if candidate_snap_audit_path is not None and candidate_catalog_path is None:
        raise ValueError("--candidate-snap-audit requires --candidate-catalog")
    route_stops_path = Path(args.route_stops)
    route_cache_path = Path(args.route_cache)
    route_manifest_path = Path(args.route_evidence_manifest)
    route_matrix_artifact_path = Path(args.route_matrix_artifact)
    for path in (
        config_path,
        phase0_plans_path,
        route_stops_path,
        route_cache_path,
        route_manifest_path,
        route_matrix_artifact_path,
        *((candidate_catalog_path,) if candidate_catalog_path is not None else ()),
        *((candidate_snap_audit_path,) if candidate_snap_audit_path is not None else ()),
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    route_manifest = _read_json(route_manifest_path)
    if route_manifest.get("publication_ready") is not True:
        raise ValueError("E3 requires a publication-ready E2 route-evidence manifest")
    bundle_id = str(route_manifest.get("bundle_id") or "")
    source_sha256 = str(route_manifest.get("artifact_hashes", {}).get("validated_cache") or "")
    if not bundle_id or len(source_sha256) != 64:
        raise ValueError("E2 route-evidence manifest lacks bundle ID or validated-cache SHA-256")

    parent_record = _select_parent_record(
        phase0_plans_path,
        parent_plan_id=str(args.parent_plan_id),
    )
    context_snapshot_id = str(parent_record.get("context_snapshot_id") or "")
    if not context_snapshot_id:
        raise ValueError("selected parent plan lacks context_snapshot_id")

    route_matrix = load_route_matrix_from_cache(
        route_cache_path,
        context_snapshot_id,
        source_bundle_id=bundle_id,
        expected_source_sha256=source_sha256,
    )
    if route_matrix.matrix_id != str(args.expected_route_matrix_id):
        raise ValueError(
            f"route matrix ID mismatch: expected {args.expected_route_matrix_id}, got {route_matrix.matrix_id}"
        )
    matrix_report = validate_route_matrix(route_matrix, require_publication_ready=True)
    if not matrix_report.publication_ready:
        raise ValueError(f"E3 route matrix is not publication-ready: {matrix_report.errors}")

    start_anchors, end_anchors = _day_anchors(
        route_stops_path,
        method_id=str(args.parent_method_id),
    )
    day_route_config = DayRouteSolverConfig(
        max_day_minutes=float(args.max_day_minutes),
        default_visit_minutes=float(args.default_visit_minutes),
        strict_route_matrix=True,
        start_anchor_by_day=start_anchors,
        end_anchor_by_day=end_anchors,
    )

    inputs_dir = output_root / "inputs"
    inputs_dir.mkdir(parents=True)
    parent_path = inputs_dir / "parent_plan.json"
    parent_path.write_text(_stable_json(parent_record), encoding="utf-8")
    parent_plan = load_plan(parent_path)
    expected_parent_hash = str(parent_record.get("content_hash") or "")
    if expected_parent_hash and parent_plan.content_hash != expected_parent_hash:
        raise ValueError("selected parent plan content hash does not match its frozen record")

    candidate_pool = (
        _load_candidate_pool(candidate_catalog_path, snap_audit_path=candidate_snap_audit_path)
        if candidate_catalog_path
        else ()
    )
    scenarios = generate_disruption_scenarios(
        parent_plan,
        seed=int(args.seed),
        evidence_status="synthetic",
        candidate_pool=candidate_pool,
        max_daily_travel_minutes=float(args.max_day_minutes),
    )
    selected_families = {str(value) for value in args.family}
    if selected_families:
        scenarios = tuple(scenario for scenario in scenarios if scenario.family.value in selected_families)
    if not scenarios:
        raise ValueError("no E3 scenarios selected")
    scenario_records = tuple(scenario.to_record() for scenario in scenarios)
    scenarios_path = inputs_dir / "scenarios.jsonl"
    _write_jsonl(scenarios_path, scenario_records)
    route_coverage = build_benchmark_route_coverage(
        parent_plan=parent_plan,
        scenarios=scenarios,
        route_matrix=route_matrix,
        start_anchor_by_day=start_anchors,
        end_anchor_by_day=end_anchors,
        entity_coordinates=_route_entity_coordinates(
            route_stops_path,
            method_id=str(args.parent_method_id),
        ),
    )
    route_coverage_path = inputs_dir / "route_coverage_preflight.json"
    route_coverage_path.write_text(_stable_json(route_coverage.to_record()), encoding="utf-8")

    input_core = {
        "benchmark_id": str(args.benchmark_id),
        "parent_plan_id": parent_plan.plan_id,
        "parent_content_hash": parent_plan.content_hash,
        "catalog_snapshot_id": parent_plan.catalog_snapshot_id,
        "context_snapshot_id": parent_plan.context_snapshot_id,
        "seed": int(args.seed),
        "evidence_status": "synthetic",
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "scenario_families": [scenario.family.value for scenario in scenarios],
        "scenario_set_hash": stable_content_hash(scenario_records),
        "candidate_pool_count": len(candidate_pool),
        "candidate_pool_ids": [str(candidate.get("stop_id") or "") for candidate in candidate_pool],
        "route_bundle_id": bundle_id,
        "route_request_set_hash": str(route_manifest.get("request_set_hash") or ""),
        "route_cache_sha256": source_sha256,
        "route_matrix_id": route_matrix.matrix_id,
        "route_matrix_source_bundle_id": route_matrix.source_bundle_id,
        "route_matrix_source_sha256": route_matrix.source_content_sha256,
        "route_matrix_publication_ready": matrix_report.publication_ready,
        "day_route_config": {
            "max_day_minutes": day_route_config.max_day_minutes,
            "default_visit_minutes": day_route_config.default_visit_minutes,
            "strict_route_matrix": day_route_config.strict_route_matrix,
            "start_anchor_by_day": day_route_config.start_anchor_by_day,
            "end_anchor_by_day": day_route_config.end_anchor_by_day,
        },
        "max_complete_candidates": int(args.max_complete_candidates),
        "allow_incomplete_route_coverage": bool(args.allow_incomplete_route_coverage),
        "route_coverage_preflight": {
            "publication_ready": route_coverage.publication_ready,
            "required_pair_count": route_coverage.required_pair_count,
            "road_validated_pair_count": route_coverage.road_validated_pair_count,
            "missing_pair_count": len(route_coverage.missing_pairs),
            "ineligible_pair_count": len(route_coverage.ineligible_pairs),
            "unlocated_entity_ids": list(route_coverage.unlocated_entity_ids),
        },
        "source_files": {
            "config": _file_record(config_path),
            "phase0_plans": _file_record(phase0_plans_path),
            "candidate_catalog": _file_record(candidate_catalog_path) if candidate_catalog_path else {},
            "candidate_snap_audit": (
                _file_record(candidate_snap_audit_path) if candidate_snap_audit_path else {}
            ),
            "route_stops": _file_record(route_stops_path),
            "route_cache": _file_record(route_cache_path),
            "route_evidence_manifest": _file_record(route_manifest_path),
            "route_matrix_artifact": _file_record(route_matrix_artifact_path),
            "frozen_parent": _file_record(parent_path),
            "frozen_scenarios": _file_record(scenarios_path),
            "route_coverage_preflight": _file_record(route_coverage_path),
        },
    }
    input_manifest = {
        **input_core,
        "input_bundle_id": f"e3_input_bundle_{stable_content_hash(input_core)}",
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": "e3-publication-input-bundle-v1",
    }
    input_manifest_path = inputs_dir / "manifest.json"
    input_manifest_path.write_text(_stable_json(input_manifest), encoding="utf-8")

    if not route_coverage.publication_ready and not args.allow_incomplete_route_coverage:
        preflight_closeout = {
            "benchmark_id": str(args.benchmark_id),
            "input_bundle_id": input_manifest["input_bundle_id"],
            "input_manifest_path": str(input_manifest_path),
            "route_coverage_path": str(route_coverage_path),
            "status": "blocked_route_coverage",
            "required_pair_count": route_coverage.required_pair_count,
            "road_validated_pair_count": route_coverage.road_validated_pair_count,
            "missing_pair_count": len(route_coverage.missing_pairs),
            "ineligible_pair_count": len(route_coverage.ineligible_pairs),
            "unlocated_entity_ids": list(route_coverage.unlocated_entity_ids),
            "schema_version": "e3-publication-preflight-closeout-v1",
        }
        preflight_closeout_path = output_root / "preflight_closeout.json"
        preflight_closeout_path.write_text(_stable_json(preflight_closeout), encoding="utf-8")
        print(f"E3 input bundle: {input_manifest['input_bundle_id']}")
        print(f"Route pairs required: {route_coverage.required_pair_count}")
        print(f"Route pairs road validated: {route_coverage.road_validated_pair_count}")
        print(f"Route pairs missing: {len(route_coverage.missing_pairs)}")
        print(f"Unlocated entities: {len(route_coverage.unlocated_entity_ids)}")
        print(f"Blocked before method execution; wrote {preflight_closeout_path}")
        return 2

    methods = build_publication_benchmark_method_adapters(
        parent_plan=parent_plan,
        route_matrix=route_matrix,
        config_path=config_path,
        output_root=output_root / "pipeline_runs",
        day_route_config=day_route_config,
        publication_mode=True,
        strict=True,
        max_complete_candidates=int(args.max_complete_candidates),
    )
    result = run_benchmark_suite(
        scenarios=scenarios,
        methods=methods,
        output_dir=output_root / "benchmark",
        benchmark_id=str(args.benchmark_id),
        publication_mode=True,
    )
    benchmark_manifest = _read_json(result.manifest_path)
    readiness = dict(benchmark_manifest.get("publication_readiness") or {})
    closeout = {
        "benchmark_id": result.benchmark_id,
        "input_bundle_id": input_manifest["input_bundle_id"],
        "input_manifest_path": str(input_manifest_path),
        "benchmark_manifest_path": str(result.manifest_path),
        "benchmark_metrics_path": str(result.metrics_path),
        "run_count": len(result.run_records),
        "publication_readiness": readiness,
        "schema_version": "e3-publication-closeout-v1",
    }
    closeout_path = output_root / "closeout.json"
    closeout_path.write_text(_stable_json(closeout), encoding="utf-8")

    print(f"E3 input bundle: {input_manifest['input_bundle_id']}")
    print(f"Parent plan: {parent_plan.plan_id} ({parent_plan.content_hash})")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Route bundle: {bundle_id}")
    print(f"Route matrix: {route_matrix.matrix_id}")
    print(f"Benchmark runs: {len(result.run_records)}")
    print(f"Ranking eligible: {readiness.get('ranking_eligible_run_count', 0)}")
    print(f"Failed runs: {readiness.get('failed_run_count', 0)}")
    print(f"Publication ready: {bool(readiness.get('publication_ready'))}")
    print(f"Wrote {closeout_path}")
    if args.require_publication_ready and readiness.get("publication_ready") is not True:
        return 1
    return 0


def _select_parent_record(path: Path, *, parent_plan_id: str) -> dict[str, Any]:
    matches = [
        record
        for record in _read_jsonl(path)
        if str(record.get("plan_id") or "") == str(parent_plan_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one parent plan {parent_plan_id!r}, found {len(matches)}")
    return matches[0]


def _load_candidate_pool(
    path: Path,
    *,
    snap_audit_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    snap_records: dict[str, dict[str, Any]] = {}
    if snap_audit_path is not None:
        with snap_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            snap_records = {
                route_anchor_key(row.get("entity_id")): dict(row)
                for row in csv.DictReader(handle)
                if route_anchor_key(row.get("entity_id"))
                and str(row.get("snap_validated") or "").strip().lower() in {"1", "true", "yes"}
            }
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("itinerary_eligible") or "").strip().lower() not in {"1", "true", "yes"}:
                continue
            stop_id = route_anchor_key(row.get("name"))
            try:
                latitude = float(row.get("latitude") or "")
                longitude = float(row.get("longitude") or "")
            except (TypeError, ValueError):
                continue
            if not stop_id:
                continue
            if snap_audit_path is not None and stop_id not in snap_records:
                continue
            candidate = {
                "stop_id": stop_id,
                "name": str(row.get("name") or stop_id),
                "city": str(row.get("city") or ""),
                "latitude": latitude,
                "longitude": longitude,
                "final_poi_value": _optional_float(row.get("final_poi_value")),
                "weather_sensitivity": _optional_float(row.get("weather_sensitivity")),
                "source_confidence": _optional_float(row.get("source_confidence")),
                "source_list": str(row.get("source_list") or ""),
                "visit_duration_minutes": 45,
                "benchmark_candidate_source": "frozen_catalog",
                "route_snap_distance_m": (
                    _optional_float(snap_records.get(stop_id, {}).get("snap_distance_m"))
                    if snap_audit_path is not None
                    else None
                ),
                "route_snap_validated": snap_audit_path is not None,
            }
            candidates.append(candidate)
    return tuple(candidates)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _route_entity_coordinates(path: Path, *, method_id: str) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}

    def add(name: Any, latitude: Any, longitude: Any) -> None:
        entity_id = route_anchor_key(name)
        if not entity_id:
            return
        try:
            coordinates[entity_id] = (float(latitude), float(longitude))
        except (TypeError, ValueError):
            return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method") or "") != method_id:
                continue
            add(
                row.get("route_start_name"),
                row.get("route_start_latitude"),
                row.get("route_start_longitude"),
            )
            add(
                row.get("route_end_name"),
                row.get("route_end_latitude"),
                row.get("route_end_longitude"),
            )
            add(row.get("attraction_name"), row.get("latitude"), row.get("longitude"))
            add(row.get("hotel_name"), row.get("hotel_latitude"), row.get("hotel_longitude"))
    return coordinates


def _day_anchors(path: Path, *, method_id: str) -> tuple[dict[int, str], dict[int, str]]:
    starts: dict[int, set[str]] = {}
    ends: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method") or "") != method_id:
                continue
            day = int(row["day"])
            start = route_anchor_key(row.get("route_start_name"))
            end = route_anchor_key(row.get("route_end_name"))
            if start:
                starts.setdefault(day, set()).add(start)
            if end:
                ends.setdefault(day, set()).add(end)
    if not starts or not ends:
        raise ValueError(f"no route anchors found for parent method {method_id!r}")
    ambiguous = {
        day: {"starts": sorted(starts.get(day, set())), "ends": sorted(ends.get(day, set()))}
        for day in sorted(set(starts) | set(ends))
        if len(starts.get(day, set())) != 1 or len(ends.get(day, set())) != 1
    }
    if ambiguous:
        raise ValueError(f"ambiguous day anchors: {ambiguous}")
    return (
        {day: next(iter(values)) for day, values in starts.items()},
        {day: next(iter(values)) for day, values in ends.items()},
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _write_jsonl(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
