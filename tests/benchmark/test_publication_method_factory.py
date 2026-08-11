import json

from itinerary_system.benchmark import (
    DisruptionFamily,
    DisruptionScenario,
    build_publication_benchmark_method_adapters,
    generate_disruption_scenarios,
    run_benchmark_suite,
)
from itinerary_system.repair import DayRouteSolverConfig
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_publication_factory",
        source_run_id="run_parent_publication_factory",
        planning_request_id="request_parent_publication_factory",
        catalog_snapshot_id="catalog_publication_factory",
        context_snapshot_id="context_publication_factory",
        selected_stops=(
            {"stop_id": "outdoor", "name": "Outdoor", "day": 1, "lodging_id": "hotel", "utility": 8},
            {"stop_id": "museum", "name": "Museum", "day": 1, "lodging_id": "hotel", "utility": 7},
        ),
        day_assignments={"outdoor": 1, "museum": 1},
        sequence=("outdoor", "museum"),
        lodging_assignments={"1": "hotel"},
        ordered_days=({"day": 1, "stop_ids": ("outdoor", "museum")},),
        route_ids_by_day={1: "route_parent"},
        created_at="2026-07-13T00:00:00+00:00",
    )


def scenario() -> DisruptionScenario:
    request = RepairRequest(
        request_id="repair_publication_factory",
        baseline_route=parent_plan().selected_stops,
        user_intent="repair weather disruption",
        confirmed_constraints={
            "parent_plan_id": "parent_publication_factory",
            "affected_days": (1,),
            "weather_risk_overrides": {"outdoor": 0.95},
            "weather_feasible": {"outdoor": False},
        },
        candidate_pois=(
            {"stop_id": "indoor", "name": "Indoor", "day": 1, "lodging_id": "hotel", "utility": 6},
        ),
    )
    return DisruptionScenario(
        scenario_id="scenario_publication_factory",
        family=DisruptionFamily.WEATHER_DETERIORATION,
        parent_plan_id="parent_publication_factory",
        catalog_snapshot_id="catalog_publication_factory",
        context_snapshot_id="context_publication_factory",
        seed=1,
        evidence_status="synthetic",
        affected_days=(1,),
        target_stop_ids=("outdoor",),
        request=request,
    )


def matrix(*, duration_s: float = 300) -> RouteMatrix:
    ids = ("start", "outdoor", "museum", "indoor", "end")
    cells = {
        (origin, destination): RouteMatrixCell(
            origin_id=origin,
            destination_id=destination,
            distance_m=5000,
            duration_s=duration_s,
            provider="unit",
            road_validated=True,
        )
        for origin in ids
        for destination in ids
        if origin != destination
    }
    return RouteMatrix(
        matrix_id="matrix_publication_factory",
        context_snapshot_id="context_publication_factory",
        entity_ids=ids,
        cells=cells,
        source_bundle_id="synthetic_publication_fixture_bundle",
        source_content_sha256="a" * 64,
    )


def route_config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        default_visit_minutes=30,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        strict_route_matrix=True,
    )


def test_four_concrete_adapters_run_together_and_retain_ineligible_baseline(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  catalog_snapshot_id: catalog_publication_factory\n  context_snapshot_id: context_publication_factory\n",
        encoding="utf-8",
    )
    methods = build_publication_benchmark_method_adapters(
        parent_plan=parent_plan(),
        route_matrix=matrix(),
        config_path=config_path,
        output_root=tmp_path / "pipeline_runs",
        day_route_config=route_config(),
        publication_mode=True,
        strict=True,
    )

    result = run_benchmark_suite(
        scenarios=(scenario(),),
        methods=methods,
        output_dir=tmp_path / "benchmark",
        publication_mode=True,
    )

    rows = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    by_method = {row["method_id"]: row for row in rows}
    assert set(by_method) == {
        "context_blind_solver",
        "deterministic_context_aware_heuristic",
        "progressive_sequential_lexicographic_repair",
        "full_reoptimization",
    }
    assert all(row["benchmark_method_provenance_valid"] for row in rows)
    assert by_method["context_blind_solver"]["status"] == "failed_strict"
    assert by_method["context_blind_solver"]["benchmark_ranking_eligible"] is False
    assert by_method["deterministic_context_aware_heuristic"]["benchmark_ranking_eligible"] is True
    assert by_method["progressive_sequential_lexicographic_repair"]["benchmark_ranking_eligible"] is True
    assert by_method["full_reoptimization"]["benchmark_ranking_eligible"] is True
    assert all(row["output_plan_id"] != "parent_publication_factory" for row in rows)
    assert len({row["benchmark_route_matrix_hash"] for row in rows}) == 1
    assert {row["benchmark_route_source_bundle_id"] for row in rows} == {
        "synthetic_publication_fixture_bundle"
    }
    assert by_method["deterministic_context_aware_heuristic"]["quality_weather_risk_delta"] == 0.95
    assert by_method["progressive_sequential_lexicographic_repair"]["quality_weather_risk_delta"] == 0.95

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    readiness = manifest["publication_readiness"]
    assert readiness["failed_run_count"] == 1
    assert readiness["ranking_eligible_run_count"] == 3
    assert readiness["route_input_consistency_complete"] is True
    assert readiness["route_source_bundle_consistency_complete"] is True
    assert readiness["publication_ready"] is True


def test_non_exact_adapters_retain_expected_physical_infeasibility_evidence(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  catalog_snapshot_id: catalog_publication_factory\n  context_snapshot_id: context_publication_factory\n",
        encoding="utf-8",
    )
    target_families = {
        DisruptionFamily.ROAD_CLOSURE,
        DisruptionFamily.REDUCED_DRIVING_TOLERANCE,
    }
    scenarios = tuple(
        item
        for item in generate_disruption_scenarios(parent_plan(), seed=19)
        if item.family in target_families
    )
    methods = tuple(
        method
        for method in build_publication_benchmark_method_adapters(
            parent_plan=parent_plan(),
            route_matrix=matrix(duration_s=2700),
            config_path=config_path,
            output_root=tmp_path / "pipeline_runs",
            day_route_config=route_config(),
            publication_mode=True,
            strict=True,
        )
        if method.method_id
        in {
            "deterministic_context_aware_heuristic",
            "progressive_sequential_lexicographic_repair",
        }
    )

    result = run_benchmark_suite(
        scenarios=scenarios,
        methods=methods,
        output_dir=tmp_path / "benchmark",
        publication_mode=False,
    )

    rows = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["benchmark_ranking_eligible"] is False for row in rows)
    assert all(row["output_plan_id"] == "" for row in rows)
    assert all(row["benchmark_method_provenance_valid"] for row in rows)

    dashboard_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "pipeline_runs").rglob("*.json")
        if path.parent.name == "dashboard"
    ]
    evidence_by_request: dict[str, list[str]] = {}
    for record in dashboard_records:
        request_id = str(record.get("request_id") or record.get("repair_request_id"))
        evidence_by_request.setdefault(request_id, []).append(json.dumps(record, sort_keys=True))
    road_request = next(item.request.request_id for item in scenarios if item.family == DisruptionFamily.ROAD_CLOSURE)
    tolerance_request = next(
        item.request.request_id
        for item in scenarios
        if item.family == DisruptionFamily.REDUCED_DRIVING_TOLERANCE
    )
    road_evidence = " ".join(evidence_by_request[road_request])
    tolerance_evidence = " ".join(evidence_by_request[tolerance_request])
    assert "context_closed_route_selected:route_parent" in road_evidence
    assert "day_time_exceeded:1" in tolerance_evidence
    assert "no_feasible_day_route_candidates" in road_evidence
    assert "no_feasible_day_route_candidates" in tolerance_evidence