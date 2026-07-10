from __future__ import annotations

import copy
import json
from pathlib import Path

from itinerary_system.benchmark import (
    DisruptionFamily,
    DisruptionGenerator,
    generate_disruption_requests,
    generate_disruption_scenarios,
)
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2

REPO_ROOT = Path(__file__).resolve().parents[2]


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_benchmark",
        source_run_id="run_parent",
        planning_request_id="request_parent",
        catalog_snapshot_id="catalog_benchmark",
        context_snapshot_id="context_benchmark",
        selected_stops=(
            {
                "stop_id": "poi_museum",
                "name": "Museum",
                "day": 1,
                "stop_order": 1,
                "lodging_id": "hotel_a",
                "weather_sensitivity": 0.1,
            },
            {
                "stop_id": "poi_bridge",
                "name": "Bridge",
                "day": 2,
                "stop_order": 1,
                "lodging_id": "hotel_b",
                "weather_sensitivity": 0.9,
                "weather_risk": 0.4,
                "route_id": "route_2",
            },
            {
                "stop_id": "poi_grove",
                "name": "Grove",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_b",
                "weather_sensitivity": 0.8,
                "weather_risk": 0.2,
                "route_id": "route_2",
            },
        ),
        day_assignments={"poi_museum": 1, "poi_bridge": 2, "poi_grove": 2},
        sequence=("poi_museum", "poi_bridge", "poi_grove"),
        lodging_assignments={"1": "hotel_a", "2": "hotel_b"},
        ordered_days=({"day": 1, "stop_ids": ("poi_museum",)}, {"day": 2, "stop_ids": ("poi_bridge", "poi_grove")}),
        route_ids_by_day={1: "route_1", 2: "route_2"},
        created_at="2026-07-09T00:00:00+00:00",
    )


def test_generate_disruption_scenarios_has_six_deterministic_families_and_metadata():
    first = generate_disruption_scenarios(parent_plan(), seed=7)
    second = generate_disruption_scenarios(parent_plan(), seed=7)

    assert tuple(scenario.family for scenario in first) == tuple(DisruptionFamily)
    assert [scenario.scenario_id for scenario in first] == [scenario.scenario_id for scenario in second]
    assert len({scenario.scenario_id for scenario in first}) == 6
    assert all(scenario.evidence_status == "synthetic" for scenario in first)
    assert all(scenario.request.confirmed_constraints["observation_status"] == "synthetic" for scenario in first)
    assert all(scenario.request.confirmed_constraints["disruption_family"] == scenario.family.value for scenario in first)
    assert all(scenario.request.confirmed_constraints["parent_plan_id"] == "parent_benchmark" for scenario in first)


def test_generate_disruption_requests_returns_repair_requests_without_mutating_parent():
    parent = parent_plan()
    before = copy.deepcopy(parent.to_record())

    requests = generate_disruption_requests(parent, seed=11)

    assert parent.to_record() == before
    assert len(requests) == 6
    assert all(isinstance(request, RepairRequest) for request in requests)
    assert all(request.baseline_route == parent.selected_stops for request in requests)
    assert {request.confirmed_constraints["disruption_family"] for request in requests} == {
        family.value for family in DisruptionFamily
    }


def test_family_specific_constraints_are_repair_ready():
    scenarios = {scenario.family: scenario for scenario in DisruptionGenerator(seed=3).generate(parent_plan())}

    assert scenarios[DisruptionFamily.WEATHER_DETERIORATION].request.confirmed_constraints["weather_risk_overrides"] == {
        "poi_bridge": 0.95
    }
    assert scenarios[DisruptionFamily.ROAD_CLOSURE].request.confirmed_constraints["closed_route_ids"] == ("route_2",)
    assert scenarios[DisruptionFamily.HOTEL_UNAVAILABILITY].request.confirmed_constraints["unavailable_lodging_ids"] == (
        "hotel_b",
    )
    assert scenarios[DisruptionFamily.ATTRACTION_CLOSURE].request.confirmed_constraints["must_delete"] == ("poi_grove",)
    assert (
        scenarios[DisruptionFamily.REDUCED_DRIVING_TOLERANCE].request.tolerance_profile["max_daily_travel_minutes"]
        == 180.0
    )
    must_visit = scenarios[DisruptionFamily.NEW_MUST_VISIT].request.confirmed_constraints["must_include"]
    assert must_visit == ("bench_must_visit_day_2",)
    assert scenarios[DisruptionFamily.NEW_MUST_VISIT].request.candidate_pois[0]["stop_id"] == "bench_must_visit_day_2"


def test_static_disruption_family_manifest_matches_generator_contract():
    manifest_path = REPO_ROOT / "data" / "benchmark" / "disruptions" / "bench_001_families.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "benchmark-disruption-family-manifest-v1"
    assert tuple(family["family"] for family in manifest["families"]) == tuple(family.value for family in DisruptionFamily)
    assert all(family["default_evidence_status"] == "synthetic" for family in manifest["families"])
