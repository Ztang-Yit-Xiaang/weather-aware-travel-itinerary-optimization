from types import SimpleNamespace

from itinerary_system.benchmark import build_benchmark_route_coverage
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def _cell(origin: str, destination: str) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=1000,
        duration_s=600,
        road_validated=True,
        provider="unit",
    )


def _parent() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_route_coverage",
        source_run_id="run_route_coverage",
        planning_request_id="request_route_coverage",
        catalog_snapshot_id="catalog_route_coverage",
        context_snapshot_id="context_route_coverage",
        selected_stops=({"stop_id": "a", "latitude": 1.0, "longitude": 2.0, "day": 1},),
        day_assignments={"a": 1},
        sequence=("a",),
    )


def _scenario(candidate: dict) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id="scenario_route_coverage",
        request=SimpleNamespace(candidate_pois=(candidate,)),
    )


def test_route_coverage_reports_complete_conservative_search_universe():
    pairs = {
        ("start", "end"),
        ("start", "a"),
        ("start", "b"),
        ("a", "end"),
        ("b", "end"),
        ("a", "b"),
        ("b", "a"),
    }
    matrix = RouteMatrix(
        matrix_id="matrix_route_coverage",
        context_snapshot_id="context_route_coverage",
        entity_ids=(),
        cells={pair: _cell(*pair) for pair in pairs},
    )

    report = build_benchmark_route_coverage(
        parent_plan=_parent(),
        scenarios=(_scenario({"stop_id": "b", "latitude": 3.0, "longitude": 4.0}),),
        route_matrix=matrix,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        entity_coordinates={"start": (0.0, 0.0), "end": (5.0, 5.0)},
    )

    assert report.publication_ready is True
    assert report.required_pair_count == 7
    assert report.road_validated_pair_count == 7
    assert report.missing_pairs == ()
    assert report.unlocated_entity_ids == ()


def test_route_coverage_retains_missing_pairs_and_unlocated_candidates():
    matrix = RouteMatrix(
        matrix_id="matrix_route_coverage_missing",
        context_snapshot_id="context_route_coverage",
        entity_ids=(),
        cells={("start", "a"): _cell("start", "a")},
    )

    report = build_benchmark_route_coverage(
        parent_plan=_parent(),
        scenarios=(_scenario({"stop_id": "b"}),),
        route_matrix=matrix,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        entity_coordinates={"start": (0.0, 0.0), "end": (5.0, 5.0)},
    )

    assert report.publication_ready is False
    assert ("a", "b") in report.missing_pairs
    assert ("b", "a") in report.missing_pairs
    assert report.unlocated_entity_ids == ("b",)
    record = report.to_record()
    assert record["missing_pair_count"] == 6
    assert record["unlocated_entity_ids"] == ["b"]
