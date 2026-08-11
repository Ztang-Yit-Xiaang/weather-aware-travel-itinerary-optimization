from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.route_coverage import audit_route_coverage

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "runs" / "california-coast-product-demo-v2"


def _cell(origin: str, destination: str) -> dict[str, object]:
    return {
        "origin_id": origin,
        "destination_id": destination,
        "route_leg_id": f"leg_{origin}_{destination}",
        "road_validated": True,
        "fallback_used": False,
        "geometry": [[44.0, -93.0], [44.1, -93.1]],
    }


def _plan() -> dict[str, object]:
    return {
        "plan_id": "plan_test",
        "sequence": ["a", "b", "c"],
        "selected_stops": [
            {"stop_id": "a", "day": 1},
            {"stop_id": "b", "day": 1},
            {"stop_id": "c", "day": 2},
        ],
    }


def test_route_coverage_accounts_for_every_ordered_stop_and_cross_day_leg() -> None:
    cells = {
        ("a", "b"): _cell("a", "b"),
        ("b", "c"): _cell("b", "c"),
    }

    report = audit_route_coverage(_plan(), None, cells)

    assert report.schema_version == "route-coverage-v1"
    assert report.required_leg_count == 2
    assert report.road_validated_leg_count == 2
    assert report.gap_count == 0
    assert report.itinerary_sequence_accounted is True
    assert report.complete is True
    assert report.legs[1].cross_day is True


def test_route_coverage_reports_missing_leg_without_fabricating_evidence() -> None:
    report = audit_route_coverage(_plan(), None, {("a", "b"): _cell("a", "b")})

    assert report.required_leg_count == 2
    assert report.road_validated_leg_count == 1
    assert report.gap_count == 1
    assert report.complete is False
    assert report.legs[1].validation_status == "unvalidated_gap"
    assert report.legs[1].failure_code == "route_leg_missing"
    assert report.legs[1].route_leg_id is None


def test_continuous_route_that_skips_a_selected_stop_is_incomplete() -> None:
    specs = (
        {
            "day": 1,
            "origin_id": "a",
            "destination_id": "c",
            "evidence_scope": "certified_daily_route_leg",
        },
    )
    cells = {("a", "c"): _cell("a", "c")}

    report = audit_route_coverage(_plan(), specs, cells)

    assert report.road_validated_leg_count == 1
    assert report.itinerary_sequence_accounted is False
    assert report.gap_count == 1
    assert report.complete is False


def test_product_demo_route_specs_account_for_all_selected_stops() -> None:
    package = load_product_demo_package(ROOT, DEMO)
    bundle = package.primary_bundle
    plans = (
        bundle.parent_plan,
        bundle.child_plan,
        *(plan for plan, _label in package.additional_plans),
    )
    cell_index = {
        (str(cell["origin_id"]), str(cell["destination_id"])): cell
        for cell in bundle.route_matrix["cells"]
    }

    reports = [
        audit_route_coverage(plan, package.route_legs_by_plan[plan["plan_id"]], cell_index)
        for plan in plans
    ]

    assert len(reports) == 3
    assert all(report.required_leg_count == 16 for report in reports)
    assert all(report.road_validated_leg_count == 16 for report in reports)
    assert all(report.itinerary_sequence_accounted for report in reports)
    assert all(report.complete for report in reports)

    broken_cells = deepcopy(cell_index)
    first = reports[0].legs[0]
    broken_cells.pop((first.origin_id, first.destination_id))
    broken = audit_route_coverage(
        plans[0],
        package.route_legs_by_plan[plans[0]["plan_id"]],
        broken_cells,
    )
    assert broken.required_leg_count == 16
    assert broken.road_validated_leg_count == 15
    assert broken.gap_count == 1
    assert broken.complete is False
