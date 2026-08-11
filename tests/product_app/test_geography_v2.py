from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from itinerary_system.product_app.geography_v2 import build_geographic_workspace_v2
from itinerary_system.product_app.product_demo import load_product_demo_package

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "runs" / "california-coast-product-demo-v2"


def _demo_geography() -> tuple[object, dict[str, object]]:
    package = load_product_demo_package(ROOT, DEMO)
    geography = build_geographic_workspace_v2(
        package.primary_bundle,
        additional_plans=package.additional_plans,
        route_legs_by_plan=package.route_legs_by_plan,
    )
    return package, geography


def test_demo_exposes_all_route_path_occurrences_and_validated_legs() -> None:
    package, geography = _demo_geography()
    plans = (
        package.primary_bundle.parent_plan,
        package.primary_bundle.child_plan,
        *(plan for plan, _label in package.additional_plans),
    )

    assert geography["schema_version"] == "product-geography-v2"
    assert geography["status"] == "ready"
    assert geography["coverage"] == {
        "status": "complete",
        "plan_count": 3,
        "route_path_node_count": 51,
        "required_leg_count": 48,
        "road_validated_leg_count": 48,
        "gap_count": 0,
        "all_itinerary_sequences_accounted": True,
    }
    assert geography["attribution"] == {
        "label": "© OpenStreetMap contributors",
        "url": "https://www.openstreetmap.org/copyright",
    }
    assert [(plan["plan_id"], plan["content_hash"]) for plan in geography["plans"]] == [
        (plan["plan_id"], plan["content_hash"]) for plan in plans
    ]

    for plan in geography["plans"]:
        assert plan["status"] == "ready"
        assert plan["coverage"] == {
            "schema_version": "route-coverage-v1",
            "status": "complete",
            "route_path_node_count": 17,
            "required_leg_count": 16,
            "road_validated_leg_count": 16,
            "gap_count": 0,
            "itinerary_sequence_accounted": True,
            "complete": True,
        }
        assert len(plan["route_path"]["features"]) == 17
        assert len(plan["stops"]["features"]) == 9
        assert len(plan["validated_legs"]["features"]) == 16
        assert plan["gaps"]["features"] == []
        assert all(
            feature["geometry"]["type"] == "LineString"
            and feature["properties"]["road_validated"] is True
            for feature in plan["validated_legs"]["features"]
        )
        assert all(
            feature["properties"]["itinerary_role"] is None
            and feature["properties"]["itinerary_role_source"] == "unavailable"
            for feature in plan["stops"]["features"]
        )


def test_malformed_itinerary_role_provenance_fails_geography_closed() -> None:
    package = load_product_demo_package(ROOT, DEMO)
    parent = deepcopy(package.primary_bundle.parent_plan)
    parent["selected_stops"][0]["itinerary_role"] = "meal"
    parent["selected_stops"][0]["itinerary_role_source"] = "place_category_inference"
    bundle = replace(package.primary_bundle, parent_plan=parent)

    geography = build_geographic_workspace_v2(
        bundle,
        additional_plans=package.additional_plans,
        route_legs_by_plan=package.route_legs_by_plan,
    )

    assert geography["status"] == "unavailable"
    assert geography["code"] == "itinerary_role_invalid"


def test_route_path_preserves_repeated_airport_and_hotel_anchors() -> None:
    _, geography = _demo_geography()
    nodes = [
        feature["properties"]
        for feature in geography["plans"][0]["route_path"]["features"]
    ]

    assert nodes[0]["node_id"] == "los_angeles_international_airport"
    assert nodes[-1]["node_id"] == "san_francisco_international_airport"
    hotel_occurrences = [
        node for node in nodes if node["node_id"] == "hotel_milo_santa_barbara"
    ]
    assert len(hotel_occurrences) == 3
    assert all(node["route_anchor"] is True for node in hotel_occurrences)
    assert sum(node["selected_stop"] is True for node in nodes) == 9


def test_missing_matrix_cell_keeps_valid_legs_and_adds_null_geometry_gaps() -> None:
    package = load_product_demo_package(ROOT, DEMO)
    route_matrix = deepcopy(package.primary_bundle.route_matrix)
    route_matrix["cells"] = [
        cell
        for cell in route_matrix["cells"]
        if not (
            cell["origin_id"] == "los_angeles_international_airport"
            and cell["destination_id"] == "hollywood_walk_of_fame"
        )
    ]
    bundle = replace(package.primary_bundle, route_matrix=route_matrix)

    geography = build_geographic_workspace_v2(
        bundle,
        additional_plans=package.additional_plans,
        route_legs_by_plan=package.route_legs_by_plan,
    )

    assert geography["status"] == "ready_with_gaps"
    assert geography["code"] == "artifact_geography_has_route_gaps"
    assert geography["coverage"]["road_validated_leg_count"] == 45
    assert geography["coverage"]["gap_count"] == 3
    for plan in geography["plans"]:
        assert plan["status"] == "ready_with_gaps"
        assert len(plan["route_path"]["features"]) == 17
        assert len(plan["validated_legs"]["features"]) == 15
        assert len(plan["gaps"]["features"]) == 1
        gap = plan["gaps"]["features"][0]
        assert gap["geometry"] is None
        assert gap["properties"]["failure_code"] == "route_leg_missing"
        assert gap["properties"]["validation_status"] == "unvalidated_gap"
        assert all(
            feature["geometry"]["type"] == "LineString"
            for feature in plan["validated_legs"]["features"]
        )


def test_route_path_accounts_for_selected_stops_in_their_exact_sequence() -> None:
    package, geography = _demo_geography()
    source_plans = {
        plan["plan_id"]: plan
        for plan in (
            package.primary_bundle.parent_plan,
            package.primary_bundle.child_plan,
            *(plan for plan, _label in package.additional_plans),
        )
    }

    for plan in geography["plans"]:
        selected_nodes = [
            feature["properties"]["node_id"]
            for feature in plan["route_path"]["features"]
            if feature["properties"]["selected_stop"]
        ]
        assert selected_nodes == source_plans[plan["plan_id"]]["sequence"]
        assert plan["coverage"]["itinerary_sequence_accounted"] is True
