from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.geography import build_geographic_workspace
from itinerary_system.product_dashboard_adapter import load_product_dashboard_source

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "e3ux-weather-repair-demo-v6"
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def test_geography_is_geojson_derived_from_distinct_validated_plans() -> None:
    bundle = load_product_dashboard_source(RUN)

    geography = build_geographic_workspace(bundle)

    assert geography["schema_version"] == "product-geography-v1"
    assert geography["status"] == "ready"
    assert geography["route_matrix_id"] == bundle.route_matrix["matrix_id"]
    assert geography["route_source_sha256"] == bundle.route_matrix["source_content_sha256"]
    assert geography["bounds"] == [-122.478528, 34.101542, -118.289934, 37.8199]
    assert [plan["role"] for plan in geography["plans"]] == [
        "original",
        "registered_repair",
    ]
    assert len({plan["content_hash"] for plan in geography["plans"]}) == 2
    for plan in geography["plans"]:
        assert plan["stops"]["type"] == "FeatureCollection"
        assert plan["routes"]["type"] == "FeatureCollection"
        assert len(plan["stops"]["features"]) == 9
        assert len(plan["routes"]["features"]) == 8
        assert all(
            feature["properties"]["road_validated"] is True
            and feature["properties"]["fallback_used"] is False
            and feature["properties"]["geometry_source"] == "cached_osrm_route_geometry"
            for feature in plan["routes"]["features"]
        )

    first_coordinate = geography["plans"][0]["routes"]["features"][0]["geometry"][
        "coordinates"
    ][0]
    assert first_coordinate == [-118.326901, 34.101632]
    original_destination = geography["plans"][0]["routes"]["features"][-1]["properties"][
        "destination_id"
    ]
    repaired_destination = geography["plans"][1]["routes"]["features"][-1]["properties"][
        "destination_id"
    ]
    assert original_destination == "golden_gate_bridge"
    assert repaired_destination == "bixby_creek_bridge_viewpoint"


def test_geography_fails_closed_when_route_evidence_is_not_road_validated() -> None:
    bundle = load_product_dashboard_source(RUN)
    route_matrix = deepcopy(bundle.route_matrix)
    route_matrix["cells"][0]["road_validated"] = False
    target = next(
        cell
        for cell in route_matrix["cells"]
        if cell["origin_id"] == "hollywood_walk_of_fame"
        and cell["destination_id"] == "tcl_chinese_theatre"
    )
    target["road_validated"] = False

    geography = build_geographic_workspace(replace(bundle, route_matrix=route_matrix))

    assert geography["status"] == "unavailable"
    assert geography["code"] == "route_leg_not_road_validated"
    assert geography["plans"] == []


def test_workspace_api_exposes_geography_and_registered_w3_candidate_drafts(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        workspace = client.get("/api/runs/california_coast_product_demo_v2/workspace")
        assert workspace.status_code == 200
        geography = workspace.json()["geography"]
        assert geography["status"] == "ready"
        assert geography["schema_version"] == "product-geography-v2"
        assert len(geography["plans"]) == 3
        for plan in geography["plans"]:
            route_features = sorted(
                plan["validated_legs"]["features"],
                key=lambda feature: feature["properties"]["leg_index"],
            )
            assert plan["coverage"]["schema_version"] == "route-coverage-v1"
            assert plan["coverage"]["status"] == "complete"
            assert plan["coverage"]["complete"] is True
            assert plan["coverage"]["itinerary_sequence_accounted"] is True
            assert plan["coverage"]["route_path_node_count"] == 17
            assert plan["coverage"]["required_leg_count"] == 16
            assert plan["coverage"]["road_validated_leg_count"] == 16
            assert plan["coverage"]["gap_count"] == 0
            assert plan["gaps"]["features"] == []
            assert all(
                previous["properties"]["destination_id"]
                == current["properties"]["origin_id"]
                for previous, current in zip(
                    route_features[:-1], route_features[1:], strict=False
                )
            )
            day_four_first = next(
                feature for feature in route_features if feature["properties"]["day"] == 4
            )
            assert day_four_first["properties"]["origin_id"] == "the_line_la"

        session_response = client.post(
            "/api/sessions", json={"run_id": "california_coast_product_demo_v2"}
        )
        assert session_response.status_code == 200
        payload = session_response.json()
        assert payload["workspace"]["geography"] == geography
        session = payload["session"]
        token = payload["mutation_token"]
        parent_stop = client.post(
            f"/api/sessions/{session['session_id']}/selection",
            headers={"X-Session-Token": token},
            json={
                "expected_revision": 0,
                "selected_day": 7,
                "selected_stop_id": "golden_gate_bridge",
            },
        )
        assert parent_stop.status_code == 200
        session = parent_stop.json()["session"]
        route_feature = geography["plans"][1]["validated_legs"]["features"][0]
        repair_plan_id = geography["plans"][1]["plan_id"]
        selected_route = client.post(
            f"/api/sessions/{session['session_id']}/selection",
            headers={"X-Session-Token": token},
            json={
                "expected_revision": session["revision"],
                "selected_segment_id": route_feature["properties"]["route_leg_id"],
                "selected_alternative_id": repair_plan_id,
            },
        )
        assert selected_route.status_code == 200
        session = selected_route.json()["session"]
        drafted = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers={"X-Session-Token": token},
            json={
                "expected_revision": session["revision"],
                "type": "add_candidate",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"day": 7},
            },
        )
        assert drafted.status_code == 200
        assert drafted.json()["operation"]["type"] == "add_candidate"
        assert drafted.json()["session"]["revision"] == session["revision"] + 1
