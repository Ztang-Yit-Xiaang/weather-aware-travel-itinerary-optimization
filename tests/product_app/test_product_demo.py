from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app import product_demo
from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.geography import build_geographic_workspace
from itinerary_system.product_app.product_demo import load_product_demo_package

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def test_route_validation_rejects_a_cross_day_continuity_gap() -> None:
    route_matrix = {
        "cells": [
            {"origin_id": "a", "destination_id": "b", "duration_s": 60, "road_validated": True, "fallback_used": False},
            {"origin_id": "c", "destination_id": "d", "duration_s": 60, "road_validated": True, "fallback_used": False},
        ]
    }
    declared = [
        {"day": 1, "origin_id": "a", "destination_id": "b", "evidence_scope": "certified_daily_route_leg"},
        {"day": 2, "origin_id": "c", "destination_id": "d", "evidence_scope": "certified_daily_route_leg"},
    ]

    with pytest.raises(product_demo.ProductDemoError, match="product_demo_route_discontinuous"):
        product_demo._validate_route_legs(declared, route_matrix)


def test_demo_package_has_two_distinct_eligible_children_and_three_map_plans() -> None:
    package = load_product_demo_package(ROOT, DEMO)

    assert [row["role"] for row in package.alternatives] == ["recommended", "low_driving"]
    assert len({row["plan_id"] for row in package.alternatives}) == 2
    assert len({row["plan_content_hash"] for row in package.alternatives}) == 2
    assert all(row["ranking_eligible"] and row["route_validated"] for row in package.alternatives)
    assert package.alternatives[1]["route_total_minutes"] < package.alternatives[0][
        "route_total_minutes"
    ]

    geography = build_geographic_workspace(
        package.primary_bundle,
        additional_plans=package.additional_plans,
        route_legs_by_plan=package.route_legs_by_plan,
    )
    assert geography["status"] == "ready"
    assert [plan["role"] for plan in geography["plans"]] == [
        "original",
        "registered_repair",
        "alternative",
    ]
    assert len({plan["plan_id"] for plan in geography["plans"]}) == 3
    assert all(
        route["properties"]["road_validated"] is True
        and route["properties"]["fallback_used"] is False
        for plan in geography["plans"]
        for route in plan["routes"]["features"]
    )
    assert all(len(plan["routes"]["features"]) == 16 for plan in geography["plans"])
    scopes_by_role = {
        plan["role"]: {route["properties"]["evidence_scope"] for route in plan["routes"]["features"]}
        for plan in geography["plans"]
    }
    assert scopes_by_role["original"] == {"road_validated_daily_route_leg"}
    assert scopes_by_role["registered_repair"] == {"certified_daily_route_leg"}
    assert scopes_by_role["alternative"] == {"certified_daily_route_leg"}
    assert all(
        evidence["methods"]["requested"]
        and evidence["methods"]["executed"]
        and evidence["source_hashes"]
        for evidence in package.evidence.values()
    )


@pytest.mark.parametrize(
    "diff",
    (
        {
            "added_stops": [{"stop_id": "x"}],
            "deleted_stops": [],
            "day_moves": [],
            "time_shifts": [],
            "reorder_changes": [],
            "lodging_changes": [],
            "road_changes": [],
            "weighted_edit_cost": 1.0,
        },
        {
            "added_stops": [],
            "deleted_stops": [],
            "day_moves": [],
            "time_shifts": [],
            "reorder_changes": [],
            "lodging_changes": [],
            "road_changes": [],
        },
    ),
)
def test_demo_diff_validation_rejects_missing_cost_metrics(diff: dict[str, object]) -> None:
    with pytest.raises(product_demo.ProductDemoError, match="product_demo_diff_cost_invalid"):
        product_demo._validate_diff(diff)


def test_default_product_session_exposes_registered_three_plan_demo(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        response = client.post("/api/sessions", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["registry"]["run_id"] == "california_coast_product_demo_v2"
    assert payload["session"]["accepted_plan_id"] == "plan_e1c4f803691e3188"
    geography = payload["workspace"]["geography"]
    alternatives = payload["workspace"]["alternatives"]
    assert len(geography["plans"]) == 3
    assert len(alternatives) == 2
    assert {row["plan_id"] for row in alternatives} == {
        "plan_f5ee52459659dcb5",
        "plan_8aa919c8323dbac0",
    }
    assert set(payload["workspace"]["alternative_evidence"]) == {
        "plan_f5ee52459659dcb5",
        "plan_8aa919c8323dbac0",
    }


def test_evidence_download_is_bound_to_the_requested_alternative(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    plan_id = "plan_8aa919c8323dbac0"
    with TestClient(app) as client:
        response = client.get(
            f"/api/runs/california_coast_product_demo_v2/evidence-bundle?plan_id={plan_id}"
        )
        missing = client.get("/api/runs/california_coast_product_demo_v2/evidence-bundle")

    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
    assert f"plans/{plan_id}.json" in names
    assert "evaluations/cert_5a6deef4c159d346.json" in names
    assert "plans/plan_f5ee52459659dcb5.json" not in names
    assert "evaluations/cert_686ef65d376b2867.json" not in names
    assert missing.status_code == 400
    assert missing.json() == {"detail": "evidence_plan_required"}


def test_redacted_config_rejects_absolute_windows_host_path(tmp_path: Path) -> None:
    relative = "alternatives/example/resolved_config.redacted.json"
    config_path = tmp_path / relative
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"_source_path": r"F:\private\repository\config.yaml"}),
        encoding="utf-8",
    )
    with pytest.raises(product_demo.ProductDemoError, match="product_demo_host_path_disclosure"):
        product_demo._validate_redacted_config_paths(tmp_path, {relative: "0" * 64})
