from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


@pytest.fixture(autouse=True)
def ready_external_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRODUCT_ROUTING_BASE_URL", raising=False)
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_routing",
        lambda self: ComponentHealthV1("routing", "ready", False, "runtime_osrm_ready"),
    )


def test_workspace_api_exposes_plan_complete_geography_v2(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        response = client.get("/api/runs/california_coast_product_demo_v2/workspace")

    assert response.status_code == 200
    geography = response.json()["geography"]
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
    assert all("validated_legs" in plan and "gaps" in plan for plan in geography["plans"])
    assert all("routes" not in plan for plan in geography["plans"])


def test_created_session_returns_the_same_v2_workspace_contract(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        response = client.post("/api/sessions", json={})

    assert response.status_code == 200
    geography = response.json()["workspace"]["geography"]
    assert geography["schema_version"] == "product-geography-v2"
    assert geography["coverage"]["gap_count"] == 0

