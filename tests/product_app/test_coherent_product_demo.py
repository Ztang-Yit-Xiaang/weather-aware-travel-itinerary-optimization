from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.registry import ProductRunRegistry

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-coherent-demo-v1"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
APPROVED_DAY_LABELS = [
    "San Francisco",
    "Stanford & Silicon Valley",
    "Yosemite National Park",
    "Sequoia & Kings Canyon",
    "Central Coast",
    "Santa Barbara",
    "Los Angeles",
]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_coherent_package_is_reproducible_and_registered_without_replacing_research() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_coherent_product_demo.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    registry = ProductRunRegistry(ROOT, REGISTRY)
    records = {record.run_id: record for record in registry.all()}
    assert registry.default.run_id == "california_coast_coherent_demo_v1"
    assert registry.default.trip_id == "california_coast_approved_seven_day"
    assert records["california_coast_product_demo_v2"].default is False


def test_coherent_package_uses_one_trip_order_and_exact_alternative_requests() -> None:
    manifest = _json(RUN / "manifest.json")
    package = load_product_demo_package(ROOT, RUN)

    assert [day["label"] for day in manifest["scenario"]["days"]] == APPROVED_DAY_LABELS
    assert manifest["scenario"]["affected_stop_id"] == "bixby_creek_bridge_viewpoint"
    assert package.primary_bundle.parent_plan["plan_id"] == "plan_359968d255ff9288"
    assert [row["role"] for row in manifest["alternatives"]] == [
        "recommended",
        "low_driving",
    ]

    for row in manifest["alternatives"]:
        request = _json(RUN / row["request_relative_path"])
        diff = _json(RUN / row["diff_relative_path"])
        assert request["confirmed_constraints"]["target_stop_ids"] == ["bixby_creek_bridge_viewpoint"]
        assert len(request["candidate_pois"]) == 1
        assert request["candidate_pois"][0]["stop_id"] == row["replacement_stop_id"]
        assert [change["stop_id"] for change in diff["deleted_stops"]] == ["bixby_creek_bridge_viewpoint"]
        assert [change["stop_id"] for change in diff["added_stops"]] == [row["replacement_stop_id"]]


def test_coherent_route_evidence_is_complete_road_validated_and_nonfallback() -> None:
    manifest = _json(RUN / "manifest.json")
    matrix = _json(RUN / manifest["route_evidence"]["relative_path"])
    cells = {(cell["origin_id"], cell["destination_id"]): cell for cell in matrix["cells"]}

    for plan in [manifest["parent"], *manifest["alternatives"]]:
        legs = plan["route_legs"]
        assert len(legs) == 15
        for leg in legs:
            cell = cells[(leg["origin_id"], leg["destination_id"])]
            assert cell["road_validated"] is True
            assert cell["fallback_used"] is False
            assert cell["provider"] == "cached_osrm"
            assert cell["routing_status"] == "osrm_live"
            assert len(cell["geometry"]) >= 2


def test_default_api_session_exposes_the_coherent_run(tmp_path: Path) -> None:
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
    assert payload["registry"]["run_id"] == "california_coast_coherent_demo_v1"
    assert payload["session"]["accepted_plan_id"] == "plan_359968d255ff9288"
    assert [row["role"] for row in payload["workspace"]["alternatives"]] == [
        "recommended",
        "low_driving",
    ]
    assert len(payload["workspace"]["geography"]["plans"]) == 3
