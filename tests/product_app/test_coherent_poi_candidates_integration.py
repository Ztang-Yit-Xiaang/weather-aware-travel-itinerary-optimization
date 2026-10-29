from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.poi_catalog import ProductPOICatalogV1

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
RUN = ROOT / "runs" / "california-coast-coherent-demo-v1"
CATALOG = (
    ROOT
    / "configs"
    / "product_poi_catalogs"
    / "california_coast_approved_seven_day"
)
EXPECTED_CANDIDATES = {
    "pfeiffer_big_sur_state_park",
    "point_lobos_state_natural_reserve",
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_coherent_route_matrix_and_candidate_catalog_regenerate_exactly() -> None:
    for script in (
        "scripts/build_coherent_product_demo.py",
        "scripts/build_coherent_product_poi_catalog.py",
    ):
        completed = subprocess.run(
            [sys.executable, script, "--check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    manifest = _json(RUN / "manifest.json")
    matrix = _json(RUN / manifest["route_evidence"]["relative_path"])
    referenced_entities = {
        entity_id
        for cell in matrix["cells"]
        for entity_id in (cell["origin_id"], cell["destination_id"])
    }
    assert set(matrix["entity_ids"]) == referenced_entities
    assert all(len(cell["query_hash"]) == 16 for cell in matrix["cells"])

    catalog = ProductPOICatalogV1.load(CATALOG / "manifest.json")
    assert catalog.catalog_id == "california_coast_approved_seven_day_candidates_v1"
    assert {place.place_id for place in catalog.places} == EXPECTED_CANDIDATES
    assert all(place.recommended_visit_minutes == 45 for place in catalog.places)
    assert all(
        access.source_ref == matrix["matrix_id"]
        for place in catalog.places
        for access in place.access_points
    )


def test_default_coherent_api_returns_exact_registered_day5_replacements(
    tmp_path: Path,
) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200
        created_payload = created.json()
        session = created_payload["session"]
        response = client.get(
            f"/api/sessions/{session['session_id']}/map/poi-candidates",
            headers={"X-Session-Token": created_payload["mutation_token"]},
            params={
                "expected_revision": session["revision"],
                "day": 5,
                "route_leg_id": (
                    "leg_comfort_inn_sequoia_kings_canyon_"
                    "bixby_creek_bridge_viewpoint"
                ),
                "replacement_target_id": "bixby_creek_bridge_viewpoint",
                "max_detour_minutes": 60,
                "limit": 10,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["kind"] == "replacement"
    assert payload["context"]["day"] == 5
    assert payload["context"]["replacement_target_id"] == (
        "bixby_creek_bridge_viewpoint"
    )
    assert payload["routing"] == {
        "matrix_id": "route_matrix_b33928f302ba9055",
        "context_snapshot_id": "context_coherent_california_demo_2026_08",
        "source_bundle_id": "route_bundle_e5b6cdc2ed2fff30",
        "source_content_sha256": (
            "e5b6cdc2ed2fff301c5416e29f4e6f0d9a3df4f7b3661311ce24e28cef108d99"
        ),
        "road_validated_only": True,
        "fallback_allowed": False,
    }
    candidates = {row["candidate_id"]: row for row in payload["candidates"]}
    assert set(candidates) == EXPECTED_CANDIDATES
    assert all(row["burden"]["evaluated_feasible"] is True for row in candidates.values())
    assert all(row["burden"]["ranking_eligible"] is False for row in candidates.values())
    assert candidates["pfeiffer_big_sur_state_park"]["registered_replacement"] == {
        "draft_type": "replace_nearby",
        "target_stop_id": "bixby_creek_bridge_viewpoint",
        "candidate_id": "pfeiffer_big_sur_state_park",
    }
    assert candidates["point_lobos_state_natural_reserve"]["burden"][
        "marginal_travel_minutes"
    ] < 0