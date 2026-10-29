from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import (
    DeterministicCopilotAdapter,
    known_place_clarification,
)
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.poi_catalog import ProductPOICatalogV1
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.region_scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = ROOT / "configs" / "product_poi_catalogs" / "california_statewide"
BUILDER_PATH = ROOT / "scripts" / "build_california_statewide_poi_catalog.py"
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def _builder():
    spec = importlib.util.spec_from_file_location("statewide_poi_builder", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_statewide_catalog_is_deterministic_bounded_and_discovery_only() -> None:
    builder = _builder()
    catalog_bytes, manifest_bytes = builder.build_catalog_documents()
    assert catalog_bytes == (CATALOG_ROOT / "catalog.json").read_bytes()
    assert manifest_bytes == (CATALOG_ROOT / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["catalog_sha256"] == hashlib.sha256(catalog_bytes).hexdigest()

    catalog = ProductPOICatalogV1.load(CATALOG_ROOT / "manifest.json")
    assert catalog.catalog_id == "california_statewide_discovery_v1"
    scenario_names = (
        set(SCENARIOS["california_statewide_nature"].coordinate_fallbacks)
        - {"Las Vegas"}
    )
    assert len(catalog.places) == 41
    assert {place.name for place in catalog.places} == scenario_names
    assert all(not place.has_road_validated_access for place in catalog.places)
    assert all(place.access_points == () for place in catalog.places)
    assert {"ca_region_redwood", "ca_region_yosemite", "ca_region_death_valley"} <= {
        place.place_id for place in catalog.places
    }
    assert {"ca_gateway_san_diego", "ca_gateway_sacramento", "ca_gateway_crescent_city"} <= {
        place.place_id for place in catalog.places
    }


def test_statewide_catalog_resolves_human_aliases_without_route_claims() -> None:
    catalog = ProductPOICatalogV1.load(CATALOG_ROOT / "manifest.json")
    for query in ("redwood", "red wood", "redwoods"):
        matches = catalog.search(query, limit=5)
        assert [place.place_id for place in matches] == ["ca_region_redwood"]
    assert catalog.search("LA", limit=1)[0].place_id == "ca_gateway_los_angeles"
    assert catalog.search("redwood", require_road_access=True) == ()
    redwood = catalog.get_place("ca_region_redwood")
    assert redwood is not None
    assert redwood.region_id == "redwood"
    assert "Red Wood" in redwood.aliases
    assert redwood.official_url is None
    assert redwood.recommended_visit_minutes is None


def test_builder_refuses_writes_outside_catalog_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_outside_catalog_root"):
        _builder().write_catalog(tmp_path / "outside")


def test_live_product_context_distinguishes_accepted_registered_and_discovery_places(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()["session"]
        service = app.state.product_service
        context = service.copilot_context(service.workspace.get(session["session_id"]))

    places = {place["place_id"]: place for place in context.known_places}
    assert len(places) <= 64
    assert places["ca_region_redwood"]["route_status"] == (
        "discovery_only_route_validation_required"
    )
    assert places["yosemite_national_park"]["route_status"] == "accepted_stop"
    assert places["pfeiffer_big_sur_state_park"]["route_status"] == (
        "registered_route_candidate"
    )
    assert "Yosemite" in places["yosemite_national_park"]["aliases"]
    assert "Bixby Creek Bridge" in places["bixby_creek_bridge_viewpoint"]["aliases"]
    bixby = known_place_clarification(context, "I want to go to Bixby Creek Bridge")
    assert bixby is not None
    assert "already in this itinerary" in bixby[0]
    assert known_place_clarification(context, "what if I want to visit red wood") == (
        "I found Redwood National and State Parks in the California statewide discovery "
        "catalog, but its road access and route are not validated for this trip. Do you "
        "want to evaluate adding it or replacing a stop, and which day should I check? "
        "Nothing was changed.",
        "I found Redwood National and State Parks in the California statewide discovery "
        "catalog, but its road access and route are not validated for this trip. Do you "
        "want to evaluate adding it or replacing a stop, and which day should I check?",
    )
    accepted = known_place_clarification(context, "Could I visit Yosemite?")
    assert accepted is not None
    assert "already in this itinerary" in accepted[0]


def test_deterministic_copilot_uses_specific_statewide_clarification() -> None:
    import asyncio

    from itinerary_system.product_app.models import CopilotContextV1

    context = CopilotContextV1(
        run_id="run",
        trip_id="trip",
        session_id="session",
        session_revision=0,
        accepted_plan_id="plan",
        selected_day=5,
        selected_stop_id=None,
        selected_segment_id=None,
        selected_candidate_id=None,
        selected_alternative_id=None,
        draft_operations=(),
        evaluated_proposal=None,
        allowed_stop_ids=(),
        allowed_candidate_ids=(),
        allowed_days=(1, 2, 3, 4, 5, 6, 7),
        known_places=(
            {
                "place_id": "ca_region_redwood",
                "name": "Redwood National and State Parks",
                "aliases": ("Redwood", "Redwoods", "Red Wood"),
                "categories": ("statewide_discovery",),
                "region_id": "redwood",
                "route_status": "discovery_only_route_validation_required",
            },
        ),
    )
    result = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=context, history=(), message="I want to visit red wood"
        )
    )
    assert result.state == "clarification_required"
    assert "Redwood National and State Parks" in result.assistant_message
    assert "not validated for this trip" in result.assistant_message
    assert result.intents[0].type == "request_clarification"

def test_authenticated_statewide_search_is_truthful_and_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        path = f"/api/sessions/{session['session_id']}/poi/search?q=red%20wood&limit=5"
        assert client.get(path).status_code == 403
        response = client.get(
            path, headers={"X-Session-Token": created["mutation_token"]}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "product-poi-statewide-discovery-v1"
        assert payload["session_revision"] == 0
        assert payload["catalog"]["scope"] == "bounded_california_statewide_discovery"
        assert [place["place_id"] for place in payload["places"]] == [
            "ca_region_redwood"
        ]
        place = payload["places"][0]
        assert place["name"] == "Redwood National and State Parks"
        assert place["route_status"] == "route_validation_required"
        assert place["road_validated_access"] is False
        assert place["recommendation_status"] == "not_evaluated"
        assert client.get(
            path + "&unknown=x",
            headers={"X-Session-Token": created["mutation_token"]},
        ).json() == {"detail": "poi_search_query_field_unknown"}

