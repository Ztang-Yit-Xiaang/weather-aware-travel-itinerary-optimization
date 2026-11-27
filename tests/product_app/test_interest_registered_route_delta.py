"""Interest exploration must accept the product's hash-bound additive route matrix."""

import math
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.candidate_discovery import POICandidateDiscoveryError, _route_matrix
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registered_delta(tmp_path_factory):
    assert Path(_route_matrix.__code__.co_filename).resolve().is_relative_to(ROOT / "src")
    app = create_product_app(
        repository_root=ROOT,
        state_root=tmp_path_factory.mktemp("registered-interest"),
        additional_allowed_authorities=("testserver",),
        copilot=DeterministicCopilotAdapter(),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200, created.text
        payload = created.json()
        service = app.state.product_runtime.require_service()
        session = service.workspace.get(payload["session"]["session_id"])
        parent, geography, matrix, scope = service.interest_plan_context(session)
        assert scope == "accepted"
        assert matrix["matrix_id"].startswith("route_matrix_delta_")
        assert matrix["source_bundle_id"].startswith("route_bundle_composite_")
        yield client, payload, service, session, parent, geography, matrix


@pytest.mark.parametrize("day", [1, 5, 6])
def test_registered_seven_day_interest_is_read_only(registered_delta, day):
    client, payload, _, session, parent, geography, matrix = registered_delta
    route = next(f["properties"] for f in geography["validated_legs"]["features"] if f["properties"]["day"] == day)
    before = deepcopy(session.public_dict())
    response = client.get(
        f"/api/sessions/{session.session_id}/map/interest-regions",
        headers={"X-Session-Token": payload["mutation_token"]},
        params={
            "expected_revision": session.revision,
            "day": day,
            "route_leg_id": route["route_leg_id"],
            "plan_id": parent["plan_id"],
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["plan_scope"] == "accepted"
    assert result["accepted_plan_id"] == parent["plan_id"]
    assert result["candidate_universe"]["truncated"] is False
    assert result["search_context"]["radius_km"] == 5
    assert all(point["distance_from_focus_km"] <= 5 for point in result["poi_predictions"])
    if result["poi_predictions"]:
        assert result["surface"]["status"] == "ready", result["surface"]
    else:
        assert result["surface"]["unavailable_reason"] == "no_optional_candidates"
    if day == 1:
        names = {p["name"] for p in result["poi_predictions"]}
        assert "Fort Point National Historic Site" in names
        assert "Point Reyes" not in names
    assert session.public_dict() == before
    assert matrix["source_content_sha256"][:16] == matrix["matrix_id"].removeprefix("route_matrix_delta_")


@pytest.mark.parametrize(
    "field,value",
    [
        ("matrix_id", "route_matrix_delta_0000000000000000"),
        ("source_bundle_id", "route_bundle_composite_0000000000000000"),
        ("source_content_sha256", "0" * 64),
        ("source_content_sha256", None),
        ("matrix_id", "arbitrary_matrix"),
    ],
)
def test_composite_matrix_identifiers_must_match_descriptor(registered_delta, field, value):
    record = {"schema_version": "route-matrix-v1", **deepcopy(registered_delta[-1])}
    record[field] = value
    with pytest.raises(POICandidateDiscoveryError, match="poi_route_matrix_invalid"):
        _route_matrix(record)


def test_registered_delta_keeps_global_duplicate_and_exact_geometry_guards(registered_delta):
    record = {"schema_version": "route-matrix-v1", **deepcopy(registered_delta[-1])}
    record["cells"].append(deepcopy(record["cells"][0]))
    with pytest.raises(POICandidateDiscoveryError, match="poi_route_matrix_invalid"):
        _route_matrix(record)
    record["cells"].pop()
    for cell in record["cells"]:
        cell["geometry"] = [[True, -122], [37, -122]]
    with pytest.raises(POICandidateDiscoveryError, match="poi_route_matrix_unavailable"):
        _route_matrix(record)


def _nearby_request(fixture, **query):
    client, payload, _, session, _, geography, _ = fixture
    day = query.pop("day", 1)
    leg = next(
        f["properties"]["route_leg_id"]
        for f in geography["validated_legs"]["features"]
        if f["properties"]["day"] == day
    )
    return client.get(
        f"/api/sessions/{session.session_id}/map/interest-regions",
        headers={"X-Session-Token": payload["mutation_token"]},
        params={"expected_revision": session.revision, "day": day, "route_leg_id": leg, **query},
    )


def test_current_stop_search_excludes_distant_point_reyes_and_radius_is_monotone(registered_delta):
    previous = set()
    for radius in [1, 2, 5, 10]:
        response = _nearby_request(registered_delta, radius_km=radius, focus_stop_id="golden_gate_bridge")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["search_context"]["name"] == "Golden Gate Bridge"
        points = result["poi_predictions"]
        assert all(p["distance_from_focus_km"] <= radius for p in points)
        assert all(p["name"] != "Point Reyes" for p in points)
        current = {p["place_id"] for p in points}
        assert previous <= current
        previous = current
    assert len(previous) >= 7


def test_density_and_all_region_boundaries_stay_inside_the_selected_radius(registered_delta):
    result = _nearby_request(registered_delta, radius_km=2).json()
    assert result["surface"]["status"] == "ready"
    origin = result["search_context"]["coordinate"]
    origin_lat, origin_lon = map(math.radians, [origin["latitude"], origin["longitude"]])
    for collection in ["density_geojson", "regions_geojson"]:
        for feature in result["surface"][collection]["features"]:
            for polygon in feature["geometry"]["coordinates"]:
                for ring in polygon:
                    for longitude, latitude in ring:
                        latitude, longitude = map(math.radians, [latitude, longitude])
                        a = (
                            math.sin((latitude - origin_lat) / 2) ** 2
                            + math.cos(latitude) * math.cos(origin_lat) * math.sin((longitude - origin_lon) / 2) ** 2
                        )
                        assert 2 * 6371.0088 * math.asin(math.sqrt(min(1, a))) <= 2.01


@pytest.mark.parametrize("radius", [0, 0.5, -1, 26, "nan", "inf"])
def test_nearby_radius_rejects_invalid_or_unbounded_queries(registered_delta, radius):
    response = _nearby_request(registered_delta, radius_km=radius)
    assert response.status_code == 422
    assert response.json()["detail"] == "interest_radius_invalid"


@pytest.mark.parametrize("stop", ["not_a_stop", "stearns_wharf"])
def test_focus_must_be_an_actual_stop_of_the_displayed_day(registered_delta, stop):
    response = _nearby_request(registered_delta, focus_stop_id=stop)
    assert response.status_code == 409
    assert response.json()["detail"] == "interest_focus_stop_invalid"


def test_two_stops_on_one_day_can_each_center_the_search(registered_delta):
    first = _nearby_request(registered_delta, day=4, focus_stop_id="sequoia_national_park")
    second = _nearby_request(registered_delta, day=4, focus_stop_id="kings_canyon_national_park")
    assert first.status_code == second.status_code == 200
    assert first.json()["search_context"]["stop_id"] == "sequoia_national_park"
    assert second.json()["search_context"]["stop_id"] == "kings_canyon_national_park"
    assert first.json()["search_context"]["coordinate"] != second.json()["search_context"]["coordinate"]


@pytest.mark.parametrize("day", range(1, 8))
def test_each_itinerary_day_has_points_inside_two_km(registered_delta, day):
    result = _nearby_request(registered_delta, day=day, radius_km=2)
    assert result.status_code == 200, result.text
    result = result.json()
    assert len(result["poi_predictions"]) >= 3
    assert result["surface"]["status"] == "ready", result["surface"]
    assert all(p["distance_from_focus_km"] <= 2 for p in result["poi_predictions"])
    if day in (3, 4):
        assert result["search_context"]["visitor_area"]
        assert "visitor area" in result["search_context"]["name"]


def test_large_park_search_is_explicit_and_original_pin_remains_available(registered_delta):
    nearby = _nearby_request(registered_delta, day=3, radius_km=2).json()
    original = _nearby_request(registered_delta, day=3, radius_km=2, search_area="itinerary_pin").json()
    assert nearby["search_context"]["area_mode"] == "visitor_area"
    assert original["search_context"]["area_mode"] == "itinerary_pin"
    assert nearby["search_context"]["coordinate"] != original["search_context"]["coordinate"]
    assert _nearby_request(registered_delta, day=3, search_area="invented").status_code == 422


def test_online_points_cache_and_outage_fallback_preserve_the_itinerary(registered_delta, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from itinerary_system.product_app import nearby_places as online

    _, _, service, session, _, _, _ = registered_delta
    monkeypatch.setattr(service, "nearby_places", online.NearbyPlaceCache(tmp_path / "online"))
    clock = [1800000000.0]
    monkeypatch.setattr(online, "time", SimpleNamespace(time=lambda: clock[0]))
    calls = []

    def fetch(query):
        calls.append(query)
        return {
            "elements": [
                {
                    "type": "node",
                    "id": 987654321099,
                    "lat": 37.8193,
                    "lon": -122.4787,
                    "tags": {"name": "API-only nearby museum", "tourism": "museum"},
                }
            ]
        }

    monkeypatch.setattr(online, "_fetch", fetch)
    before = deepcopy(session.public_dict())
    first = _nearby_request(registered_delta, radius_km=2)
    assert first.status_code == 200, first.text
    first = first.json()
    point = next(p for p in first["poi_predictions"] if p["name"] == "API-only nearby museum")
    assert point["feasibility_status"] == "not_evaluated" and point["candidate"] is None
    assert first["place_data"]["status"] == "live"
    assert _nearby_request(registered_delta, radius_km=2).json()["place_data"]["status"] == "fresh_cache"
    assert len(calls) == 1
    clock[0] += online.FRESH_SECONDS + 1

    def unavailable(_):
        raise TimeoutError("offline")

    monkeypatch.setattr(online, "_fetch", unavailable)
    cached = _nearby_request(registered_delta, radius_km=2).json()
    assert cached["place_data"]["status"] == "stale_cache"
    assert any(p["name"] == "API-only nearby museum" for p in cached["poi_predictions"])
    assert session.public_dict() == before
    assert _nearby_request(registered_delta, refresh_places="yes").status_code == 422


def test_a_new_selection_during_the_online_lookup_cannot_return_a_mixed_revision(
    registered_delta, tmp_path, monkeypatch
):
    from itinerary_system.product_app import nearby_places as online

    _, _, service, session, _, _, _ = registered_delta
    monkeypatch.setattr(service, "nearby_places", online.NearbyPlaceCache(tmp_path / "online"))
    revision = session.revision

    def fetch(_):
        session.revision += 1
        return {"elements": []}

    monkeypatch.setattr(online, "_fetch", fetch)
    try:
        response = _nearby_request(registered_delta, radius_km=2)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "stale_session_revision"
    finally:
        session.revision = revision
