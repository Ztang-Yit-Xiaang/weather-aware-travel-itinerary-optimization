from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.interest_regions import _exploration_places, required_interest_points

ROOT = Path(__file__).resolve().parents[2]
RUN = "california_coast_product_demo_v2"
LEG = "leg_stearns_wharf_surf_n_wear_s_beach_house"


def test_big_sur_exploration_keeps_only_local_sourced_entries_without_readding_the_current_park():
    parent = {
        "plan_id": "test",
        "selected_stops": [
            {
                "attraction_name": "Pfeiffer Big Sur State Park",
                "latitude": 36.2503,
                "longitude": -121.7831,
            }
        ],
    }
    geography = {
        "plans": [
            {
                "plan_id": "test",
                "validated_legs": {
                    "features": [
                        {
                            "properties": {"route_leg_id": "test_leg"},
                            "geometry": {"coordinates": [[-121.7831, 36.2503], [-121.8947, 36.6002]]},
                        }
                    ]
                },
            }
        ]
    }
    search = {"coordinate": {"latitude": 36.2503, "longitude": -121.7831}, "radius_km": 10}
    places, manifest = _exploration_places(
        ROOT, {"context": {"route_leg_id": "test_leg"}}, parent, geography, None, search
    )
    names = {point["name"] for point in places}
    assert "Andrew Molera State Park" in names
    assert not {"Garrapata State Park", "Point Lobos State Natural Reserve"} & names
    assert "Pfeiffer Big Sur State Park" not in names
    snapshot = json.loads((ROOT / "configs/interest_regions/monterey_park_entries.geojson").read_text())
    originals = {"csp_entry_" + feature["properties"]["GISID"].lower(): feature for feature in snapshot["features"]}
    for point in places:
        if point["place_id"] in originals:
            original = originals[point["place_id"]]
            coordinate = point["display_coordinate"]
            assert [coordinate["longitude"], coordinate["latitude"]] == original["geometry"]["coordinates"]
            assert original["properties"]["PARK_WEB_PG"] in point["source_refs"]
    assert manifest["supplemental_source"]["attribution"] == "California State Parks"


@pytest.mark.parametrize(
    "geometry,valid",
    [
        ([[34, -119], [34.1, -119.1]], True),
        (((-90, -180), (90, 180)), True),
        ([[True, -119], [34, -119]], False),
        ([[34, False], [34, -119]], False),
        ([["34", -119], [34, -119]], False),
        ([[float("nan"), -119], [34, -119]], False),
        ([[34, float("inf")], [34, -119]], False),
        ([[91, -119], [34, -119]], False),
        ([[34, -181], [34, -119]], False),
        ([[34, -119, 3], [34, -119]], False),
        ([[34, -119]], False),
        ({"coordinates": [[34, -119], [34, -119]]}, False),
    ],
)
def test_fast_route_geometry_validation_keeps_exact_type_shape_and_range_rules(geometry, valid):
    from itinerary_system.product_app.candidate_discovery import _valid_route_geometry

    assert _valid_route_geometry(geometry) is valid


@pytest.fixture(scope="module")
def interest_app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("interest-api")
    registry = json.loads((ROOT / "configs/product_app_registry.json").read_text())
    registry["runs"] = [row for row in registry["runs"] if row["run_id"] == RUN]
    registry["runs"][0]["default"] = True
    path = tmp / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    app = create_product_app(
        repository_root=ROOT,
        registry_path=path,
        state_root=tmp / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        response = client.post("/api/sessions", json={"run_id": RUN})
        assert response.status_code == 200, response.text
        created = response.json()
        service = app.state.product_runtime.require_service()
        yield client, service, created, tmp


def request(app, **overrides):
    client, _, created, _ = app
    query = {"expected_revision": 0, "day": 4, "route_leg_id": LEG, **overrides}
    return client.get(
        f"/api/sessions/{created['session']['session_id']}/map/interest-regions",
        params=query,
        headers={"X-Session-Token": created["mutation_token"]},
    )


def test_real_catalog_prior_has_complete_universe_masked_regions_and_zero_mutation(interest_app):
    _, service, created, _ = interest_app
    session = service.workspace.authenticate(created["session"]["session_id"], created["mutation_token"])
    before = deepcopy(session.public_dict())
    response = request(interest_app)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["candidate_universe"]["count"] == len(data["poi_predictions"]) > 10
    assert data["candidate_universe"]["truncated"] is False
    assert data["model"]["evidence_mode"] == "prior_only"
    assert data["model"]["feedback_count"] == 0
    assert data["model"]["calibration_status"] == "not_calibrated"
    assert data["surface"]["status"] == "ready"
    assert abs(data["surface"]["integrated_mass"] - 1) < 1e-10
    assert len(data["surface"]["regions_geojson"]["features"]) == 3
    assert any(p["feasibility_status"] == "not_evaluated" and p["candidate"] is None for p in data["poi_predictions"])
    assert session.public_dict() == before
    json.dumps(data, allow_nan=False)


def test_preferences_change_probabilities_and_mass_levels_do_not_require_new_probabilities(interest_app):
    nature = request(interest_app, nature_preference=1).json()
    culture = request(interest_app, nature_preference=-1).json()
    pairs = zip(nature["poi_predictions"], culture["poi_predictions"], strict=True)
    assert any(abs(a["interest_probability_mean"] - b["interest_probability_mean"]) > 0.2 for a, b in pairs)
    assert nature["model"]["model_sha256"] == culture["model"]["model_sha256"]
    assert {f["properties"]["mass_level"] for f in nature["surface"]["regions_geojson"]["features"]} == {0.5, 0.8, 0.95}


@pytest.mark.parametrize(
    "query,expected",
    [
        ({"expected_revision": 999}, 409),
        ({"nature_preference": "nan"}, 422),
        ({"nature_preference": 1.1}, 422),
        ({"limit": 10}, 422),
        ({"day": 999}, 422),
    ],
)
def test_stale_or_invalid_queries_fail_explicitly(interest_app, query, expected):
    assert request(interest_app, **query).status_code == expected


def test_authentication_is_required(interest_app):
    client, _, created, _ = interest_app
    result = client.get(
        f"/api/sessions/{created['session']['session_id']}/map/interest-regions",
        params={"expected_revision": 0, "day": 4, "route_leg_id": LEG},
    )
    assert result.status_code in {401, 403}


@pytest.mark.parametrize("replacement", [None, "surf_n_wear_s_beach_house"])
def test_focused_route_validation_matches_full_matrix_validation(interest_app, monkeypatch, replacement):
    import itinerary_system.product_app.candidate_discovery as module

    focused = request(interest_app, **({"replacement_target_id": replacement} if replacement else {}))
    original = module._route_matrix
    monkeypatch.setattr(module, "_route_matrix", lambda record, **_: original(record))
    full = request(interest_app, **({"replacement_target_id": replacement} if replacement else {}))
    assert focused.status_code == full.status_code == 200
    assert focused.json() == full.json()


@pytest.mark.parametrize("corruption", ["selected_geometry", "unrelated_duplicate"])
def test_focused_routes_keep_exact_geometry_and_global_identity_guards(interest_app, corruption):
    _, service, _, _ = interest_app
    record = service._bundles[RUN].route_matrix
    original = deepcopy(record)
    try:
        if corruption == "selected_geometry":
            cell = next(cell for cell in record["cells"] if cell["route_leg_id"] == LEG)
            cell["geometry"][1][0] = True
        else:
            unrelated = next(cell for cell in record["cells"] if cell["origin_id"] == "golden_gate_bridge")
            record["cells"].append(deepcopy(unrelated))
        assert request(interest_app).status_code == 409
    finally:
        record.clear()
        record.update(original)


def test_support_hash_failure_hides_only_region_and_keeps_predictions(interest_app, monkeypatch):
    import itinerary_system.product_app.interest_regions as module

    def invalid(_):
        raise ValueError("interest_support_hash_mismatch")

    monkeypatch.setattr(module, "load_land_support", invalid)
    response = request(interest_app)
    assert response.status_code == 200
    assert response.json()["poi_predictions"]
    assert response.json()["surface"]["unavailable_reason"] == "interest_support_hash_mismatch"


def test_required_mapper_uses_confirmed_existence_and_preserves_missing_conflicts():
    def constraint(id_, target, relation="must_keep", value=True, confirmed=True):
        return {
            "constraint_id": id_,
            "origin": "user",
            "strength": "hard",
            "scope": "stop",
            "target_id": target,
            "relation": relation,
            "value": value,
            "confirmed": confirmed,
            "relaxation_policy": "explicit_only",
            "evidence_refs": [],
            "schema_version": "owned-constraint-v1",
        }

    parent = {
        "plan_id": "parent",
        "owned_constraints": [
            constraint("keep", "present"),
            constraint("missing", "absent"),
            constraint(
                "duration",
                "duration_only",
                "duration",
                {"mode": "exact", "preferred_minutes": 30, "minimum_minutes": 30, "maximum_minutes": 30},
            ),
            constraint("unconfirmed", "unconfirmed", confirmed=False),
        ],
    }
    geography = {
        "plans": [
            {
                "plan_id": "parent",
                "stops": {
                    "features": [
                        {
                            "properties": {"stop_id": "present", "day": 1, "name": "Present"},
                            "geometry": {"coordinates": [-119.7, 34.4]},
                        }
                    ]
                },
            }
        ]
    }
    points = required_interest_points(parent, geography, 1)
    assert {p["place_id"] for p in points} == {"present", "absent"}
    assert next(p for p in points if p["place_id"] == "absent")["conflict_status"] == "required_stop_missing"
    assert next(p for p in points if p["place_id"] == "absent")["coordinate"] is None
    next(p for p in points if p["place_id"] == "present")["coordinate"][0] = 0
    assert geography["plans"][0]["stops"]["features"][0]["geometry"]["coordinates"][0] == -119.7


def test_route_checked_universe_is_not_truncated_to_fifty_catalog_matches(interest_app, monkeypatch):
    from test_poi_candidates_api import _append_candidate_cells, _catalog_record, _cell, _write_catalog

    client, service, created, tmp = interest_app
    isolated = tmp / "full-universe-repository"
    records = _catalog_record()
    records["places"] = []
    cells = []
    for i in range(60):
        place_id = f"candidate_{i:03d}"
        place = _catalog_record(place_id=place_id)["places"][0]
        place["name"] = f"Synthetic candidate {i:03d}"
        records["places"].append(place)
        cells.extend(
            [_cell("stearns_wharf", place_id, minutes=10), _cell(place_id, "surf_n_wear_s_beach_house", minutes=10)]
        )
    _write_catalog(isolated, records)
    shutil.copytree(ROOT / "configs/interest_regions", isolated / "configs/interest_regions")
    original_matrix = deepcopy(service._bundles[RUN].route_matrix)
    monkeypatch.setattr(service.registry, "repository_root", isolated)
    try:
        _append_candidate_cells(service, cells)
        response = request(interest_app)
        assert response.status_code == 200, response.text
        assert response.json()["candidate_universe"]["route_checked_candidate_count"] == 60
        assert len(response.json()["poi_predictions"]) > 60
        assert response.json()["candidate_universe"]["truncated"] is False
    finally:
        service._bundles[RUN].route_matrix.clear()
        service._bundles[RUN].route_matrix.update(original_matrix)


def test_complete_500_place_endpoint_over_loopback_http_is_read_only_and_concurrent(interest_app, monkeypatch):
    """Synthetic volume data; actual auth, discovery, inference, clipping and HTTP serialization."""
    client, service, created, tmp = interest_app
    baseline = request(interest_app).json()
    isolated = tmp / "load-500-repository"
    shutil.copytree(ROOT / "configs/interest_regions", isolated / "configs/interest_regions")
    shutil.copytree(ROOT / "configs/product_poi_catalogs", isolated / "configs/product_poi_catalogs")
    rows = [
        {
            "place_id": point["place_id"],
            "name": point["name"],
            "display_coordinate": point["display_coordinate"],
            "place_categories": point["place_categories"],
            "source_refs": point["source_refs"],
        }
        for point in baseline["poi_predictions"]
    ]
    rng = np.random.default_rng(20261021)
    focus = baseline["search_context"]["coordinate"]
    for index in range(500 - len(rows)):
        rows.append(
            {
                "place_id": f"load_only_{index:04d}",
                "name": f"Synthetic load fixture {index:04d}",
                "display_coordinate": {
                    "longitude": float(rng.uniform(focus["longitude"] - 0.01, focus["longitude"] + 0.01)),
                    "latitude": float(rng.uniform(focus["latitude"] - 0.01, focus["latitude"] + 0.01)),
                },
                "place_categories": ["nature", "outdoor"],
                "source_refs": ["synthetic_load_test_not_a_real_place"],
            }
        )
    directory = isolated / "configs/interest_regions"
    payload = json.dumps({"schema_version": "interest-exploration-catalog-v1", "places": rows}).encode()
    (directory / "exploration_catalog.json").write_bytes(payload)
    import hashlib

    manifest = {
        "schema_version": "interest-exploration-manifest-v1",
        "file": "exploration_catalog.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "count": 500,
        "data_role": "synthetic_load_test_not_real_POIs",
    }
    (directory / "exploration_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(service.registry, "repository_root", isolated)
    session = service.workspace.authenticate(created["session"]["session_id"], created["mutation_token"])
    before = deepcopy(session.public_dict())
    from itinerary_system.product_app import interest_regions as interest_module

    timings = {name: [] for name in ["backend", "surface", "catalog_scope", "route_discovery", "model_predict"]}

    def measured(name, function):
        def execute(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[name].append((time.perf_counter() - started) * 1000)

        return execute

    monkeypatch.setattr(service, "interest_regions", measured("backend", service.interest_regions))
    monkeypatch.setattr(
        interest_module, "build_interest_surface", measured("surface", interest_module.build_interest_surface)
    )
    monkeypatch.setattr(
        interest_module, "_exploration_places", measured("catalog_scope", interest_module._exploration_places)
    )
    monkeypatch.setattr(
        interest_module, "discover_poi_candidates", measured("route_discovery", interest_module.discover_poi_candidates)
    )
    monkeypatch.setattr(
        interest_module.BayesianInterestModel,
        "predict",
        measured("model_predict", interest_module.BayesianInterestModel.predict),
    )
    response_sizes = []
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(client.app, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = (
        f"http://127.0.0.1:{port}/api/sessions/{session.session_id}/map/interest-regions"
        f"?expected_revision={session.revision}&day=4&route_leg_id={LEG}"
    )

    def query():
        started = time.perf_counter()
        # testserver is the fixture's explicitly allowed authority; no mutation request is made.
        http_request = urllib.request.Request(
            url, headers={"Host": "testserver", "X-Session-Token": created["mutation_token"]}
        )
        with urllib.request.urlopen(http_request, timeout=30) as response:
            body = response.read()
        response_sizes.append(len(body))
        data = json.loads(body)
        elapsed = (time.perf_counter() - started) * 1000
        assert data["candidate_universe"]["count"] == len(data["poi_predictions"]) == 500
        assert data["candidate_universe"]["truncated"] is False
        assert data["surface"]["algorithm_id"] == "masked-posterior-mixture-partial-cells-v2"
        assert data["surface"]["grid_size"] == 128
        assert abs(data["surface"]["integrated_mass"] - 1) < 1e-3
        return elapsed, data["candidate_universe"]["sha256"]

    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline and thread.is_alive():
            time.sleep(0.01)
        assert server.started
        serial = [query() for _ in range(16)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            concurrent = list(pool.map(lambda _: query(), range(6)))
        assert len({digest for _, digest in serial + concurrent}) == 1
        assert session.public_dict() == before
        receipt = {
            "evidence_mode": "synthetic_500_place_neighborhood_real_loopback_HTTP",
            "count": 500,
            "grid_size": 128,
            "cold_ms": serial[0][0],
            "warm_samples": 15,
            "warm_p95_ms": float(np.percentile([elapsed for elapsed, _ in serial[1:]], 95)),
            "warm_median_ms": float(np.median([elapsed for elapsed, _ in serial[1:]])),
            "concurrent_workers": 3,
            "concurrent_requests": 6,
            "all_succeeded": True,
            "session_unchanged": True,
            "real_user_feedback_count": 0,
            "response_bytes": max(response_sizes),
            "component_warm_median_ms": {name: float(np.median(values[1:16])) for name, values in timings.items()},
        }
        evidence = Path(os.environ.get("INTEREST_EVIDENCE_ROOT", str(tmp)))
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "http_500_benchmark.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        assert not thread.is_alive()
