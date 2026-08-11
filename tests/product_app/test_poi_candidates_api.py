from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
TRIP_ID = "california_coast_demo"
INSERTION_LEG = "leg_stearns_wharf_surf_n_wear_s_beach_house"
REPLACEMENT_LEG = "leg_hotel_milo_santa_barbara_golden_gate_bridge"
MATRIX_ID = "route_matrix_68ab535465b06808"


def _query_hash(origin_id: str, destination_id: str) -> str:
    return hashlib.sha256(f"{origin_id}->{destination_id}".encode()).hexdigest()[:16]


def _catalog_record(
    *,
    place_id: str = "candidate_alpha",
    access_evidence_refs: tuple[str, str] | None = None,
) -> dict[str, Any]:
    refs = access_evidence_refs or (
        _query_hash("stearns_wharf", place_id),
        _query_hash(place_id, "surf_n_wear_s_beach_house"),
    )
    return {
        "schema_version": "product-poi-catalog-v1",
        "catalog_id": "california_coast_candidates_v1",
        "generated_at": "2026-08-08T12:00:00Z",
        "sources": [
            {
                "source_id": MATRIX_ID,
                "source_type": "provider",
                "source_url": None,
                "retrieved_at": "2026-08-08T11:00:00Z",
            }
        ],
        "places": [
            {
                "place_id": place_id,
                "name": "Candidate scenic stop",
                "place_categories": ["scenic_stop", "outdoor"],
                "display_coordinate": {"longitude": -119.70, "latitude": 34.42},
                "access_points": [
                    {
                        "access_point_id": f"{place_id}_entrance",
                        "access_type": "parking_entrance",
                        "coordinate": {"longitude": -119.701, "latitude": 34.421},
                        "source_ref": MATRIX_ID,
                        "road_validated": True,
                        "access_confidence": "verified_entrance",
                        "evidence_refs": list(refs),
                    }
                ],
                "description": "A source-backed candidate for route-aware inspection.",
                "official_url": "https://example.org/candidate",
                "informational_urls": [],
                "source_refs": [MATRIX_ID],
                "source_freshness": "2026-08-08T11:00:00Z",
                "opening_hours_evidence_ref": None,
                "recommended_visit_minutes": 30,
                "weather_suitability": None,
                "static_utility": None,
            }
        ],
    }


def _write_catalog(repository_root: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    catalog_root = repository_root / "configs" / "product_poi_catalogs" / TRIP_ID
    catalog_root.mkdir(parents=True)
    catalog_file = catalog_root / "catalog.json"
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    catalog_file.write_bytes(payload)
    manifest_file = catalog_root / "manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": "product-poi-catalog-manifest-v1",
                "catalog_file": "catalog.json",
                "catalog_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return manifest_file, catalog_file


def _cell(
    origin_id: str,
    destination_id: str,
    *,
    minutes: float,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "origin_id": origin_id,
        "destination_id": destination_id,
        "distance_m": minutes * 1_000,
        "duration_s": minutes * 60,
        "route_leg_id": f"leg_{origin_id}_{destination_id}",
        "road_validated": not fallback_used,
        "fallback_used": fallback_used,
        "provider": "cached_osrm",
        "query_hash": _query_hash(origin_id, destination_id),
        "context_snapshot_id": "context_static_demo_2026_06",
        "geometry": [[34.0, -119.0], [34.1, -119.1]],
        "routing_profile": "driving",
        "routing_status": "osrm_live" if not fallback_used else "fallback",
        "geometry_source": "cached_osrm_route_geometry",
        "distance_source": "cached_osrm_route_distance",
        "duration_source": "cached_osrm_route_duration",
        "fallback_reason": "test_fallback" if fallback_used else None,
    }


@pytest.fixture
def candidate_app(tmp_path: Path) -> tuple[TestClient, Any, dict[str, Any], dict[str, str], Path]:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    client = TestClient(app)
    with client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200
        payload = created.json()
    runtime = app.state.product_runtime
    service = runtime.require_service()
    # The run is already hash-validated and loaded.  Redirect only the catalog
    # convention to an isolated repository root for each adversarial test.
    service.registry.repository_root = tmp_path / "repository"
    session = payload["session"]
    headers = {"X-Session-Token": payload["mutation_token"]}
    return client, service, session, headers, service.registry.repository_root


def _append_candidate_cells(service: Any, cells: list[dict[str, Any]]) -> None:
    run_id = service.registry.default.run_id
    bundle = service._bundles[run_id]
    bundle.route_matrix["cells"].extend(deepcopy(cells))
    entity_ids = bundle.route_matrix["entity_ids"]
    for cell in cells:
        for entity_id in (cell["origin_id"], cell["destination_id"]):
            if entity_id not in entity_ids:
                entity_ids.append(entity_id)


def _request(
    client: TestClient,
    session: dict[str, Any],
    headers: dict[str, str] | None,
    **query: Any,
):
    parameters = {
        "expected_revision": 0,
        "day": 4,
        "route_leg_id": INSERTION_LEG,
        "role": "scenic_stop",
        "max_detour_minutes": 30,
        "limit": 5,
        **query,
    }
    return client.get(
        f"/api/sessions/{session['session_id']}/map/poi-candidates",
        headers=headers,
        params=parameters,
    )


def test_insertion_candidates_are_authenticated_revision_bound_and_read_only(
    candidate_app,
) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    _append_candidate_cells(
        service,
        [
            _cell("stearns_wharf", "candidate_alpha", minutes=5),
            _cell("candidate_alpha", "surf_n_wear_s_beach_house", minutes=6),
        ],
    )

    unauthorized = _request(client, session, None)
    stale = _request(client, session, headers, expected_revision=1)
    response = _request(client, session, headers)
    restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)

    assert unauthorized.status_code == 403
    assert stale.status_code == 409
    assert stale.json() == {"detail": "stale_session_revision"}
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "product-poi-candidates-v1"
    assert payload["session_id"] == session["session_id"]
    assert payload["session_revision"] == 0
    assert payload["context"]["kind"] == "insertion"
    assert payload["context"]["predecessor_id"] == "stearns_wharf"
    assert payload["context"]["successor_id"] == "surf_n_wear_s_beach_house"
    assert payload["context"]["baseline_route_leg_ids"] == [INSERTION_LEG]
    assert payload["catalog"]["catalog_id"] == "california_coast_candidates_v1"
    assert len(payload["catalog"]["catalog_sha256"]) == 64
    assert payload["routing"]["road_validated_only"] is True
    assert payload["routing"]["fallback_allowed"] is False
    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["place"]["place_id"] == "candidate_alpha"
    assert candidate["selected_access_point"]["road_validated"] is True
    assert candidate["burden"]["context_kind"] == "insertion"
    assert candidate["burden"]["marginal_travel_minutes"] == pytest.approx(8.475)
    assert candidate["burden"]["parking_minutes"] is None
    assert candidate["burden"]["walking_minutes"] is None
    assert candidate["burden"]["waiting_minutes"] is None
    assert candidate["burden"]["total_insertion_minutes"] is None
    assert candidate["burden"]["ranking_eligible"] is False
    assert candidate["burden"]["recommended"] is False
    assert candidate["precheck"] == {
        "predicted_arrival": None,
        "open_at_arrival": None,
        "status": "unavailable",
        "evidence_refs": [],
    }
    assert len(candidate["route_evidence_refs"]) == 3
    assert {row["role"] for row in candidate["route_evidence_refs"]} == {
        "predecessor_candidate",
        "candidate_successor",
        "predecessor_successor",
    }
    assert candidate["registered_replacement"] is None
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []


def test_replacement_uses_incoming_and_outgoing_target_baseline(candidate_app) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(
        repository_root,
        _catalog_record(
            access_evidence_refs=(
                _query_hash("hotel_milo_santa_barbara", "candidate_alpha"),
                _query_hash("candidate_alpha", "san_francisco_international_airport"),
            )
        ),
    )
    _append_candidate_cells(
        service,
        [
            _cell("hotel_milo_santa_barbara", "candidate_alpha", minutes=300),
            _cell("candidate_alpha", "san_francisco_international_airport", minutes=30),
        ],
    )

    response = _request(
        client,
        session,
        headers,
        day=7,
        route_leg_id=REPLACEMENT_LEG,
        replacement_target_id="golden_gate_bridge",
        max_detour_minutes=60,
    )

    assert response.status_code == 200
    payload = response.json()
    context = payload["context"]
    assert context["kind"] == "replacement"
    assert context["replacement_target_id"] == "golden_gate_bridge"
    assert context["predecessor_id"] == "hotel_milo_santa_barbara"
    assert context["successor_id"] == "san_francisco_international_airport"
    assert context["baseline_route_leg_ids"] == [
        REPLACEMENT_LEG,
        "leg_golden_gate_bridge_san_francisco_international_airport",
    ]
    burden = payload["candidates"][0]["burden"]
    assert burden["context_kind"] == "replacement"
    assert burden["predecessor_to_successor_minutes"] is None
    assert burden["marginal_travel_minutes"] == pytest.approx(
        330 - context["baseline_travel_minutes"]
    )
    assert len(payload["candidates"][0]["route_evidence_refs"]) == 4
    assert {
        row["role"] for row in payload["candidates"][0]["route_evidence_refs"]
    } == {
        "predecessor_candidate",
        "candidate_successor",
        "predecessor_target",
        "target_successor",
    }


@pytest.mark.parametrize(
    ("place_id", "longitude", "latitude", "access_refs"),
    [
        (
            "bixby_creek_bridge_viewpoint",
            -121.9018,
            36.3715,
            ("001da97c5482f649", "26e91ac90b1bd16b"),
        ),
        (
            "santa_barbara_museum_of_natural_history_sea_center",
            -119.6853711,
            34.4093803,
            ("03d34a8e6a8a89bb", "aa247cb3abc422f3"),
        ),
    ],
)
def test_every_exact_registered_replacement_is_evaluated_but_never_ranked(
    candidate_app,
    place_id: str,
    longitude: float,
    latitude: float,
    access_refs: tuple[str, str],
) -> None:
    client, _, session, headers, repository_root = candidate_app
    record = _catalog_record(
        place_id=place_id,
        access_evidence_refs=access_refs,
    )
    record["places"][0]["display_coordinate"] = {
        "longitude": longitude,
        "latitude": latitude,
    }
    _write_catalog(repository_root, record)

    response = _request(
        client,
        session,
        headers,
        day=7,
        route_leg_id=REPLACEMENT_LEG,
        replacement_target_id="golden_gate_bridge",
        max_detour_minutes=480,
    )

    assert response.status_code == 200
    candidate = response.json()["candidates"][0]
    assert candidate["burden"]["evaluated_feasible"] is True
    assert candidate["burden"]["candidate_state"] == "evaluated_feasible"
    assert candidate["burden"]["ranking_eligible"] is False
    assert candidate["burden"]["recommended"] is False
    mapping = candidate["registered_replacement"]
    assert mapping == {
        "draft_type": "replace_nearby",
        "target_stop_id": "golden_gate_bridge",
        "candidate_id": place_id,
    }
    assert any(ref.startswith("cert_") for ref in candidate["burden"]["evidence_refs"])
    assert candidate["candidate_id"] == place_id == candidate["place"]["place_id"]

    selected = client.post(
        f"/api/sessions/{session['session_id']}/selection",
        headers=headers,
        json={"expected_revision": 0, "selected_candidate_id": place_id},
    )
    unknown = client.post(
        f"/api/sessions/{session['session_id']}/selection",
        headers=headers,
        json={
            "expected_revision": 1,
            "selected_candidate_id": "candidate_deadbeefdeadbeef",
        },
    )
    assert selected.status_code == 200
    assert selected.json()["session"]["selected_candidate_id"] == place_id
    assert unknown.status_code == 422
    assert unknown.json() == {"detail": "invalid_selected_candidate"}


def test_default_detour_exposes_both_hash_pinned_registered_replacements(
    tmp_path: Path,
) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "real-catalog-state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        response = client.get(
            f"/api/sessions/{session['session_id']}/map/poi-candidates",
            headers=headers,
            params={
                "expected_revision": 0,
                "day": 7,
                "route_leg_id": REPLACEMENT_LEG,
                "replacement_target_id": "golden_gate_bridge",
                "limit": 10,
            },
        )

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert {row["candidate_id"] for row in candidates} == {
        "bixby_creek_bridge_viewpoint",
        "santa_barbara_museum_of_natural_history_sea_center",
    }
    assert all(row["burden"]["marginal_travel_minutes"] < 0 for row in candidates)
    assert all(row["burden"]["candidate_state"] == "evaluated_feasible" for row in candidates)


@pytest.mark.parametrize(
    ("query", "detail"),
    [
        ({"day": 3}, "selected_route_segment_day_mismatch"),
        ({"route_leg_id": "leg_unknown"}, "selected_route_segment_not_found"),
        ({"role": "invented_role"}, "poi_candidate_role_invalid"),
        ({"limit": 11}, "poi_candidate_limit_invalid"),
        ({"max_detour_minutes": 481}, "poi_candidate_max_detour_invalid"),
    ],
)
def test_query_and_route_context_fail_closed(candidate_app, query, detail) -> None:
    client, _, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())

    response = _request(client, session, headers, **query)

    assert response.status_code == 422
    assert response.json() == {"detail": detail}


def test_replacement_target_must_be_exact_interior_stop_and_match_selected_leg(
    candidate_app,
) -> None:
    client, _, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())

    missing = _request(
        client,
        session,
        headers,
        replacement_target_id="not_a_stop",
    )
    mismatch = _request(
        client,
        session,
        headers,
        day=7,
        route_leg_id="leg_golden_gate_bridge_san_francisco_international_airport",
        replacement_target_id="stearns_wharf",
    )

    assert missing.status_code == 422
    assert missing.json() == {"detail": "replacement_target_not_found"}
    assert mismatch.status_code == 422
    assert mismatch.json() == {"detail": "replacement_target_not_found"}


def test_missing_or_invalid_catalog_fails_explicitly(candidate_app) -> None:
    client, _, session, headers, repository_root = candidate_app
    missing = _request(client, session, headers)
    _, catalog_file = _write_catalog(repository_root, _catalog_record())
    catalog_file.write_bytes(catalog_file.read_bytes() + b" ")
    invalid_hash = _request(client, session, headers)

    assert missing.status_code == 409
    assert missing.json() == {"detail": "poi_catalog_unavailable"}
    assert invalid_hash.status_code == 409
    assert invalid_hash.json() == {"detail": "poi_catalog_hash_mismatch"}


@pytest.mark.parametrize("cell_kind", ["missing", "fallback"])
def test_missing_or_fallback_candidate_cell_is_never_exposed(
    candidate_app, cell_kind: str
) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    cells = [_cell("stearns_wharf", "candidate_alpha", minutes=5)]
    if cell_kind == "fallback":
        cells.append(
            _cell(
                "candidate_alpha",
                "surf_n_wear_s_beach_house",
                minutes=6,
                fallback_used=True,
            )
        )
    _append_candidate_cells(service, cells)

    response = _request(client, session, headers)

    assert response.status_code == 200
    assert response.json()["candidates"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_hash", ""),
        ("query_hash", "NOT-LOWER-HEX"),
        ("provider", "untrusted_router"),
        ("routing_profile", "walking"),
        ("routing_status", "unknown"),
        ("geometry_source", "straight_line"),
        ("distance_source", "geodesic"),
        ("duration_source", "speed_proxy"),
        ("context_snapshot_id", "wrong_context"),
        ("road_validated", False),
        ("fallback_reason", "foreign_fallback"),
        ("geometry", []),
        ("geometry", [[999.0, -119.0], [34.1, -119.1]]),
        ("distance_m", float("inf")),
        ("duration_s", 0),
    ],
)
def test_candidate_cell_provenance_must_match_frozen_osrm_contract(
    candidate_app, field: str, value: Any
) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    second = _cell(
        "candidate_alpha", "surf_n_wear_s_beach_house", minutes=6
    )
    second[field] = value
    _append_candidate_cells(
        service,
        [
            _cell("stearns_wharf", "candidate_alpha", minutes=5),
            second,
        ],
    )

    response = _request(client, session, headers)

    assert response.status_code == 200
    assert response.json()["candidates"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "forged-route-matrix-v999"),
        ("matrix_id", ""),
        ("matrix_id", "route_matrix_not_hex"),
        ("context_snapshot_id", ""),
        ("source_bundle_id", ""),
        ("source_bundle_id", "route_bundle_not_hex"),
        ("source_content_sha256", ""),
        ("source_content_sha256", "A" * 64),
        ("entity_ids", "not-a-list"),
        ("entity_ids", ["duplicate", "duplicate"]),
        ("entity_ids", [1]),
    ],
)
def test_matrix_header_provenance_is_mandatory(
    candidate_app, field: str, value: Any
) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    bundle = service._bundles[service.registry.default.run_id]
    bundle.route_matrix[field] = value

    response = _request(client, session, headers)

    assert response.status_code == 409
    assert response.json() == {"detail": "poi_route_matrix_invalid"}


def test_unknown_matrix_top_level_field_is_rejected(candidate_app) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    bundle = service._bundles[service.registry.default.run_id]
    bundle.route_matrix["unexpected_top_level"] = "ignored_before_remediation"

    response = _request(client, session, headers)

    assert response.status_code == 409
    assert response.json() == {"detail": "poi_route_matrix_invalid"}


def test_malformed_accepted_baseline_cell_fails_the_endpoint(candidate_app) -> None:
    client, service, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    bundle = service._bundles[service.registry.default.run_id]
    baseline = next(
        cell
        for cell in bundle.route_matrix["cells"]
        if cell["route_leg_id"] == INSERTION_LEG
    )
    baseline["provider"] = "foreign_router"

    response = _request(client, session, headers)

    assert response.status_code == 409
    assert response.json() == {"detail": "accepted_route_baseline_unavailable"}


@pytest.mark.parametrize("mismatch", ["source", "evidence"])
def test_catalog_access_must_bind_to_exact_matrix_candidate_cells(
    candidate_app, mismatch: str
) -> None:
    client, service, session, headers, repository_root = candidate_app
    record = _catalog_record()
    access = record["places"][0]["access_points"][0]
    if mismatch == "source":
        record["sources"].append(
            {
                "source_id": "foreign_route_source",
                "source_type": "provider",
                "source_url": None,
                "retrieved_at": "2026-08-08T11:00:00Z",
            }
        )
        access["source_ref"] = "foreign_route_source"
    else:
        access["evidence_refs"] = ["0000000000000000", "1111111111111111"]
    _write_catalog(repository_root, record)
    _append_candidate_cells(
        service,
        [
            _cell("stearns_wharf", "candidate_alpha", minutes=5),
            _cell("candidate_alpha", "surf_n_wear_s_beach_house", minutes=6),
        ],
    )

    response = _request(client, session, headers)

    assert response.status_code == 200
    assert response.json()["candidates"] == []


def test_unknown_duplicate_and_missing_query_fields_have_stable_errors(
    candidate_app,
) -> None:
    client, _, session, headers, repository_root = candidate_app
    _write_catalog(repository_root, _catalog_record())
    path = f"/api/sessions/{session['session_id']}/map/poi-candidates"

    unknown = client.get(
        path,
        headers=headers,
        params={
            "expected_revision": 0,
            "day": 4,
            "route_leg_id": INSERTION_LEG,
            "unknown": "x",
        },
    )
    duplicate = client.get(
        path,
        headers=headers,
        params=[
            ("expected_revision", "0"),
            ("expected_revision", "0"),
            ("day", "4"),
            ("route_leg_id", INSERTION_LEG),
        ],
    )
    missing = client.get(
        path,
        headers=headers,
        params={"expected_revision": 0, "day": 4},
    )

    assert unknown.json() == {"detail": "poi_candidate_query_field_unknown"}
    assert duplicate.json() == {"detail": "poi_candidate_query_field_duplicate"}
    assert missing.json() == {"detail": "poi_candidate_query_field_missing"}
