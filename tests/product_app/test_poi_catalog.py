from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from itinerary_system.product_app.poi_catalog import (
    CATALOG_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    POICatalogError,
    ProductPOICatalogV1,
)


def _catalog_record() -> dict[str, object]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_id": "california_coast_pois_v1",
        "generated_at": "2026-08-07T12:00:00Z",
        "sources": [
            {
                "source_id": "official_source",
                "source_type": "official",
                "source_url": "https://parks.example/place",
                "retrieved_at": "2026-08-06T10:00:00Z",
            },
            {
                "source_id": "info_source",
                "source_type": "informational",
                "source_url": "https://guide.example/place",
                "retrieved_at": None,
            },
        ],
        "places": [
            {
                "place_id": "bixby_bridge",
                "name": "Bixby Creek Bridge Viewpoint",
                "place_categories": ["scenic_stop", "outdoor"],
                "display_coordinate": {"longitude": -121.901, "latitude": 36.371},
                "access_points": [
                    {
                        "access_point_id": "bixby_parking",
                        "access_type": "parking",
                        "coordinate": {"longitude": -121.9007, "latitude": 36.3708},
                        "source_ref": "official_source",
                        "road_validated": True,
                        "access_confidence": "verified_entrance",
                        "evidence_refs": ["route_evidence_bixby"],
                    }
                ],
                "description": "A sourced scenic viewpoint.",
                "official_url": "https://parks.example/place",
                "informational_urls": ["https://guide.example/place"],
                "source_refs": ["official_source", "info_source"],
                "source_freshness": "2026-08-06T10:00:00Z",
                "opening_hours_evidence_ref": "hours_bixby",
                "recommended_visit_minutes": 30,
            },
            {
                "place_id": "museum_alpha",
                "name": "Alpha Museum",
                "place_categories": ["attraction", "indoor"],
                "display_coordinate": {"longitude": -118.25, "latitude": 34.05},
                "access_points": [
                    {
                        "access_point_id": "museum_raw_snap",
                        "access_type": "road_snap",
                        "coordinate": {"longitude": -118.249, "latitude": 34.051},
                        "source_ref": "info_source",
                        "road_validated": False,
                        "access_confidence": "uncertain",
                        "evidence_refs": [],
                    }
                ],
                "official_url": None,
                "informational_urls": ["https://guide.example/museum"],
                "source_refs": ["info_source"],
            },
            {
                "place_id": "alpha_garden",
                "name": "Alpha Garden",
                "place_categories": ["attraction", "outdoor"],
                "display_coordinate": {"longitude": -122.42, "latitude": 37.77},
                "access_points": [],
                "source_refs": ["info_source"],
            },
        ],
    }


def _write_fixture(tmp_path: Path, catalog: dict[str, object] | None = None) -> Path:
    catalog_file = tmp_path / "catalog.json"
    payload = json.dumps(catalog or _catalog_record(), sort_keys=True).encode("utf-8")
    catalog_file.write_bytes(payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "catalog_file": "catalog.json",
        "catalog_sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_file


def _assert_code(exc_info: pytest.ExceptionInfo[POICatalogError], code: str) -> None:
    assert exc_info.value.code == code
    assert str(exc_info.value) == code


def test_loads_hash_pinned_immutable_catalog_and_separates_access_points(
    tmp_path: Path,
) -> None:
    catalog = ProductPOICatalogV1.load(_write_fixture(tmp_path))

    place = catalog.get_place("bixby_bridge")
    assert place is not None
    assert place.display_coordinate.longitude == -121.901
    assert place.access_points[0].coordinate.longitude == -121.9007
    assert place.has_road_validated_access is True
    assert place.official_url == "https://parks.example/place"
    assert place.informational_urls == ("https://guide.example/place",)
    assert len(catalog.manifest_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        place.name = "Mutated"  # type: ignore[misc]


def test_omitted_evidence_is_nullable_and_never_coerced_to_zero(tmp_path: Path) -> None:
    catalog = ProductPOICatalogV1.load(_write_fixture(tmp_path))

    place = catalog.get_place("museum_alpha")
    assert place is not None
    assert place.description is None
    assert place.source_freshness is None
    assert place.opening_hours_evidence_ref is None
    assert place.recommended_visit_minutes is None
    assert place.weather_suitability is None
    assert place.static_utility is None


def test_rejects_catalog_hash_mismatch(tmp_path: Path) -> None:
    manifest_file = _write_fixture(tmp_path)
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_bytes(catalog_file.read_bytes() + b" ")

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(manifest_file)
    _assert_code(exc_info, "catalog_hash_mismatch")


@pytest.mark.parametrize("catalog_file", ["../catalog.json", "C:/catalog.json"])
def test_rejects_noncontained_catalog_path(tmp_path: Path, catalog_file: str) -> None:
    manifest_file = _write_fixture(tmp_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["catalog_file"] = catalog_file
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(manifest_file)
    _assert_code(exc_info, "catalog_path_not_relative")


@pytest.mark.parametrize(
    ("target", "field", "code"),
    [
        ("catalog", "unexpected", "catalog_field_unknown"),
        ("place", "itinerary_role", "place_field_unknown"),
        ("access", "place_name", "access_point_field_unknown"),
    ],
)
def test_rejects_unknown_fields_at_every_catalog_boundary(
    tmp_path: Path, target: str, field: str, code: str
) -> None:
    record = _catalog_record()
    if target == "catalog":
        record[field] = True
    elif target == "place":
        record["places"][0][field] = True  # type: ignore[index]
    else:
        record["places"][0]["access_points"][0][field] = True  # type: ignore[index]

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, code)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    catalog_payload = (
        b'{"schema_version":"product-poi-catalog-v1",'
        b'"schema_version":"product-poi-catalog-v1",'
        b'"catalog_id":"x","generated_at":"2026-08-07T12:00:00Z",'
        b'"sources":[],"places":[]}'
    )
    (tmp_path / "catalog.json").write_bytes(catalog_payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "catalog_file": "catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_payload).hexdigest(),
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(manifest_file)
    _assert_code(exc_info, "json_duplicate_key")


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("place", "place_id_duplicate"),
        ("access", "access_point_id_duplicate"),
        ("source", "source_id_duplicate"),
    ],
)
def test_rejects_duplicate_entity_ids(tmp_path: Path, mutation: str, code: str) -> None:
    record = _catalog_record()
    if mutation == "place":
        record["places"][1]["place_id"] = "bixby_bridge"  # type: ignore[index]
    elif mutation == "access":
        record["places"][1]["access_points"][0]["access_point_id"] = (  # type: ignore[index]
            "bixby_parking"
        )
    else:
        record["sources"][1]["source_id"] = "official_source"  # type: ignore[index]

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, code)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///C:/secret",
        "https://user:password@example.com/place",
        "https://example.com/place\nHeader: injected",
    ],
)
def test_rejects_unsafe_official_urls(tmp_path: Path, url: str) -> None:
    record = _catalog_record()
    record["places"][0]["official_url"] = url  # type: ignore[index]

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, "official_url_invalid")


def test_rejects_same_link_as_official_and_informational(tmp_path: Path) -> None:
    record = _catalog_record()
    record["places"][0]["informational_urls"] = [  # type: ignore[index]
        "https://parks.example/place"
    ]

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, "place_url_role_duplicate")


def test_road_validated_access_requires_evidence_and_confident_access(
    tmp_path: Path,
) -> None:
    record = _catalog_record()
    access = record["places"][0]["access_points"][0]  # type: ignore[index]
    access["evidence_refs"] = []

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, "road_validated_access_missing_evidence")


def test_rejects_unknown_place_and_access_source_references(tmp_path: Path) -> None:
    record = _catalog_record()
    record["places"][0]["source_refs"] = ["missing_source"]  # type: ignore[index]

    with pytest.raises(POICatalogError) as exc_info:
        ProductPOICatalogV1.load(_write_fixture(tmp_path, record))
    _assert_code(exc_info, "place_source_ref_unknown")


def test_search_is_deterministic_bounded_and_not_a_recommendation(tmp_path: Path) -> None:
    catalog = ProductPOICatalogV1.load(_write_fixture(tmp_path))

    assert [place.place_id for place in catalog.search("alpha")] == [
        "alpha_garden",
        "museum_alpha",
    ]
    assert [place.place_id for place in catalog.search(categories=["outdoor"])] == [
        "alpha_garden",
        "bixby_bridge",
    ]
    assert [
        place.place_id
        for place in catalog.search(source_types=["official"], require_road_access=True)
    ] == ["bixby_bridge"]
    assert [
        place.place_id
        for place in catalog.search(bounds=(-119.0, 33.0, -118.0, 35.0))
    ] == ["museum_alpha"]
    assert len(catalog.search(limit=1)) == 1


@pytest.mark.parametrize("limit", [0, 51, True])
def test_search_rejects_unbounded_or_boolean_limits(tmp_path: Path, limit: int) -> None:
    catalog = ProductPOICatalogV1.load(_write_fixture(tmp_path))

    with pytest.raises(POICatalogError) as exc_info:
        catalog.search(limit=limit)
    _assert_code(exc_info, "search_limit_invalid")
