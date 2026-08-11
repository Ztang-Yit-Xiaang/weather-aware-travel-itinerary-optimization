from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from itinerary_system.product_app.poi_catalog import ProductPOICatalogV1
from scripts.build_product_poi_catalog import (
    CATALOG_FILENAME,
    MANIFEST_FILENAME,
    CatalogBuildError,
    build_catalog_documents,
    check_catalog,
    write_catalog,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPOSITORY_ROOT / "configs/product_poi_catalogs/california_coast_demo"
)
EXPECTED_CATALOG_SHA256 = (
    "780aef9184b4f1186d3bfdd9174d2a458864c7a26f5600d553495f4e697967b6"
)

SOURCE_HASHES = {
    "runs/california-coast-product-demo-v2/manifest.json": (
        "925eea6e5722a782d48f657efa931e18536e502a7803a839a983747ed79b5e40"
    ),
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_recommended_v2/requests/w2_weather_recommended_v2.json"
    ): "26c760edc833913879c012e97f2f4ddccf49115cec43a98b692a02bb7f3f9ce6",
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_recommended_v2/plans/plan_f5ee52459659dcb5.json"
    ): "1e4528a293280db9c601e237dbce51021ae663e328e9cbe842c80b2a0ff3ebba",
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_recommended_v2/evaluations/cert_686ef65d376b2867.json"
    ): "7357273268fe65a13f10d3d72a5dd04ee4732aabe08a3560da4d61d76a6add5c",
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_low_driving_v2/requests/w2_weather_low_driving_v2.json"
    ): "675c406eb81c4ec548390932d18976c585e979cac9aa0c78da71e4235c53cf7b",
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_low_driving_v2/plans/plan_8aa919c8323dbac0.json"
    ): "bfa85e93d0d187f6a26d8ae8a27a1689cae852c8bc9ca186985cea53b715732f",
    (
        "runs/california-coast-product-demo-v2/alternatives/"
        "w2_weather_low_driving_v2/evaluations/cert_5a6deef4c159d346.json"
    ): "4bb3d2c2888fd667bcedf2711acd740aa3d2f4f43d662c3f6217d09a91e785df",
    "runs/e3ux-weather-repair-demo-v6/routing/route_matrix_68ab535465b06808.json": (
        "6f3b5b8eda1a9f73e28b6404ef15126086dbd7861c683790af6f6e87b41b4677"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_catalog() -> dict[str, object]:
    value = json.loads((FIXTURE_ROOT / CATALOG_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_catalog_and_manifest_regenerate_byte_for_byte() -> None:
    catalog_bytes, manifest_bytes = build_catalog_documents(
        repository_root=REPOSITORY_ROOT
    )

    assert catalog_bytes == (FIXTURE_ROOT / CATALOG_FILENAME).read_bytes()
    assert manifest_bytes == (FIXTURE_ROOT / MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest == {
        "catalog_file": "catalog.json",
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "schema_version": "product-poi-catalog-manifest-v1",
    }
    assert hashlib.sha256(catalog_bytes).hexdigest() == EXPECTED_CATALOG_SHA256
    check_catalog(repository_root=REPOSITORY_ROOT)


def test_fixture_has_only_the_two_exact_registered_candidate_places() -> None:
    catalog = ProductPOICatalogV1.load(FIXTURE_ROOT / MANIFEST_FILENAME)

    assert catalog.manifest_sha256 == EXPECTED_CATALOG_SHA256
    assert [place.place_id for place in catalog.places] == [
        "bixby_creek_bridge_viewpoint",
        "santa_barbara_museum_of_natural_history_sea_center",
    ]
    assert [place.name for place in catalog.places] == [
        "Bixby Creek Bridge Viewpoint",
        "Santa Barbara Museum of Natural History Sea Center",
    ]
    assert all(place.place_categories == ("registered_candidate",) for place in catalog.places)


@pytest.mark.parametrize(
    (
        "place_id",
        "display_longitude",
        "display_latitude",
        "access_longitude",
        "access_latitude",
        "evidence_refs",
    ),
    [
        (
            "bixby_creek_bridge_viewpoint",
            -121.9018,
            36.3715,
            -121.901894,
            36.371479,
            ("001da97c5482f649", "26e91ac90b1bd16b"),
        ),
        (
            "santa_barbara_museum_of_natural_history_sea_center",
            -119.6853711,
            34.4093803,
            -119.685364,
            34.409383,
            ("03d34a8e6a8a89bb", "aa247cb3abc422f3"),
        ),
    ],
)
def test_display_and_validated_route_access_are_separate_and_exact(
    place_id: str,
    display_longitude: float,
    display_latitude: float,
    access_longitude: float,
    access_latitude: float,
    evidence_refs: tuple[str, str],
) -> None:
    catalog = ProductPOICatalogV1.load(FIXTURE_ROOT / MANIFEST_FILENAME)
    place = catalog.get_place(place_id)

    assert place is not None
    assert (
        place.display_coordinate.longitude,
        place.display_coordinate.latitude,
    ) == (display_longitude, display_latitude)
    assert len(place.access_points) == 1
    access = place.access_points[0]
    assert (access.coordinate.longitude, access.coordinate.latitude) == (
        access_longitude,
        access_latitude,
    )
    assert access.coordinate != place.display_coordinate
    assert access.access_type == "road_route_endpoint"
    assert access.road_validated is True
    assert access.access_confidence == "provider_access_point"
    assert access.source_ref == "route_matrix_68ab535465b06808"
    assert access.evidence_refs == evidence_refs


def test_sources_keep_artifact_roles_and_do_not_invent_external_metadata() -> None:
    catalog = ProductPOICatalogV1.load(FIXTURE_ROOT / MANIFEST_FILENAME)
    sources = {source.source_id: source for source in catalog.sources}

    assert set(sources) == {
        "bixby_candidate_request_w2",
        "registered_certificate_5a6deef4c159d346",
        "registered_certificate_686ef65d376b2867",
        "registered_child_plan_8aa919c8323dbac0",
        "registered_child_plan_f5ee52459659dcb5",
        "route_matrix_68ab535465b06808",
        "sea_center_candidate_request_w2",
    }
    assert sources["route_matrix_68ab535465b06808"].source_type == "provider"
    assert all(
        source.source_type == "curated_dataset"
        for source_id, source in sources.items()
        if source_id != "route_matrix_68ab535465b06808"
    )
    assert all(source.source_url is None for source in sources.values())
    assert all(source.retrieved_at is None for source in sources.values())

    for place in catalog.places:
        assert place.description is None
        assert place.official_url is None
        assert place.informational_urls == ()
        assert place.source_freshness is None
        assert place.opening_hours_evidence_ref is None
        assert place.recommended_visit_minutes is None
        assert place.weather_suitability is None
        assert place.static_utility is None
        assert any(ref.startswith("registered_child_plan_") for ref in place.source_refs)
        assert any(ref.startswith("registered_certificate_") for ref in place.source_refs)
        assert "route_matrix_68ab535465b06808" in place.source_refs

    raw_places = _raw_catalog()["places"]
    assert isinstance(raw_places, list)
    unsupported = {
        "description",
        "official_url",
        "informational_urls",
        "opening_hours_evidence_ref",
        "recommended_visit_minutes",
        "parking_minutes",
        "walking_minutes",
        "waiting_minutes",
    }
    assert all(unsupported.isdisjoint(place) for place in raw_places)
    assert "recommended" not in (FIXTURE_ROOT / CATALOG_FILENAME).read_text(
        encoding="utf-8"
    ).casefold()


def test_builder_reads_but_does_not_mutate_any_pinned_source_artifact() -> None:
    before = {
        relative: _sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_HASHES
    }
    assert before == SOURCE_HASHES

    build_catalog_documents(repository_root=REPOSITORY_ROOT)

    after = {
        relative: _sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_HASHES
    }
    assert after == before


def test_builder_refuses_output_outside_the_product_catalog_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(CatalogBuildError, match="output_outside_catalog_root"):
        write_catalog(tmp_path / "catalog", repository_root=REPOSITORY_ROOT)

