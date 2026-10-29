"""Build the bounded California statewide POI discovery catalog.

This catalog is a name-aware discovery layer, not a route-feasibility or
recommendation artifact. Coordinates come from the repository's curated
California regional scenario. Every place intentionally has no routing access
point; runtime routing must validate access before any itinerary operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from itinerary_system.region_scenarios import (  # noqa: E402
    CALIFORNIA_STATEWIDE_NATURE_REGIONS,
    SCENARIOS,
)

OUTPUT_RELATIVE = Path("configs/product_poi_catalogs/california_statewide")
CATALOG_FILENAME = "catalog.json"
MANIFEST_FILENAME = "manifest.json"
CATALOG_ID = "california_statewide_discovery_v1"
GENERATED_AT = "2026-08-13T00:00:00+00:00"

NATURE_SOURCE_ID = "california_statewide_nature_regions_v1"
GATEWAY_SOURCE_ID = "california_statewide_gateway_coordinates_v1"
EXCLUDED_NON_CALIFORNIA_COORDINATES = frozenset({"Las Vegas"})
ALIASES: dict[str, tuple[str, ...]] = {
    "Redwood National and State Parks": ("Redwood", "Redwoods", "Red Wood", "Redwood Parks"),
    "Yosemite National Park": ("Yosemite",),
    "Sequoia National Park": ("Sequoia",),
    "Kings Canyon National Park": ("Kings Canyon",),
    "Joshua Tree National Park": ("Joshua Tree",),
    "Death Valley National Park": ("Death Valley",),
    "Pinnacles National Park": ("Pinnacles",),
    "Lassen Volcanic National Park": ("Lassen", "Lassen Volcanic"),
    "Point Reyes": ("Point Reyes National Seashore",),
    "Lake Tahoe": ("Tahoe",),
    "Big Sur": ("Big Sur Coast",),
    "Los Angeles": ("LA",),
    "San Francisco": ("SF",),
    "San Luis Obispo": ("SLO",),
    "South Lake Tahoe": ("Tahoe Gateway",),
}


class StatewideCatalogBuildError(ValueError):
    """Stable deterministic statewide catalog build failure."""


def _slug(value: str) -> str:
    return "_".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def _place(
    *,
    place_id: str,
    name: str,
    latitude: float,
    longitude: float,
    categories: list[str],
    region_id: str,
    source_id: str,
) -> dict[str, Any]:
    return {
        "access_points": [],
        "aliases": list(ALIASES.get(name, ())),
        "display_coordinate": {"latitude": float(latitude), "longitude": float(longitude)},
        "name": name,
        "place_categories": categories,
        "place_id": place_id,
        "region_id": region_id,
        "source_refs": [source_id],
    }


def build_catalog_documents() -> tuple[bytes, bytes]:
    scenario = SCENARIOS["california_statewide_nature"]
    places = [
        _place(
            place_id=f"ca_region_{region.region_id}",
            name=region.name,
            latitude=region.latitude,
            longitude=region.longitude,
            categories=["statewide_discovery", region.park_type],
            region_id=region.region_id,
            source_id=NATURE_SOURCE_ID,
        )
        for region in CALIFORNIA_STATEWIDE_NATURE_REGIONS
    ]
    nature_names = {region.name for region in CALIFORNIA_STATEWIDE_NATURE_REGIONS}
    gateway_names = sorted(
        set(scenario.coordinate_fallbacks)
        - nature_names
        - EXCLUDED_NON_CALIFORNIA_COORDINATES
    )
    for name in gateway_names:
        coordinate = scenario.coordinate_fallbacks.get(name)
        if coordinate is None:
            raise StatewideCatalogBuildError("gateway_coordinate_missing")
        places.append(
            _place(
                place_id=f"ca_gateway_{_slug(name)}",
                name=name,
                latitude=coordinate[0],
                longitude=coordinate[1],
                categories=["statewide_discovery", "destination_gateway"],
                region_id="california_gateway",
                source_id=GATEWAY_SOURCE_ID,
            )
        )
    catalog = {
        "catalog_id": CATALOG_ID,
        "generated_at": GENERATED_AT,
        "places": sorted(places, key=lambda item: item["place_id"]),
        "schema_version": "product-poi-catalog-v1",
        "sources": [
            {
                "retrieved_at": None,
                "source_id": GATEWAY_SOURCE_ID,
                "source_type": "curated_dataset",
                "source_url": None,
            },
            {
                "retrieved_at": None,
                "source_id": NATURE_SOURCE_ID,
                "source_type": "curated_dataset",
                "source_url": None,
            },
        ],
    }
    catalog_bytes = _stable_bytes(catalog)
    manifest = {
        "catalog_file": CATALOG_FILENAME,
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "schema_version": "product-poi-catalog-manifest-v1",
    }
    return catalog_bytes, _stable_bytes(manifest)


def write_catalog(output_dir: Path | None = None) -> Path:
    root = REPOSITORY_ROOT.resolve(strict=True)
    allowed = (root / "configs" / "product_poi_catalogs").resolve(strict=True)
    target = (output_dir or root / OUTPUT_RELATIVE).resolve()
    try:
        relative = target.relative_to(allowed)
    except ValueError as exc:
        raise StatewideCatalogBuildError("output_outside_catalog_root") from exc
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise StatewideCatalogBuildError("output_directory_invalid")
    target.mkdir(parents=True, exist_ok=True)
    catalog, manifest = build_catalog_documents()
    _atomic_write(target / CATALOG_FILENAME, catalog)
    _atomic_write(target / MANIFEST_FILENAME, manifest)
    return target / MANIFEST_FILENAME


def check_catalog() -> None:
    catalog, manifest = build_catalog_documents()
    target = REPOSITORY_ROOT / OUTPUT_RELATIVE
    if (target / CATALOG_FILENAME).read_bytes() != catalog:
        raise StatewideCatalogBuildError("committed_catalog_not_deterministic")
    if (target / MANIFEST_FILENAME).read_bytes() != manifest:
        raise StatewideCatalogBuildError("committed_manifest_not_deterministic")


def _stable_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        if args.output is not None:
            raise StatewideCatalogBuildError("check_output_conflict")
        check_catalog()
    else:
        write_catalog(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
