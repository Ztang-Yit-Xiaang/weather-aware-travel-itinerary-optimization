"""Validate a local MapLibre/PMTiles package before the live G1 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from itinerary_system.product_app.map_asset_contract import (  # noqa: E402
    EXPECTED_LICENSE_NOTICES,
    EXPECTED_STYLE_ATTRIBUTION,
    REQUIRED_CORE_ASSETS,
    pmtiles_evidence_sha256,
    required_fontstacks,
    valid_provenance,
)
from itinerary_system.product_app.registry import ProductRunRegistry  # noqa: E402
from itinerary_system.product_app.service import ProductService  # noqa: E402

REQUIRED_LICENSES = frozenset(
    {"BSD-3-Clause", "CC0-1.0", "OFL-1.1", "MIT", "ODbL-1.0", "Protomaps-Data-Notices"}
)
MAP_ORIGIN = "http://127.0.0.1:8080"
PMTILES_CLI_COMMIT = "39722fde57646c59c23e67a8cf739a59fc657590"


class MapAssetValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise MapAssetValidationError("invalid_asset_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise MapAssetValidationError("asset_path_escape") from exc
    if not path.is_file():
        raise MapAssetValidationError("asset_missing")
    return path


def _pmtiles_header(path: Path) -> tuple[list[float], int]:
    with path.open("rb") as handle:
        header = handle.read(127)
    if len(header) != 127 or header[:7] != b"PMTiles" or header[7] != 3:
        raise MapAssetValidationError("pmtiles_header_invalid")
    file_size = path.stat().st_size
    sections = [
        (struct.unpack_from("<Q", header, offset)[0], struct.unpack_from("<Q", header, offset + 8)[0])
        for offset in (8, 24, 40, 56)
    ]
    root, metadata, leaf, tile_data = sections
    counts = [struct.unpack_from("<Q", header, offset)[0] for offset in (72, 80, 88)]
    if (
        root[1] == 0
        or metadata[1] == 0
        or tile_data[1] == 0
        or any(offset + length > file_size for offset, length in sections)
        or any(count == 0 for count in counts)
        or header[96] not in {0, 1}
        or header[97] not in {1, 2, 3, 4}
        or header[98] not in {1, 2, 3, 4}
        or header[99] != 1
        or header[100] > header[101]
        or (leaf[0] != 0 and leaf[0] < metadata[0] + metadata[1])
    ):
        raise MapAssetValidationError("pmtiles_structure_invalid")
    maxzoom = header[101]
    bounds = [
        struct.unpack_from("<i", header, 102)[0] / 10_000_000,
        struct.unpack_from("<i", header, 106)[0] / 10_000_000,
        struct.unpack_from("<i", header, 110)[0] / 10_000_000,
        struct.unpack_from("<i", header, 114)[0] / 10_000_000,
    ]
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise MapAssetValidationError("pmtiles_bounds_invalid")
    return bounds, maxzoom


def _workspace_coordinates(repository_root: Path) -> list[tuple[float, float]]:
    registry = ProductRunRegistry(
        repository_root, repository_root / "configs" / "product_app_registry.json"
    )
    service = ProductService(registry, repository_root / ".product_app_state")
    coordinates: list[tuple[float, float]] = []
    for record in registry.all():
        workspace = service.workspace_view(record.run_id)
        for plan_name in ("parent", "child"):
            plan_map = workspace["map"][plan_name]
            for stop in plan_map["stops"]:
                coordinates.append((float(stop["longitude"]), float(stop["latitude"])))
            for segment in plan_map["segments"]:
                coordinates.extend(
                    (float(item[1]), float(item[0])) for item in segment["coordinates"]
                )
    if not coordinates:
        raise MapAssetValidationError("workspace_coordinates_missing")
    return coordinates


def _bounds(coordinates: list[tuple[float, float]]) -> list[float]:
    longitudes, latitudes = zip(*coordinates, strict=True)
    return [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]


def _covers(container: list[float], subject: list[float]) -> bool:
    return (
        container[0] <= subject[0]
        and container[1] <= subject[1]
        and container[2] >= subject[2]
        and container[3] >= subject[3]
    )


def _command_output(command: list[str]) -> bytes:
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MapAssetValidationError("pmtiles_cli_unavailable") from exc
    if completed.returncode != 0:
        raise MapAssetValidationError("pmtiles_cli_failed")
    return completed.stdout + completed.stderr


def _tile_coordinate(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    scale = 1 << zoom
    x = int((longitude + 180.0) / 360.0 * scale)
    latitude_radians = math.radians(max(-85.05112878, min(85.05112878, latitude)))
    y = int(
        (1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2.0 * scale
    )
    return max(0, min(scale - 1, x)), max(0, min(scale - 1, y))


def _run_pmtiles(
    pmtiles_cli: str,
    archive: Path,
    samples: list[tuple[int, int, int]],
) -> dict[str, bytes]:
    version = _command_output([pmtiles_cli, "version"])
    version_text = version.decode("utf-8", errors="replace").strip()
    expected_version = rf"pmtiles 1\.30\.0, commit {PMTILES_CLI_COMMIT}, built at \S+"
    if re.fullmatch(expected_version, version_text) is None:
        raise MapAssetValidationError("pmtiles_cli_version_mismatch")
    evidence = {
        "header_json": _command_output([pmtiles_cli, "show", str(archive), "--header-json"]),
        "metadata_json": _command_output([pmtiles_cli, "show", str(archive), "--metadata"]),
        "verify_output": _command_output([pmtiles_cli, "verify", str(archive)]),
    }
    for zoom, x, y in samples:
        if not _command_output([pmtiles_cli, "tile", str(archive), str(zoom), str(x), str(y)]):
            raise MapAssetValidationError("pmtiles_sample_tile_missing")
    return evidence


def validate_package(
    asset_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
    *,
    pmtiles_cli: str = "pmtiles",
    pmtiles_runner: Callable[
        [str, Path, list[tuple[int, int, int]]], dict[str, bytes]
    ] = _run_pmtiles,
) -> dict[str, Any]:
    root = asset_root.resolve()
    provenance_path = _safe_file(root, "provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapAssetValidationError("provenance_invalid") from exc
    artifact_url = provenance.get("artifact_url")
    if not isinstance(artifact_url, str) or not artifact_url.startswith("/data/"):
        raise MapAssetValidationError("artifact_url_invalid")
    if not valid_provenance(provenance, expected_artifact_url=artifact_url):
        raise MapAssetValidationError("provenance_invalid")
    archive = _safe_file(root, artifact_url.removeprefix("/"))
    actual_hash = _sha256(archive)
    if provenance.get("artifact_sha256") != actual_hash:
        raise MapAssetValidationError("artifact_hash_mismatch")
    if provenance.get("artifact_size") != archive.stat().st_size:
        raise MapAssetValidationError("artifact_size_mismatch")
    header_bounds, maxzoom = _pmtiles_header(archive)
    declared_bounds = provenance.get("bounds")
    if declared_bounds != header_bounds or provenance.get("maxzoom") != maxzoom:
        raise MapAssetValidationError("artifact_header_mismatch")
    workspace_coordinates = _workspace_coordinates(repository_root)
    sample_coordinates = list(dict.fromkeys(workspace_coordinates))
    samples = list(
        dict.fromkeys(
            (zoom, *_tile_coordinate(longitude, latitude, zoom))
            for longitude, latitude in sample_coordinates
            for zoom in (8, 12)
        )
    )
    command_evidence = pmtiles_runner(pmtiles_cli, archive, samples)
    expected_evidence = {
        "header_json": provenance["header_json_sha256"],
        "metadata_json": provenance["metadata_json_sha256"],
        "verify_output": provenance["verify_output_sha256"],
    }
    if set(command_evidence) != set(expected_evidence) or any(
        not isinstance(output, bytes)
        or pmtiles_evidence_sha256(name, output) != expected_evidence[name]
        for name, output in command_evidence.items()
    ):
        raise MapAssetValidationError("pmtiles_evidence_mismatch")
    try:
        header_evidence = json.loads(command_evidence["header_json"].decode("utf-8"))
        metadata_evidence = json.loads(command_evidence["metadata_json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapAssetValidationError("pmtiles_evidence_invalid") from exc
    vector_layers = metadata_evidence.get("vector_layers") if isinstance(metadata_evidence, dict) else None
    if not isinstance(header_evidence, dict) or not isinstance(vector_layers, list) or not vector_layers:
        raise MapAssetValidationError("pmtiles_evidence_invalid")
    required_bounds = _bounds(workspace_coordinates)
    if not _covers(header_bounds, required_bounds):
        raise MapAssetValidationError("workspace_coverage_incomplete")

    assets = provenance.get("assets")
    if not isinstance(assets, list) or not assets:
        raise MapAssetValidationError("asset_manifest_missing")
    observed_licenses: set[str] = set()
    manifested_paths: set[str] = set()
    for entry in assets:
        if not isinstance(entry, dict):
            raise MapAssetValidationError("asset_manifest_invalid")
        path = _safe_file(root, entry.get("path"))
        relative = path.relative_to(root).as_posix()
        if relative in manifested_paths:
            raise MapAssetValidationError("duplicate_asset_path")
        manifested_paths.add(relative)
        if entry.get("sha256") != _sha256(path):
            raise MapAssetValidationError("asset_hash_mismatch")
        license_id = entry.get("license")
        if not isinstance(license_id, str):
            raise MapAssetValidationError("asset_license_missing")
        observed_licenses.add(license_id)
    if not REQUIRED_LICENSES.issubset(observed_licenses) or not REQUIRED_CORE_ASSETS.issubset(
        manifested_paths
    ):
        raise MapAssetValidationError("license_inventory_incomplete")
    actual_runtime_assets = {
        path.relative_to(root).as_posix()
        for directory in ("maplibre", "pmtiles", "styles", "sprites", "fonts", "licenses")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    if actual_runtime_assets != manifested_paths:
        raise MapAssetValidationError("asset_manifest_not_closed")
    for notice_path in EXPECTED_LICENSE_NOTICES.values():
        _safe_file(root, notice_path)

    style_path = _safe_file(root, "styles/protomaps-light.json")
    style = json.loads(style_path.read_text(encoding="utf-8"))
    sources = style.get("sources") if isinstance(style, dict) else None
    if not isinstance(sources, dict):
        raise MapAssetValidationError("map_style_invalid")
    if (
        style.get("sprite") != f"{MAP_ORIGIN}/sprites/light"
        or style.get("glyphs") != f"{MAP_ORIGIN}/fonts/{{fontstack}}/{{range}}.pbf"
        or style.get("imports", [])
    ):
        raise MapAssetValidationError("map_style_assets_invalid")
    for sprite_path in (
        "sprites/light.json",
        "sprites/light.png",
        "sprites/light@2x.json",
        "sprites/light@2x.png",
    ):
        if sprite_path not in manifested_paths:
            raise MapAssetValidationError("sprite_manifest_incomplete")
    actual_glyphs = {path for path in manifested_paths if path.startswith("fonts/")}
    declared_glyphs = set(provenance["glyph_ranges"])
    if declared_glyphs != actual_glyphs:
        raise MapAssetValidationError("glyph_manifest_incomplete")
    fontstacks = required_fontstacks(style)
    if fontstacks is None:
        raise MapAssetValidationError("style_fontstack_invalid")
    if not {f"fonts/{fontstack}/0-255.pbf" for fontstack in fontstacks}.issubset(
        declared_glyphs
    ):
        raise MapAssetValidationError("glyph_fontstack_incomplete")
    for source in sources.values():
        if not isinstance(source, dict):
            raise MapAssetValidationError("map_style_invalid")
        urls = [source.get(name) for name in ("url", "data") if source.get(name) is not None]
        tiles = source.get("tiles", [])
        if not isinstance(tiles, list) or any(not isinstance(item, str) for item in tiles):
            raise MapAssetValidationError("map_style_invalid")
        urls.extend(tiles)
        for value in urls:
            if not isinstance(value, str):
                raise MapAssetValidationError("map_style_invalid")
            parsed = urlparse(value.removeprefix("pmtiles://"))
            if parsed.scheme != "http" or parsed.netloc != "127.0.0.1:8080":
                raise MapAssetValidationError("map_style_url_not_loopback")
    matching_sources = [
        source
        for source in sources.values()
        if isinstance(source, dict)
        and urlparse(str(source.get("url", "")).removeprefix("pmtiles://")).scheme == "http"
        and urlparse(str(source.get("url", "")).removeprefix("pmtiles://")).netloc
        == "127.0.0.1:8080"
        and urlparse(str(source.get("url", "")).removeprefix("pmtiles://")).path
        == artifact_url
    ]
    if len(matching_sources) != 1:
        raise MapAssetValidationError("style_artifact_mismatch")
    attribution = matching_sources[0].get("attribution")
    if attribution != EXPECTED_STYLE_ATTRIBUTION:
        raise MapAssetValidationError("map_attribution_invalid")
    return {
        "schema_version": "local-map-preflight-v1",
        "status": "passed",
        "archive_sha256": actual_hash,
        "archive_size": archive.stat().st_size,
        "archive_bounds": header_bounds,
        "workspace_bounds": required_bounds,
        "maxzoom": maxzoom,
        "asset_count": len(assets),
        "sample_tile_count": len(samples),
        "licenses": sorted(observed_licenses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--pmtiles-cli", default="pmtiles")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_package(
            args.asset_root,
            args.repository_root.resolve(),
            pmtiles_cli=args.pmtiles_cli,
        )
    except (MapAssetValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
