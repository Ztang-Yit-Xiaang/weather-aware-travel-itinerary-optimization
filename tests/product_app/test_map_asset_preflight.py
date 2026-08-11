from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from itinerary_system.product_app.map_asset_contract import (
    EXPECTED_SOURCE_BUILD,
    EXPECTED_SOURCE_URL,
    EXPECTED_UPSTREAM_BLAKE3,
    EXPECTED_UPSTREAM_PACKAGES,
    pmtiles_evidence_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_local_map_assets.py"
PMTILES_EVIDENCE = {
    "header_json": b'{"specVersion":3}\n',
    "metadata_json": b'{"name":"test","vector_layers":[{"id":"roads"}]}\n',
    "verify_output": b"archive verified\n",
}


def load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("local_map_asset_validator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_package(root: Path) -> Path:
    archive = root / "data" / "california-coast-v1.pmtiles"
    archive.parent.mkdir(parents=True)
    header = bytearray(127)
    header[:7] = b"PMTiles"
    header[7] = 3
    for offset, value in ((8, 127), (16, 1), (24, 128), (32, 2), (40, 0), (48, 0), (56, 130), (64, 1)):
        struct.pack_into("<Q", header, offset, value)
    for offset in (72, 80, 88):
        struct.pack_into("<Q", header, offset, 1)
    header[96] = 1
    header[97] = 1
    header[98] = 1
    header[99] = 1
    header[100] = 0
    header[101] = 15
    for offset, value in zip((102, 106, 110, 114), (-123.0, 33.5, -117.5, 38.5), strict=True):
        struct.pack_into("<i", header, offset, round(value * 10_000_000))
    archive.write_bytes(bytes(header) + b"r{}t" + bytes(1917))

    style = root / "styles" / "protomaps-light.json"
    style.parent.mkdir(parents=True)
    style.write_text(
        json.dumps(
            {
                "version": 8,
                "sprite": "http://127.0.0.1:8080/sprites/light",
                "glyphs": "http://127.0.0.1:8080/fonts/{fontstack}/{range}.pbf",
                "sources": {
                    "protomaps": {
                        "type": "vector",
                        "url": (
                            "pmtiles://http://127.0.0.1:8080/data/"
                            "california-coast-v1.pmtiles"
                        ),
                        "attribution": (
                            'Protomaps | <a href="https://www.openstreetmap.org/copyright" '
                            'target="_blank" rel="noopener">'
                            "© OpenStreetMap contributors</a>"
                        ),
                    }
                },
                "layers": [
                    {
                        "id": "labels",
                        "type": "symbol",
                        "source": "protomaps",
                        "layout": {"text-field": ["get", "name"], "text-font": ["NotoSans"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    asset_specs = (
        ("maplibre/maplibre-gl.js", "BSD-3-Clause"),
        ("maplibre/maplibre-gl.css", "BSD-3-Clause"),
        ("pmtiles/pmtiles.js", "BSD-3-Clause"),
        ("styles/protomaps-light.json", "CC0-1.0"),
        ("fonts/NotoSans/0-255.pbf", "OFL-1.1"),
        ("sprites/light.json", "MIT"),
        ("sprites/light.png", "MIT"),
        ("sprites/light@2x.json", "MIT"),
        ("sprites/light@2x.png", "MIT"),
        ("licenses/BSD-3-Clause.txt", "BSD-3-Clause"),
        ("licenses/CC0-1.0.txt", "CC0-1.0"),
        ("licenses/MIT.txt", "MIT"),
        ("licenses/ODbL-1.0.txt", "ODbL-1.0"),
        ("licenses/OFL-1.1.txt", "OFL-1.1"),
        ("licenses/PROTOMAPS_DATA.txt", "Protomaps-Data-Notices"),
    )
    assets = []
    for relative, license_id in asset_specs:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(relative.encode())
        assets.append({"path": relative, "sha256": sha256(path), "license": license_id})
    provenance = {
        "schema_version": "map-data-provenance-v1",
        "source": "OpenStreetMap",
        "source_url": EXPECTED_SOURCE_URL,
        "source_build": EXPECTED_SOURCE_BUILD,
        "upstream_blake3": EXPECTED_UPSTREAM_BLAKE3,
        "upstream_packages": EXPECTED_UPSTREAM_PACKAGES,
        "license": "ODbL-1.0",
        "attribution": "Protomaps | © OpenStreetMap contributors",
        "attribution_url": "https://www.openstreetmap.org/copyright",
        "artifact_url": "/data/california-coast-v1.pmtiles",
        "artifact_sha256": sha256(archive),
        "artifact_size": archive.stat().st_size,
        "bounds": [-123.0, 33.5, -117.5, 38.5],
        "maxzoom": 15,
        "generated_at": "2026-08-04T00:00:00Z",
        "extraction_command": (
            "pmtiles extract SOURCE data/california-coast-v1.pmtiles "
            "--bbox=-123.0,33.5,-117.5,38.5 --maxzoom=15"
        ),
        "header_json_sha256": pmtiles_evidence_sha256(
            "header_json", PMTILES_EVIDENCE["header_json"]
        ),
        "metadata_json_sha256": pmtiles_evidence_sha256(
            "metadata_json", PMTILES_EVIDENCE["metadata_json"]
        ),
        "verify_output_sha256": pmtiles_evidence_sha256(
            "verify_output", PMTILES_EVIDENCE["verify_output"]
        ),
        "tool_versions": {
            "maplibre-gl": "5.24.0",
            "pmtiles-js": "4.4.1",
            "protomaps-basemap-style": "5.7.2",
            "pmtiles-cli": "1.30.0",
        },
        "license_notices": {
            "BSD-3-Clause": "licenses/BSD-3-Clause.txt",
            "CC0-1.0": "licenses/CC0-1.0.txt",
            "MIT": "licenses/MIT.txt",
            "ODbL-1.0": "licenses/ODbL-1.0.txt",
            "OFL-1.1": "licenses/OFL-1.1.txt",
            "Protomaps-Data-Notices": "licenses/PROTOMAPS_DATA.txt",
        },
        "glyph_ranges": ["fonts/NotoSans/0-255.pbf"],
        "assets": assets,
    }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return archive


def validate_package(validator: ModuleType, asset_root: Path) -> dict[str, object]:
    def runner(
        command: str, archive: Path, samples: list[tuple[int, int, int]]
    ) -> dict[str, bytes]:
        assert command == "pmtiles"
        assert archive == asset_root / "data" / "california-coast-v1.pmtiles"
        assert samples
        return PMTILES_EVIDENCE.copy()

    return validator.validate_package(asset_root, ROOT, pmtiles_runner=runner)


def test_preflight_binds_archive_assets_licenses_and_workspace_coverage(tmp_path: Path) -> None:
    validator = load_validator()
    build_package(tmp_path)

    result = validate_package(validator, tmp_path)

    assert result["status"] == "passed"
    assert result["archive_bounds"] == [-123.0, 33.5, -117.5, 38.5]
    assert result["asset_count"] == 15
    assert result["licenses"] == [
        "BSD-3-Clause",
        "CC0-1.0",
        "MIT",
        "ODbL-1.0",
        "OFL-1.1",
        "Protomaps-Data-Notices",
    ]


def test_preflight_rejects_archive_changed_after_provenance_was_written(tmp_path: Path) -> None:
    validator = load_validator()
    archive = build_package(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(validator.MapAssetValidationError, match="artifact_hash_mismatch"):
        validate_package(validator, tmp_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "source_url",
        "source_build",
        "upstream_blake3",
        "upstream_packages",
        "tool_versions",
        "license_notices",
    ],
)
def test_preflight_rejects_incomplete_source_and_license_provenance(
    tmp_path: Path, missing_field: str
) -> None:
    validator = load_validator()
    build_package(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance[missing_field]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(validator.MapAssetValidationError, match="provenance_invalid"):
        validate_package(validator, tmp_path)


@pytest.mark.parametrize("field", ["upstream_blake3", "upstream_packages"])
def test_preflight_rejects_unpinned_upstream_provenance(
    tmp_path: Path, field: str
) -> None:
    validator = load_validator()
    build_package(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if field == "upstream_blake3":
        provenance[field] = "0" * 64
    else:
        provenance[field]["maplibre-gl"]["git_commit"] = "0" * 40
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(validator.MapAssetValidationError, match="provenance_invalid"):
        validate_package(validator, tmp_path)


def test_preflight_rejects_style_font_without_matching_glyphs(tmp_path: Path) -> None:
    validator = load_validator()
    build_package(tmp_path)
    style_path = tmp_path / "styles" / "protomaps-light.json"
    style = json.loads(style_path.read_text(encoding="utf-8"))
    style["layers"][0]["layout"]["text-font"] = ["Definitely Missing Font"]
    style_path.write_text(json.dumps(style), encoding="utf-8")
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for asset in provenance["assets"]:
        if asset["path"] == "styles/protomaps-light.json":
            asset["sha256"] = sha256(style_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(validator.MapAssetValidationError, match="glyph_fontstack_incomplete"):
        validate_package(validator, tmp_path)


def test_preflight_rejects_unbound_pmtiles_cli_evidence(tmp_path: Path) -> None:
    validator = load_validator()
    build_package(tmp_path)

    with pytest.raises(validator.MapAssetValidationError, match="pmtiles_evidence_mismatch"):
        validator.validate_package(
            tmp_path,
            ROOT,
            pmtiles_runner=lambda _command, _archive, _samples: {
                **PMTILES_EVIDENCE,
                "verify_output": b"different output",
            },
        )


def test_verify_evidence_hash_ignores_cli_timestamp_and_duration() -> None:
    first = b"2026/08/04 19:26:19 verify.go:169: Completed verify in 43.7175ms.\n"
    second = b"2026/08/04 20:40:01 verify.go:169: Completed verify in 91.2ms.\n"

    assert pmtiles_evidence_sha256("verify_output", first) == pmtiles_evidence_sha256(
        "verify_output", second
    )


def test_pmtiles_cli_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    validator = load_validator()
    archive = build_package(tmp_path)

    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"invalid archive"),
    )
    with pytest.raises(validator.MapAssetValidationError, match="pmtiles_cli_failed"):
        validator._run_pmtiles("pmtiles", archive, [(8, 0, 0)])


@pytest.mark.parametrize("reported_version", [b"pmtiles 11.30.0\n", b"pmtiles 1.30.0-dev\n"])
def test_pmtiles_cli_requires_exact_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reported_version: bytes
) -> None:
    validator = load_validator()
    archive = build_package(tmp_path)
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, reported_version, b""),
    )

    with pytest.raises(validator.MapAssetValidationError, match="pmtiles_cli_version_mismatch"):
        validator._run_pmtiles("pmtiles", archive, [(8, 0, 0)])
