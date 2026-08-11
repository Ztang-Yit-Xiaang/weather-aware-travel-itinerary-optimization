"""Create the closed provenance manifest for a staged local map package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from itinerary_system.product_app.map_asset_contract import (  # noqa: E402
    EXPECTED_LICENSE_NOTICES,
    EXPECTED_SOURCE_BUILD,
    EXPECTED_SOURCE_URL,
    EXPECTED_TOOL_VERSIONS,
    EXPECTED_UPSTREAM_BLAKE3,
    EXPECTED_UPSTREAM_PACKAGES,
    pmtiles_evidence_sha256,
)

PMTILES_CLI_COMMIT = "39722fde57646c59c23e67a8cf739a59fc657590"
ARCHIVE_RELATIVE = Path("data/california-coast-v1.pmtiles")
RUNTIME_DIRECTORIES = ("maplibre", "pmtiles", "styles", "sprites", "fonts", "licenses")


class FinalizeError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> bytes:
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if completed.returncode != 0:
        raise FinalizeError("pmtiles_cli_failed")
    return completed.stdout + completed.stderr


def asset_license(relative: str) -> str:
    if relative.startswith(("maplibre/", "pmtiles/")):
        return "BSD-3-Clause"
    if relative.startswith("styles/") or relative.endswith("CC0-1.0.txt"):
        return "CC0-1.0"
    if relative.startswith("fonts/") or relative.endswith("OFL-1.1.txt"):
        return "OFL-1.1"
    if relative.startswith("sprites/") or relative.endswith("MIT.txt"):
        return "MIT"
    if relative.endswith("ODbL-1.0.txt"):
        return "ODbL-1.0"
    if relative.endswith("PROTOMAPS_DATA.txt"):
        return "Protomaps-Data-Notices"
    if relative.endswith(
        ("BSD-3-Clause.txt", "PMTILES-BSD-3-Clause.txt", "PROTOMAPS_BASEMAPS_LICENSE.md")
    ):
        return "BSD-3-Clause"
    raise FinalizeError(f"unmapped_asset_license:{relative}")


def collect_assets(root: Path) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for directory in RUNTIME_DIRECTORIES:
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink():
                raise FinalizeError("symlink_not_allowed")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            assets.append(
                {"path": relative, "sha256": sha256(path), "license": asset_license(relative)}
            )
    return assets


def finalize(asset_root: Path, pmtiles_cli: Path) -> dict[str, Any]:
    root = asset_root.resolve()
    archive = (root / ARCHIVE_RELATIVE).resolve()
    if not archive.is_file() or root not in archive.parents:
        raise FinalizeError("archive_missing")
    version = run([str(pmtiles_cli), "version"]).decode("utf-8", errors="replace").strip()
    if re.fullmatch(
        rf"pmtiles 1\.30\.0, commit {PMTILES_CLI_COMMIT}, built at \S+", version
    ) is None:
        raise FinalizeError("pmtiles_cli_version_mismatch")
    evidence = {
        "header_json": run([str(pmtiles_cli), "show", str(archive), "--header-json"]),
        "metadata_json": run([str(pmtiles_cli), "show", str(archive), "--metadata"]),
        "verify_output": run([str(pmtiles_cli), "verify", str(archive)]),
    }
    try:
        header = json.loads(evidence["header_json"].decode("utf-8"))
        metadata = json.loads(evidence["metadata_json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizeError("pmtiles_evidence_invalid") from exc
    if not isinstance(metadata.get("vector_layers"), list) or not metadata["vector_layers"]:
        raise FinalizeError("pmtiles_vector_layers_missing")
    assets = collect_assets(root)
    glyph_ranges = [item["path"] for item in assets if item["path"].startswith("fonts/")]
    provenance: dict[str, Any] = {
        "schema_version": "map-data-provenance-v1",
        "source": "OpenStreetMap",
        "source_url": EXPECTED_SOURCE_URL,
        "source_build": EXPECTED_SOURCE_BUILD,
        "upstream_blake3": EXPECTED_UPSTREAM_BLAKE3,
        "license": "ODbL-1.0",
        "attribution": "Protomaps | \u00a9 OpenStreetMap contributors",
        "attribution_url": "https://www.openstreetmap.org/copyright",
        "artifact_url": f"/{ARCHIVE_RELATIVE.as_posix()}",
        "artifact_sha256": sha256(archive),
        "artifact_size": archive.stat().st_size,
        "bounds": header["bounds"],
        "maxzoom": header["maxzoom"],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "extraction_command": (
            f"pmtiles extract {EXPECTED_SOURCE_URL} {ARCHIVE_RELATIVE.as_posix()} "
            "--bbox=-123.0,33.5,-117.5,38.5 --maxzoom=15"
        ),
        "header_json_sha256": pmtiles_evidence_sha256(
            "header_json", evidence["header_json"]
        ),
        "metadata_json_sha256": pmtiles_evidence_sha256(
            "metadata_json", evidence["metadata_json"]
        ),
        "verify_output_sha256": pmtiles_evidence_sha256(
            "verify_output", evidence["verify_output"]
        ),
        "tool_versions": EXPECTED_TOOL_VERSIONS,
        "license_notices": EXPECTED_LICENSE_NOTICES,
        "glyph_ranges": glyph_ranges,
        "assets": assets,
        "upstream_packages": EXPECTED_UPSTREAM_PACKAGES,
    }
    destination = root / "provenance.json"
    temporary = root / "provenance.json.tmp"
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pmtiles-cli", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = finalize(args.asset_root, args.pmtiles_cli.resolve())
    except (FinalizeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "finalized",
                "artifact_sha256": result["artifact_sha256"],
                "artifact_size": result["artifact_size"],
                "asset_count": len(result["assets"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
