"""Shared, deterministic contract for a local MapLibre/PMTiles package."""

from __future__ import annotations

import hashlib
import re
from typing import Any

EXPECTED_TOOL_VERSIONS = {
    "maplibre-gl": "5.24.0",
    "pmtiles-js": "4.4.1",
    "protomaps-basemap-style": "5.7.2",
    "pmtiles-cli": "1.30.0",
}
EXPECTED_LICENSE_NOTICES = {
    "BSD-3-Clause": "licenses/BSD-3-Clause.txt",
    "CC0-1.0": "licenses/CC0-1.0.txt",
    "MIT": "licenses/MIT.txt",
    "ODbL-1.0": "licenses/ODbL-1.0.txt",
    "OFL-1.1": "licenses/OFL-1.1.txt",
    "Protomaps-Data-Notices": "licenses/PROTOMAPS_DATA.txt",
}
EXPECTED_STYLE_ATTRIBUTION = (
    'Protomaps | <a href="https://www.openstreetmap.org/copyright" '
    'target="_blank" rel="noopener">© OpenStreetMap contributors</a>'
)
EXPECTED_SOURCE_BUILD = "20260804"
EXPECTED_SOURCE_URL = f"https://build.protomaps.com/{EXPECTED_SOURCE_BUILD}.pmtiles"
EXPECTED_UPSTREAM_BLAKE3 = (
    "665c468f538ffaebd6fdf5b1185982b105056577a32f2fdc15c87fddd4b24b95"
)
EXPECTED_UPSTREAM_PACKAGES = {
    "maplibre-gl": {
        "version": "5.24.0",
        "git_commit": "fd31bd85967c19446d836236e93021fb10ffe30d",
        "npm_integrity": (
            "sha512-ALyFxgtd5R+65UqZ/++lOqwWcC0SNho9c27fYSyLmG7AfnAul2o46F05aDJGPbFU57wos9dgcIySHs0Xe6ia3A=="
        ),
    },
    "pmtiles-js": {
        "version": "4.4.1",
        "git_commit": "0cebcaeade40034b86facb6e7da4ec726b9053fb",
        "npm_integrity": (
            "sha512-5oTeQc/yX/ft1evbpIlnoCZugQuug/iYIAj/ZTqIqzdGek4uZEho99En890EE6NOSI3JTI3IG8R7r8+SltphxA=="
        ),
    },
    "protomaps-basemaps": {
        "version": "5.7.2",
        "git_commit": "3ea8293a28131c3dc63f1bb20827bdb8a76df06f",
        "npm_integrity": (
            "sha512-K1Yk6bWdULulYg+R2QRVXx4NzJZan5YQhpejEG0c1/sXruJrfPIPZuakpf3jwAgVmjIRVQwAv+yRafDeN0aaUQ=="
        ),
    },
    "basemaps-assets": {
        "git_commit": "028c18f713baecad011301ff7a69acc39bcc2ae7"
    },
    "pmtiles-cli": {
        "version": "1.30.0",
        "git_commit": "39722fde57646c59c23e67a8cf739a59fc657590",
        "release_sha256": (
            "8a983059b1e7a24f6930f677f5b7aa6d01e125eb2e15cb0f34bd11247e82df35"
        ),
    },
}
REQUIRED_CORE_ASSETS = {
    "maplibre/maplibre-gl.js",
    "maplibre/maplibre-gl.css",
    "pmtiles/pmtiles.js",
    "styles/protomaps-light.json",
    *EXPECTED_LICENSE_NOTICES.values(),
}


def pmtiles_evidence_sha256(name: str, value: bytes) -> str:
    """Hash deterministic CLI evidence while removing verify timing noise."""

    normalized = value
    if name == "verify_output":
        text = value.decode("utf-8", errors="replace")
        text = re.sub(
            r"^.*verify\.go:\d+: Completed verify in [^\r\n]+\.\r?$",
            "pmtiles verify completed.",
            text,
            flags=re.MULTILINE,
        )
        normalized = text.encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def required_fontstacks(style: Any) -> set[str] | None:
    """Return every concrete font stack used by text symbol layers."""

    if not isinstance(style, dict) or not isinstance(style.get("layers"), list):
        return None
    required: set[str] = set()

    def collect(value: Any, *, direct: bool = False) -> bool:
        if isinstance(value, list) and value and all(
            isinstance(item, str) and item for item in value
        ) and direct:
            required.add(",".join(value))
            return True
        if not isinstance(value, list) or not value:
            return False
        if value[0] == "literal" and len(value) == 2:
            return collect(value[1], direct=True)
        found = False
        for item in value[1:]:
            if isinstance(item, list):
                found = collect(item) or found
        return found

    text_layers = 0
    for layer in style["layers"]:
        if not isinstance(layer, dict) or layer.get("type") != "symbol":
            continue
        layout = layer.get("layout")
        if not isinstance(layout, dict) or "text-field" not in layout:
            continue
        text_layers += 1
        if not collect(layout.get("text-font"), direct=True):
            return None
    return required if text_layers and required else None


def valid_provenance(value: Any, *, expected_artifact_url: str) -> bool:
    if not isinstance(value, dict) or value.get("schema_version") != "map-data-provenance-v1":
        return False
    source_url = value.get("source_url")
    build_match = (
        re.fullmatch(r"https://build\.protomaps\.com/(\d{8})\.pmtiles", source_url)
        if isinstance(source_url, str)
        else None
    )
    bounds = value.get("bounds")
    assets = value.get("assets")
    glyph_ranges = value.get("glyph_ranges")
    if not (
        value.get("source") == "OpenStreetMap"
        and build_match is not None
        and value.get("source_url") == EXPECTED_SOURCE_URL
        and value.get("source_build") == EXPECTED_SOURCE_BUILD == build_match.group(1)
        and value.get("license") == "ODbL-1.0"
        and isinstance(value.get("attribution"), str)
        and "© OpenStreetMap contributors" in value["attribution"]
        and value.get("attribution_url") == "https://www.openstreetmap.org/copyright"
        and value.get("artifact_url") == expected_artifact_url
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("artifact_sha256", "")))
        and isinstance(value.get("artifact_size"), int)
        and value["artifact_size"] > 1024
        and isinstance(bounds, list)
        and len(bounds) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in bounds)
        and bounds[0] <= -123.0
        and bounds[1] <= 33.5
        and bounds[2] >= -117.5
        and bounds[3] >= 38.5
        and isinstance(value.get("maxzoom"), int)
        and value["maxzoom"] >= 15
        and isinstance(value.get("generated_at"), str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+", value["generated_at"])
        and value.get("tool_versions") == EXPECTED_TOOL_VERSIONS
        and value.get("license_notices") == EXPECTED_LICENSE_NOTICES
        and isinstance(assets, list)
        and bool(assets)
        and isinstance(glyph_ranges, list)
        and bool(glyph_ranges)
        and all(isinstance(item, str) and item.startswith("fonts/") for item in glyph_ranges)
        and value.get("upstream_blake3") == EXPECTED_UPSTREAM_BLAKE3
        and value.get("upstream_packages") == EXPECTED_UPSTREAM_PACKAGES
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("header_json_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("metadata_json_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("verify_output_sha256", "")))
        and isinstance(value.get("extraction_command"), str)
        and "--bbox=-123.0,33.5,-117.5,38.5" in value["extraction_command"]
        and "--maxzoom=15" in value["extraction_command"]
    ):
        return False
    paths: set[str] = set()
    for entry in assets:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "license"}:
            return False
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or path in paths
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
            or not isinstance(entry.get("license"), str)
        ):
            return False
        paths.add(path)
    return REQUIRED_CORE_ASSETS.issubset(paths) and set(glyph_ranges).issubset(paths)
