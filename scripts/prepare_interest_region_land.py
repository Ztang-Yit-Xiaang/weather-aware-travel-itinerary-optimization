"""Prepare a small, source-pinned California land/water mask (Natural Earth public domain).

Only this preparation command downloads data. Product requests use the saved
GeoJSON and verify its hash. No GIS dependency or worldwide archive is retained.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = (-125.0, 32.0, -113.0, 43.0)


def _overlaps(bounds: tuple[float, ...]) -> bool:
    west, south, east, north = bounds
    return west <= COVERAGE[2] and east >= COVERAGE[0] and south <= COVERAGE[3] and north >= COVERAGE[1]


def _polygons(payload: bytes) -> list[list[list[list[float]]]]:
    polygons = []
    offset = 100
    while offset + 8 <= len(payload):
        _, length = struct.unpack_from(">2i", payload, offset)
        record = payload[offset + 8 : offset + 8 + 2 * length]
        offset += 8 + 2 * length
        if len(record) < 44 or struct.unpack_from("<i", record)[0] != 5:
            continue
        if not _overlaps(struct.unpack_from("<4d", record, 4)):
            continue
        count, point_count = struct.unpack_from("<2i", record, 36)
        starts = [*struct.unpack_from(f"<{count}i", record, 44), point_count]
        coordinates = struct.unpack_from(f"<{point_count * 2}d", record, 44 + 4 * count)
        current = None
        for start, stop in zip(starts[:-1], starts[1:], strict=True):
            ring = [[round(coordinates[2 * i], 5), round(coordinates[2 * i + 1], 5)] for i in range(start, stop)]
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring[:-1], ring[1:], strict=True))
            if area < 0:
                bounds = (
                    min(p[0] for p in ring),
                    min(p[1] for p in ring),
                    max(p[0] for p in ring),
                    max(p[1] for p in ring),
                )
                current = [ring] if _overlaps(bounds) else None
                if current is not None:
                    polygons.append(current)
            elif current is not None:
                current.append(ring)
    return polygons


def main() -> None:
    output = ROOT / "configs" / "interest_regions"
    output.mkdir(parents=True, exist_ok=True)
    features, sources = [], []
    for kind, filename in (("land", "ne_10m_land"), ("water", "ne_10m_lakes")):
        url = f"https://naciscdn.org/naturalearth/10m/physical/{filename}.zip"
        with urllib.request.urlopen(url, timeout=45) as response:
            archive = response.read(12_000_001)
        if len(archive) > 12_000_000:
            raise ValueError("land_source_too_large")
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            geometry = _polygons(zipped.read(f"{filename}.shp"))
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": kind},
                "geometry": {"type": "MultiPolygon", "coordinates": geometry},
            }
        )
        sources.append({"kind": kind, "url": url, "archive_sha256": hashlib.sha256(archive).hexdigest()})
    payload = json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode()
    (output / "land_support.geojson").write_bytes(payload)
    manifest = {
        "schema_version": "interest-land-support-v1",
        "file": "land_support.geojson",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "coverage_bbox": list(COVERAGE),
        "source": "Natural Earth 1:10m land and lakes",
        "license": "public domain",
        "sources": sources,
        "precision_note": "Cartographic land/water mask, not an accessibility or closure certificate.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "bytes": len(payload),
                "polygons": [len(f["geometry"]["coordinates"]) for f in features],
                "sha256": manifest["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
