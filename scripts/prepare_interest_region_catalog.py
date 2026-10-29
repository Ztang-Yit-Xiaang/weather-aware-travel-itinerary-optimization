"""Freeze the existing enriched POI catalog for interest exploration, not route acceptance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "results" / "outputs" / "production_enriched_poi_catalog.csv"
    payload = source.read_bytes()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    points = []
    for row in rows:
        latitude, longitude = float(row["latitude"]), float(row["longitude"])
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("exploration_coordinate_invalid")
        identity = json.dumps([row["name"], latitude, longitude], separators=(",", ":"))
        categories = []
        raw = row.get("category", "").casefold()
        if row.get("is_nature", "").casefold() in {"true", "1", "1.0"} or any(
            t in raw for t in ("park", "beach", "viewpoint", "hiking", "coast")
        ):
            categories.extend(["nature", "outdoor"])
        if any(t in raw for t in ("museum", "aquarium", "observatory", "campus", "histor", "courthouse", "art gall")):
            categories.append("culture")
        points.append(
            {
                "place_id": "interest_poi_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "name": row["name"],
                "display_coordinate": {"longitude": longitude, "latitude": latitude},
                "place_categories": categories,
                "category_source_text": row.get("category"),
                "source_refs": [row.get("source_url") or row.get("website_url") or row["source_list"]],
                "description": row.get("description") or None,
                "weather_suitability": None,
                "recommended_visit_minutes": None,
                "static_utility": None,
                "source_confidence": float(row["source_confidence"]) if row.get("source_confidence") else None,
            }
        )
    if len({p["place_id"] for p in points}) != len(points):
        raise ValueError("exploration_duplicate_identity")
    output = ROOT / "configs" / "interest_regions"
    park_source = output / "monterey_park_entries.geojson"
    park_payload = park_source.read_bytes()
    parks = json.loads(park_payload)
    suffixes = {"SP": "State Park", "SB": "State Beach", "SNR": "State Natural Reserve", "SHP": "State Historic Park"}
    park_service = "https://services2.arcgis.com/AhxrK3F6WM8ECvDi/arcgis/rest/services/ParkEntryPoints/FeatureServer/2"
    for feature in parks["features"]:
        attributes = feature["properties"]
        longitude, latitude = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] != "Point" or not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError("exploration_park_coordinate_invalid")
        stem, suffix = attributes["PARK_NAME"].rsplit(" ", 1)
        name = stem + " " + suffixes[suffix]
        points.append({
            "place_id": "csp_entry_" + attributes["GISID"].lower(),
            "name": name,
            "display_coordinate": {"longitude": longitude, "latitude": latitude},
            "place_categories": ["culture", "outdoor"] if suffix == "SHP" else ["nature", "outdoor"],
            "source_refs": [attributes["PARK_WEB_PG"], park_service],
            "description": "California State Parks mapped park entry. Current opening status and itinerary access have not been checked.",
            "location_kind": "park_entry",
            "source_name": attributes["PARK_NAME"],
            "source_attribution": "California State Parks",
            "recommended_visit_minutes": None,
            "weather_suitability": None,
            "static_utility": None,
        })
    if len({p["place_id"] for p in points}) != len(points):
        raise ValueError("exploration_duplicate_identity")
    output.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        {"schema_version": "interest-exploration-catalog-v1", "places": points},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (output / "exploration_catalog.json").write_bytes(data)
    manifest = {
        "schema_version": "interest-exploration-manifest-v1",
        "file": "exploration_catalog.json",
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(points),
        "data_role": "existing_POI_metadata_not_user_feedback",
        "routing_status": "not_evaluated",
        "supplemental_source": {
            "file": park_source.name,
            "sha256": hashlib.sha256(park_payload).hexdigest(),
            "service_url": park_service,
            "count": len(parks["features"]),
            "attribution": "California State Parks",
            "license_url": "https://www.parks.ca.gov/?page_id=29682",
        },
    }
    (output / "exploration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(points), "bytes": len(data), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
