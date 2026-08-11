"""Build the deterministic California Coast product POI catalog.

The catalog contains only facts already present in the immutable product-demo
package and its pinned route matrix.  It deliberately does not add descriptions,
opening hours, URLs, visit guidance, or recommendation claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = Path("configs/product_poi_catalogs/california_coast_demo")
PACKAGE_RELATIVE = Path("runs/california-coast-product-demo-v2/manifest.json")
MATRIX_RELATIVE = Path(
    "runs/e3ux-weather-repair-demo-v6/routing/route_matrix_68ab535465b06808.json"
)

CATALOG_SCHEMA = "product-poi-catalog-v1"
MANIFEST_SCHEMA = "product-poi-catalog-manifest-v1"
CATALOG_FILENAME = "catalog.json"
MANIFEST_FILENAME = "manifest.json"
CATALOG_ID = "california_coast_demo_candidates_v1"

PINNED_PACKAGE_SHA256 = (
    "925eea6e5722a782d48f657efa931e18536e502a7803a839a983747ed79b5e40"
)
PINNED_MATRIX_FILE_SHA256 = (
    "6f3b5b8eda1a9f73e28b6404ef15126086dbd7861c683790af6f6e87b41b4677"
)
PINNED_MATRIX_ID = "route_matrix_68ab535465b06808"
PINNED_MATRIX_CONTENT_SHA256 = (
    "3d2ded2d2a862aa6bcf026429b0a0376386928cfa3af109758301d1647584f71"
)

DAY_7_ORIGIN = "hotel_milo_santa_barbara"
DAY_7_DESTINATION = "san_francisco_international_airport"

PLACE_SPECS = (
    {
        "place_id": "bixby_creek_bridge_viewpoint",
        "request_file": (
            "alternatives/w2_weather_recommended_v2/requests/"
            "w2_weather_recommended_v2.json"
        ),
        "plan_file": (
            "alternatives/w2_weather_recommended_v2/plans/"
            "plan_f5ee52459659dcb5.json"
        ),
        "certificate_file": (
            "alternatives/w2_weather_recommended_v2/evaluations/"
            "cert_686ef65d376b2867.json"
        ),
        "plan_id": "plan_f5ee52459659dcb5",
        "certificate_id": "cert_686ef65d376b2867",
        "request_source_id": "bixby_candidate_request_w2",
        "plan_source_id": "registered_child_plan_f5ee52459659dcb5",
        "certificate_source_id": "registered_certificate_686ef65d376b2867",
    },
    {
        "place_id": "santa_barbara_museum_of_natural_history_sea_center",
        "request_file": (
            "alternatives/w2_weather_low_driving_v2/requests/"
            "w2_weather_low_driving_v2.json"
        ),
        "plan_file": (
            "alternatives/w2_weather_low_driving_v2/plans/"
            "plan_8aa919c8323dbac0.json"
        ),
        "certificate_file": (
            "alternatives/w2_weather_low_driving_v2/evaluations/"
            "cert_5a6deef4c159d346.json"
        ),
        "plan_id": "plan_8aa919c8323dbac0",
        "certificate_id": "cert_5a6deef4c159d346",
        "request_source_id": "sea_center_candidate_request_w2",
        "plan_source_id": "registered_child_plan_8aa919c8323dbac0",
        "certificate_source_id": "registered_certificate_5a6deef4c159d346",
    },
)

ROUTE_SOURCE_ID = "route_matrix_68ab535465b06808"


class CatalogBuildError(ValueError):
    """A stable deterministic fixture-build failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed fixture without writing files.",
    )
    return parser.parse_args()


def build_catalog_documents(
    *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[bytes, bytes]:
    """Return canonical catalog and manifest bytes from pinned sources."""

    repository_root = repository_root.resolve(strict=True)
    package_path = repository_root / PACKAGE_RELATIVE
    matrix_path = repository_root / MATRIX_RELATIVE
    _verify_hash(package_path, PINNED_PACKAGE_SHA256, "package_manifest_hash_mismatch")
    _verify_hash(matrix_path, PINNED_MATRIX_FILE_SHA256, "route_matrix_hash_mismatch")

    package = _read_json(package_path)
    matrix = _read_json(matrix_path)
    _validate_package_route_source(package)
    _validate_matrix(matrix)

    package_root = package_path.parent
    artifacts = _required_object(package.get("artifacts_sha256"), "package_artifacts_invalid")
    alternatives = {
        alternative.get("plan_id"): alternative
        for alternative in _required_list(package.get("alternatives"), "alternatives_invalid")
        if isinstance(alternative, dict)
    }

    sources: list[dict[str, Any]] = [
        {
            "retrieved_at": None,
            "source_id": ROUTE_SOURCE_ID,
            "source_type": "provider",
            "source_url": None,
        }
    ]
    places: list[dict[str, Any]] = []
    evaluated_at_values: list[str] = []

    for spec in PLACE_SPECS:
        request = _read_pinned_package_artifact(package_root, artifacts, spec["request_file"])
        plan = _read_pinned_package_artifact(package_root, artifacts, spec["plan_file"])
        certificate = _read_pinned_package_artifact(
            package_root, artifacts, spec["certificate_file"]
        )
        alternative = alternatives.get(spec["plan_id"])
        if not isinstance(alternative, dict):
            raise CatalogBuildError("registered_alternative_missing")
        place = _build_place(
            spec=spec,
            request=request,
            plan=plan,
            certificate=certificate,
            alternative=alternative,
            matrix=matrix,
        )
        places.append(place)
        evaluated_at_values.append(
            _required_text(certificate.get("evaluated_at"), "certificate_time_invalid")
        )
        sources.extend(
            (
                _artifact_source(spec["request_source_id"]),
                _artifact_source(spec["plan_source_id"]),
                _artifact_source(spec["certificate_source_id"]),
            )
        )

    catalog = {
        "catalog_id": CATALOG_ID,
        "generated_at": max(evaluated_at_values),
        "places": sorted(places, key=lambda item: item["place_id"]),
        "schema_version": CATALOG_SCHEMA,
        "sources": sorted(sources, key=lambda item: item["source_id"]),
    }
    catalog_bytes = _stable_json_bytes(catalog)
    manifest = {
        "catalog_file": CATALOG_FILENAME,
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "schema_version": MANIFEST_SCHEMA,
    }
    return catalog_bytes, _stable_json_bytes(manifest)


def write_catalog(
    output_dir: Path | None = None, *, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    """Atomically write the fixture within the configured catalog boundary."""

    repository_root = repository_root.resolve(strict=True)
    target = output_dir or repository_root / OUTPUT_RELATIVE
    target = _safe_output_directory(repository_root, target)
    catalog_bytes, manifest_bytes = build_catalog_documents(repository_root=repository_root)
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write(target / CATALOG_FILENAME, catalog_bytes)
    _atomic_write(target / MANIFEST_FILENAME, manifest_bytes)
    return target / MANIFEST_FILENAME


def check_catalog(*, repository_root: Path = REPOSITORY_ROOT) -> None:
    """Fail if the committed fixture differs from pinned-source regeneration."""

    repository_root = repository_root.resolve(strict=True)
    catalog_bytes, manifest_bytes = build_catalog_documents(repository_root=repository_root)
    target = repository_root / OUTPUT_RELATIVE
    if (target / CATALOG_FILENAME).read_bytes() != catalog_bytes:
        raise CatalogBuildError("committed_catalog_not_deterministic")
    if (target / MANIFEST_FILENAME).read_bytes() != manifest_bytes:
        raise CatalogBuildError("committed_manifest_not_deterministic")


def _build_place(
    *,
    spec: dict[str, str],
    request: dict[str, Any],
    plan: dict[str, Any],
    certificate: dict[str, Any],
    alternative: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    place_id = spec["place_id"]
    candidates = _required_list(request.get("candidate_pois"), "candidate_list_invalid")
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise CatalogBuildError("candidate_cardinality_invalid")
    candidate = candidates[0]
    if candidate.get("stop_id") != place_id:
        raise CatalogBuildError("candidate_place_mismatch")

    selected = [
        stop
        for stop in _required_list(plan.get("selected_stops"), "plan_stops_invalid")
        if isinstance(stop, dict) and stop.get("stop_id") == place_id
    ]
    if len(selected) != 1:
        raise CatalogBuildError("registered_child_place_missing")
    if plan.get("plan_id") != spec["plan_id"] or plan.get("parent_plan_id") is None:
        raise CatalogBuildError("registered_child_identity_invalid")
    if candidate.get("latitude") != selected[0].get("latitude") or candidate.get(
        "longitude"
    ) != selected[0].get("longitude"):
        raise CatalogBuildError("candidate_plan_coordinate_mismatch")

    if (
        certificate.get("certificate_id") != spec["certificate_id"]
        or certificate.get("plan_id") != spec["plan_id"]
        or certificate.get("eligible") is not True
        or certificate.get("route_validation", {}).get("publication_ready") is not True
    ):
        raise CatalogBuildError("registered_certificate_invalid")
    if (
        alternative.get("certificate_id") != spec["certificate_id"]
        or alternative.get("route_publication_ready") is not True
    ):
        raise CatalogBuildError("registered_alternative_invalid")

    route_pairs = {
        (leg.get("origin_id"), leg.get("destination_id"))
        for leg in _required_list(alternative.get("route_legs"), "alternative_routes_invalid")
        if isinstance(leg, dict) and leg.get("day") == 7
    }
    required_pairs = {
        (DAY_7_ORIGIN, place_id),
        (place_id, DAY_7_DESTINATION),
    }
    if route_pairs != required_pairs:
        raise CatalogBuildError("candidate_adjacent_route_contract_invalid")

    cells = {
        (cell.get("origin_id"), cell.get("destination_id")): cell
        for cell in _required_list(matrix.get("cells"), "route_cells_invalid")
        if isinstance(cell, dict)
    }
    inbound = cells.get((DAY_7_ORIGIN, place_id))
    outbound = cells.get((place_id, DAY_7_DESTINATION))
    if not isinstance(inbound, dict) or not isinstance(outbound, dict):
        raise CatalogBuildError("candidate_adjacent_route_evidence_missing")
    access_coordinate, evidence_refs = _validated_access_endpoint(inbound, outbound)

    display_coordinate = {
        "latitude": _required_number(candidate.get("latitude"), "candidate_latitude_invalid"),
        "longitude": _required_number(
            candidate.get("longitude"), "candidate_longitude_invalid"
        ),
    }
    if display_coordinate == access_coordinate:
        raise CatalogBuildError("display_access_coordinate_not_separated")

    return {
        "access_points": [
            {
                "access_confidence": "provider_access_point",
                "access_point_id": f"{place_id}_road_endpoint",
                "access_type": "road_route_endpoint",
                "coordinate": access_coordinate,
                "evidence_refs": list(evidence_refs),
                "road_validated": True,
                "source_ref": ROUTE_SOURCE_ID,
            }
        ],
        "display_coordinate": display_coordinate,
        "name": _required_text(candidate.get("name"), "candidate_name_invalid"),
        "place_categories": ["registered_candidate"],
        "place_id": place_id,
        "source_refs": [
            spec["request_source_id"],
            spec["plan_source_id"],
            spec["certificate_source_id"],
            ROUTE_SOURCE_ID,
        ],
    }


def _validated_access_endpoint(
    inbound: dict[str, Any], outbound: dict[str, Any]
) -> tuple[dict[str, float], tuple[str, str]]:
    for cell in (inbound, outbound):
        if (
            cell.get("road_validated") is not True
            or cell.get("fallback_used") is not False
            or cell.get("provider") != "cached_osrm"
            or cell.get("routing_status") != "osrm_live"
            or cell.get("geometry_source") != "cached_osrm_route_geometry"
        ):
            raise CatalogBuildError("candidate_route_not_road_validated")
    inbound_geometry = _required_list(inbound.get("geometry"), "route_geometry_invalid")
    outbound_geometry = _required_list(outbound.get("geometry"), "route_geometry_invalid")
    if not inbound_geometry or not outbound_geometry:
        raise CatalogBuildError("route_geometry_empty")
    inbound_endpoint = _geometry_coordinate(inbound_geometry[-1])
    outbound_endpoint = _geometry_coordinate(outbound_geometry[0])
    if inbound_endpoint != outbound_endpoint:
        raise CatalogBuildError("adjacent_route_access_endpoint_mismatch")
    query_hashes = (
        _required_text(inbound.get("query_hash"), "route_query_hash_invalid"),
        _required_text(outbound.get("query_hash"), "route_query_hash_invalid"),
    )
    if query_hashes[0] == query_hashes[1]:
        raise CatalogBuildError("route_query_hash_duplicate")
    return inbound_endpoint, query_hashes


def _geometry_coordinate(value: Any) -> dict[str, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CatalogBuildError("route_geometry_coordinate_invalid")
    # RouteMatrix geometry uses [latitude, longitude].
    return {
        "latitude": _required_number(value[0], "route_latitude_invalid"),
        "longitude": _required_number(value[1], "route_longitude_invalid"),
    }


def _validate_package_route_source(package: dict[str, Any]) -> None:
    source = _required_object(package.get("route_evidence"), "package_route_source_invalid")
    if (
        source.get("matrix_id") != PINNED_MATRIX_ID
        or source.get("source_content_sha256") != PINNED_MATRIX_CONTENT_SHA256
        or source.get("matrix_file_sha256") != PINNED_MATRIX_FILE_SHA256
        or source.get("relative_path") != MATRIX_RELATIVE.as_posix()
    ):
        raise CatalogBuildError("package_route_source_mismatch")


def _validate_matrix(matrix: dict[str, Any]) -> None:
    if (
        matrix.get("schema_version") != "route-matrix-v1"
        or matrix.get("matrix_id") != PINNED_MATRIX_ID
        or matrix.get("source_content_sha256") != PINNED_MATRIX_CONTENT_SHA256
    ):
        raise CatalogBuildError("route_matrix_identity_invalid")


def _artifact_source(source_id: str) -> dict[str, Any]:
    return {
        "retrieved_at": None,
        "source_id": source_id,
        "source_type": "curated_dataset",
        "source_url": None,
    }


def _read_pinned_package_artifact(
    package_root: Path, artifacts: dict[str, Any], relative_name: str
) -> dict[str, Any]:
    expected = artifacts.get(relative_name)
    if not isinstance(expected, str) or len(expected) != 64:
        raise CatalogBuildError("package_artifact_hash_missing")
    path = (package_root / relative_name).resolve(strict=True)
    try:
        path.relative_to(package_root.resolve(strict=True))
    except ValueError as exc:
        raise CatalogBuildError("package_artifact_path_escape") from exc
    _verify_hash(path, expected, "package_artifact_hash_mismatch")
    return _read_json(path)


def _safe_output_directory(repository_root: Path, output_dir: Path) -> Path:
    allowed_root = (repository_root / "configs/product_poi_catalogs").resolve()
    target = output_dir.resolve()
    try:
        relative = target.relative_to(allowed_root)
    except ValueError as exc:
        raise CatalogBuildError("output_outside_catalog_root") from exc
    if not relative.parts or target == allowed_root or any(part == ".." for part in relative.parts):
        raise CatalogBuildError("output_directory_invalid")
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise CatalogBuildError("output_directory_unsafe")
    return target


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_hash(path: Path, expected: str, code: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CatalogBuildError(f"{code}_unreadable") from exc
    if actual != expected:
        raise CatalogBuildError(code)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogBuildError("source_json_invalid") from exc
    if not isinstance(value, dict):
        raise CatalogBuildError("source_object_required")
    return value


def _required_object(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogBuildError(code)
    return value


def _required_list(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list):
        raise CatalogBuildError(code)
    return value


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CatalogBuildError(code)
    return value


def _required_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogBuildError(code)
    return float(value)


def _stable_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.check:
        if args.output is not None:
            raise CatalogBuildError("check_output_conflict")
        check_catalog(repository_root=args.repository_root)
    else:
        write_catalog(args.output, repository_root=args.repository_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
