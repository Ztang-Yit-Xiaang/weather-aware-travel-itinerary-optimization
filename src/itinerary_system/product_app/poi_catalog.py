"""Immutable, source-backed POI catalog contracts for the product application.

The catalog is deliberately limited to place discovery.  It does not claim that
a place is route-near, feasible, or recommended; those decisions require the
runtime router and independent evaluator.  Display coordinates and routing
access points are separate so a map label is never silently treated as a
road-validated entrance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MANIFEST_SCHEMA_VERSION = "product-poi-catalog-manifest-v1"
CATALOG_SCHEMA_VERSION = "product-poi-catalog-v1"
MAX_SEARCH_RESULTS = 50
MAX_QUERY_CHARACTERS = 120

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SOURCE_TYPES = frozenset(
    {"official", "informational", "curated_dataset", "provider", "user_supplied"}
)
_ACCESS_CONFIDENCE = frozenset(
    {"verified_entrance", "provider_access_point", "road_snap_only", "uncertain"}
)


class POICatalogError(ValueError):
    """Stable, sanitized catalog validation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CoordinateV1:
    longitude: float
    latitude: float


@dataclass(frozen=True, slots=True)
class POISourceV1:
    source_id: str
    source_type: str
    source_url: str | None
    retrieved_at: str | None


@dataclass(frozen=True, slots=True)
class RouteAccessPointV1:
    """A possible routing access point, never the place's display coordinate."""

    access_point_id: str
    access_type: str
    coordinate: CoordinateV1
    source_ref: str
    road_validated: bool
    access_confidence: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlaceEntityV1:
    """Real-world place identity, independent of any itinerary stop role."""

    place_id: str
    name: str
    place_categories: tuple[str, ...]
    display_coordinate: CoordinateV1
    access_points: tuple[RouteAccessPointV1, ...]
    description: str | None
    official_url: str | None
    informational_urls: tuple[str, ...]
    source_refs: tuple[str, ...]
    source_freshness: str | None
    opening_hours_evidence_ref: str | None
    recommended_visit_minutes: int | None
    weather_suitability: float | None
    static_utility: float | None

    @property
    def has_road_validated_access(self) -> bool:
        return any(point.road_validated for point in self.access_points)


@dataclass(frozen=True, slots=True)
class ProductPOICatalogManifestV1:
    schema_version: str
    catalog_file: str
    catalog_sha256: str


@dataclass(frozen=True, slots=True)
class ProductPOICatalogV1:
    """An immutable POI catalog loaded from a hash-pinned manifest."""

    schema_version: str
    catalog_id: str
    generated_at: str
    sources: tuple[POISourceV1, ...]
    places: tuple[PlaceEntityV1, ...]
    manifest_sha256: str

    @classmethod
    def load(cls, manifest_path: str | Path) -> ProductPOICatalogV1:
        manifest_file = Path(manifest_path)
        manifest_data = _read_json(manifest_file, "manifest")
        manifest = _parse_manifest(manifest_data)
        catalog_path = _resolve_contained_file(manifest_file.parent, manifest.catalog_file)
        try:
            payload = catalog_path.read_bytes()
        except OSError as exc:
            raise POICatalogError("catalog_file_unreadable") from exc
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != manifest.catalog_sha256.lower():
            raise POICatalogError("catalog_hash_mismatch")
        catalog_data = _decode_json(payload, "catalog")
        return _parse_catalog(catalog_data, manifest_sha256=actual_hash)

    def get_place(self, place_id: str) -> PlaceEntityV1 | None:
        """Return an exact place without making a feasibility claim."""

        return next((place for place in self.places if place.place_id == place_id), None)

    def search(
        self,
        query: str | None = None,
        *,
        categories: Iterable[str] = (),
        source_types: Iterable[str] = (),
        bounds: tuple[float, float, float, float] | None = None,
        require_road_access: bool | None = None,
        limit: int = 20,
    ) -> tuple[PlaceEntityV1, ...]:
        """Return a deterministic, bounded discovery set.

        Results are lexical/geographic catalog matches only.  Their order is
        suitable as deterministic input to later route-aware ranking, but it is
        not itself a recommendation or feasibility ranking.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise POICatalogError("search_limit_invalid")
        normalized_query = _normalize_query(query)
        category_filter = _normalize_filter(categories, "search_category_invalid")
        source_filter = _normalize_filter(source_types, "search_source_type_invalid")
        unknown_source_types = source_filter - _SOURCE_TYPES
        if unknown_source_types:
            raise POICatalogError("search_source_type_unknown")
        parsed_bounds = _validate_bounds(bounds)
        sources = {source.source_id: source for source in self.sources}

        matched: list[tuple[tuple[int, str, str], PlaceEntityV1]] = []
        for place in self.places:
            place_categories = {category.casefold() for category in place.place_categories}
            if category_filter and not category_filter.intersection(place_categories):
                continue
            if source_filter:
                place_source_types = {
                    sources[source_ref].source_type
                    for source_ref in place.source_refs
                    if source_ref in sources
                }
                if not source_filter.intersection(place_source_types):
                    continue
            if require_road_access is not None and (
                place.has_road_validated_access is not require_road_access
            ):
                continue
            if parsed_bounds is not None and not _coordinate_in_bounds(
                place.display_coordinate, parsed_bounds
            ):
                continue
            match_rank = _lexical_match_rank(place, normalized_query)
            if match_rank is None:
                continue
            matched.append(
                ((match_rank, place.name.casefold(), place.place_id.casefold()), place)
            )
        matched.sort(key=lambda row: row[0])
        return tuple(place for _, place in matched[:limit])


def load_product_poi_catalog(manifest_path: str | Path) -> ProductPOICatalogV1:
    """Load and validate a catalog through its immutable manifest."""

    return ProductPOICatalogV1.load(manifest_path)


def _read_json(path: Path, kind: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise POICatalogError(f"{kind}_file_unreadable") from exc
    return _decode_json(payload, kind)


def _decode_json(payload: bytes, kind: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise POICatalogError(f"{kind}_encoding_invalid") from exc
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except POICatalogError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise POICatalogError(f"{kind}_json_invalid") from exc
    if not isinstance(value, dict):
        raise POICatalogError(f"{kind}_object_required")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise POICatalogError("json_duplicate_key")
        result[key] = value
    return result


def _parse_manifest(data: Mapping[str, Any]) -> ProductPOICatalogManifestV1:
    _require_fields(
        data,
        required={"schema_version", "catalog_file", "catalog_sha256"},
        allowed={"schema_version", "catalog_file", "catalog_sha256"},
        kind="manifest",
    )
    if data["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise POICatalogError("manifest_schema_unsupported")
    catalog_file = _required_text(data["catalog_file"], "manifest_catalog_file_invalid", 256)
    catalog_hash = _required_text(data["catalog_sha256"], "manifest_hash_invalid", 64)
    if not _SHA256.fullmatch(catalog_hash):
        raise POICatalogError("manifest_hash_invalid")
    return ProductPOICatalogManifestV1(
        schema_version=MANIFEST_SCHEMA_VERSION,
        catalog_file=catalog_file,
        catalog_sha256=catalog_hash.lower(),
    )


def _resolve_contained_file(root: Path, relative_name: str) -> Path:
    candidate_name = Path(relative_name)
    if candidate_name.is_absolute() or any(part == ".." for part in candidate_name.parts):
        raise POICatalogError("catalog_path_not_relative")
    if candidate_name.name != relative_name and len(candidate_name.parts) == 1:
        raise POICatalogError("catalog_path_invalid")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = (resolved_root / candidate_name).resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except FileNotFoundError as exc:
        raise POICatalogError("catalog_file_unreadable") from exc
    except (OSError, ValueError) as exc:
        raise POICatalogError("catalog_path_escape") from exc
    if not resolved_candidate.is_file():
        raise POICatalogError("catalog_file_unreadable")
    return resolved_candidate


def _parse_catalog(data: Mapping[str, Any], *, manifest_sha256: str) -> ProductPOICatalogV1:
    fields = {"schema_version", "catalog_id", "generated_at", "sources", "places"}
    _require_fields(data, required=fields, allowed=fields, kind="catalog")
    if data["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise POICatalogError("catalog_schema_unsupported")
    catalog_id = _identifier(data["catalog_id"], "catalog_id_invalid")
    generated_at = _timestamp(data["generated_at"], "catalog_generated_at_invalid", nullable=False)
    sources_raw = _list(data["sources"], "catalog_sources_invalid")
    places_raw = _list(data["places"], "catalog_places_invalid")

    sources = tuple(_parse_source(item) for item in sources_raw)
    _reject_duplicate_values((source.source_id for source in sources), "source_id_duplicate")
    source_ids = {source.source_id for source in sources}
    places = tuple(_parse_place(item, source_ids=source_ids) for item in places_raw)
    _reject_duplicate_values((place.place_id for place in places), "place_id_duplicate")
    _reject_duplicate_values(
        (
            access.access_point_id
            for place in places
            for access in place.access_points
        ),
        "access_point_id_duplicate",
    )
    return ProductPOICatalogV1(
        schema_version=CATALOG_SCHEMA_VERSION,
        catalog_id=catalog_id,
        generated_at=generated_at,
        sources=sources,
        places=places,
        manifest_sha256=manifest_sha256,
    )


def _parse_source(value: Any) -> POISourceV1:
    data = _object(value, "source_object_required")
    fields = {"source_id", "source_type", "source_url", "retrieved_at"}
    _require_fields(data, required=fields, allowed=fields, kind="source")
    source_type = _required_text(data["source_type"], "source_type_invalid", 32)
    if source_type not in _SOURCE_TYPES:
        raise POICatalogError("source_type_unknown")
    return POISourceV1(
        source_id=_identifier(data["source_id"], "source_id_invalid"),
        source_type=source_type,
        source_url=_safe_url(data["source_url"], "source_url_invalid"),
        retrieved_at=_timestamp(data["retrieved_at"], "source_retrieved_at_invalid"),
    )


def _parse_place(value: Any, *, source_ids: set[str]) -> PlaceEntityV1:
    data = _object(value, "place_object_required")
    required = {
        "place_id",
        "name",
        "place_categories",
        "display_coordinate",
        "access_points",
        "source_refs",
    }
    optional = {
        "description",
        "official_url",
        "informational_urls",
        "source_freshness",
        "opening_hours_evidence_ref",
        "recommended_visit_minutes",
        "weather_suitability",
        "static_utility",
    }
    _require_fields(data, required=required, allowed=required | optional, kind="place")
    categories = _text_tuple(data["place_categories"], "place_categories_invalid", maximum=32)
    if not categories:
        raise POICatalogError("place_categories_empty")
    source_refs = _identifier_tuple(data["source_refs"], "place_source_refs_invalid", maximum=32)
    if not source_refs:
        raise POICatalogError("place_source_refs_empty")
    if any(source_ref not in source_ids for source_ref in source_refs):
        raise POICatalogError("place_source_ref_unknown")
    access_points = tuple(
        _parse_access_point(item, source_ids=source_ids)
        for item in _list(data["access_points"], "access_points_invalid")
    )
    official_url = _safe_url(data.get("official_url"), "official_url_invalid")
    informational_urls = _url_tuple(
        data.get("informational_urls", []), "informational_url_invalid"
    )
    if official_url is not None and official_url in informational_urls:
        raise POICatalogError("place_url_role_duplicate")
    return PlaceEntityV1(
        place_id=_identifier(data["place_id"], "place_id_invalid"),
        name=_required_text(data["name"], "place_name_invalid", 120),
        place_categories=categories,
        display_coordinate=_coordinate(data["display_coordinate"]),
        access_points=access_points,
        description=_optional_text(data.get("description"), "place_description_invalid", 2_000),
        official_url=official_url,
        informational_urls=informational_urls,
        source_refs=source_refs,
        source_freshness=_timestamp(
            data.get("source_freshness"), "place_source_freshness_invalid"
        ),
        opening_hours_evidence_ref=_optional_identifier(
            data.get("opening_hours_evidence_ref"), "opening_hours_evidence_ref_invalid"
        ),
        recommended_visit_minutes=_optional_positive_int(
            data.get("recommended_visit_minutes"), "recommended_visit_minutes_invalid"
        ),
        weather_suitability=_optional_finite_number(
            data.get("weather_suitability"), "weather_suitability_invalid"
        ),
        static_utility=_optional_finite_number(
            data.get("static_utility"), "static_utility_invalid"
        ),
    )


def _parse_access_point(value: Any, *, source_ids: set[str]) -> RouteAccessPointV1:
    data = _object(value, "access_point_object_required")
    fields = {
        "access_point_id",
        "access_type",
        "coordinate",
        "source_ref",
        "road_validated",
        "access_confidence",
        "evidence_refs",
    }
    _require_fields(data, required=fields, allowed=fields, kind="access_point")
    source_ref = _identifier(data["source_ref"], "access_point_source_ref_invalid")
    if source_ref not in source_ids:
        raise POICatalogError("access_point_source_ref_unknown")
    road_validated = data["road_validated"]
    if not isinstance(road_validated, bool):
        raise POICatalogError("access_point_road_validated_invalid")
    confidence = _required_text(
        data["access_confidence"], "access_point_confidence_invalid", 32
    )
    if confidence not in _ACCESS_CONFIDENCE:
        raise POICatalogError("access_point_confidence_unknown")
    evidence_refs = _identifier_tuple(
        data["evidence_refs"], "access_point_evidence_refs_invalid", maximum=16
    )
    if road_validated and not evidence_refs:
        raise POICatalogError("road_validated_access_missing_evidence")
    if road_validated and confidence in {"road_snap_only", "uncertain"}:
        raise POICatalogError("road_validated_access_confidence_invalid")
    return RouteAccessPointV1(
        access_point_id=_identifier(data["access_point_id"], "access_point_id_invalid"),
        access_type=_required_text(data["access_type"], "access_point_type_invalid", 48),
        coordinate=_coordinate(data["coordinate"]),
        source_ref=source_ref,
        road_validated=road_validated,
        access_confidence=confidence,
        evidence_refs=evidence_refs,
    )


def _coordinate(value: Any) -> CoordinateV1:
    data = _object(value, "coordinate_object_required")
    fields = {"longitude", "latitude"}
    _require_fields(data, required=fields, allowed=fields, kind="coordinate")
    longitude = _finite_number(data["longitude"], "longitude_invalid")
    latitude = _finite_number(data["latitude"], "latitude_invalid")
    if not -180 <= longitude <= 180:
        raise POICatalogError("longitude_out_of_range")
    if not -90 <= latitude <= 90:
        raise POICatalogError("latitude_out_of_range")
    return CoordinateV1(longitude=longitude, latitude=latitude)


def _safe_url(value: Any, code: str) -> str | None:
    if value is None:
        return None
    url = _required_text(value, code, 2_048)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise POICatalogError(code) from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character in url for character in "\r\n\t")
        or port is not None and not 1 <= port <= 65_535
    ):
        raise POICatalogError(code)
    return url


def _timestamp(value: Any, code: str, *, nullable: bool = True) -> str | None:
    if value is None and nullable:
        return None
    text = _required_text(value, code, 64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise POICatalogError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise POICatalogError(code)
    return text


def _identifier(value: Any, code: str) -> str:
    text = _required_text(value, code, 128)
    if not _IDENTIFIER.fullmatch(text):
        raise POICatalogError(code)
    return text


def _optional_identifier(value: Any, code: str) -> str | None:
    return None if value is None else _identifier(value, code)


def _required_text(value: Any, code: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise POICatalogError(code)
    return value


def _optional_text(value: Any, code: str, maximum: int) -> str | None:
    return None if value is None else _required_text(value, code, maximum)


def _finite_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise POICatalogError(code)
    converted = float(value)
    if not math.isfinite(converted):
        raise POICatalogError(code)
    return converted


def _optional_finite_number(value: Any, code: str) -> float | None:
    return None if value is None else _finite_number(value, code)


def _optional_positive_int(value: Any, code: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_440:
        raise POICatalogError(code)
    return value


def _object(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise POICatalogError(code)
    return value


def _list(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list):
        raise POICatalogError(code)
    return value


def _text_tuple(value: Any, code: str, *, maximum: int) -> tuple[str, ...]:
    items = _list(value, code)
    if len(items) > maximum:
        raise POICatalogError(code)
    values = tuple(_required_text(item, code, 64) for item in items)
    _reject_duplicate_values((item.casefold() for item in values), code)
    return values


def _identifier_tuple(value: Any, code: str, *, maximum: int) -> tuple[str, ...]:
    items = _list(value, code)
    if len(items) > maximum:
        raise POICatalogError(code)
    values = tuple(_identifier(item, code) for item in items)
    _reject_duplicate_values(values, code)
    return values


def _url_tuple(value: Any, code: str) -> tuple[str, ...]:
    items = _list(value, code)
    if len(items) > 16:
        raise POICatalogError(code)
    values = tuple(_safe_url(item, code) for item in items)
    if any(item is None for item in values):
        raise POICatalogError(code)
    result = tuple(item for item in values if item is not None)
    _reject_duplicate_values(result, code)
    return result


def _require_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    kind: str,
) -> None:
    if missing := required - value.keys():
        del missing
        raise POICatalogError(f"{kind}_field_missing")
    if unknown := value.keys() - allowed:
        del unknown
        raise POICatalogError(f"{kind}_field_unknown")


def _reject_duplicate_values(values: Iterable[str], code: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise POICatalogError(code)
        seen.add(value)


def _normalize_query(query: str | None) -> str:
    if query is None:
        return ""
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARACTERS:
        raise POICatalogError("search_query_invalid")
    return " ".join(query.casefold().split())


def _normalize_filter(values: Iterable[str], code: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise POICatalogError(code)
    try:
        normalized = {_required_text(value, code, 64).casefold() for value in values}
    except TypeError as exc:
        raise POICatalogError(code) from exc
    if len(normalized) > 32:
        raise POICatalogError(code)
    return normalized


def _validate_bounds(
    bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if bounds is None:
        return None
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        raise POICatalogError("search_bounds_invalid")
    west, south, east, north = (
        _finite_number(value, "search_bounds_invalid") for value in bounds
    )
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise POICatalogError("search_bounds_invalid")
    return west, south, east, north


def _coordinate_in_bounds(
    coordinate: CoordinateV1, bounds: tuple[float, float, float, float]
) -> bool:
    west, south, east, north = bounds
    return west <= coordinate.longitude <= east and south <= coordinate.latitude <= north


def _lexical_match_rank(place: PlaceEntityV1, normalized_query: str) -> int | None:
    if not normalized_query:
        return 4
    name = " ".join(place.name.casefold().split())
    categories = tuple(" ".join(item.casefold().split()) for item in place.place_categories)
    if name == normalized_query:
        return 0
    if name.startswith(normalized_query):
        return 1
    if normalized_query in name:
        return 2
    if any(normalized_query in category for category in categories):
        return 3
    return None


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_SEARCH_RESULTS",
    "CoordinateV1",
    "PlaceEntityV1",
    "POICatalogError",
    "POISourceV1",
    "ProductPOICatalogManifestV1",
    "ProductPOICatalogV1",
    "RouteAccessPointV1",
    "load_product_poi_catalog",
]
