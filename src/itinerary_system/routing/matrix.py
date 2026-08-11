"""Route matrix contracts for solver-facing travel evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from .cache import route_anchor_key
from .models import RouteLegResult, RouteResult


class RouteMatrixError(ValueError):
    """Base class for route matrix validation failures."""


class RouteMatrixMissing(RouteMatrixError):
    """Raised when strict routing requires a matrix but none was supplied."""


class RouteMatrixCellMissing(RouteMatrixError):
    """Raised when a required directed matrix cell is absent."""

    def __init__(self, origin_id: str, destination_id: str) -> None:
        super().__init__(f"missing route matrix cell: {origin_id!r} -> {destination_id!r}")
        self.origin_id = origin_id
        self.destination_id = destination_id


class RouteMatrixNotPublicationEligible(RouteMatrixError):
    """Raised when strict routing sees fallback or non-road-validated evidence."""


@dataclass(frozen=True)
class RouteMatrixValidationReport:
    """Publication-readiness summary for a route matrix."""

    matrix_id: str
    context_snapshot_id: str
    required_leg_count: int
    present_leg_count: int
    road_validated_leg_count: int
    fallback_leg_count: int
    invalid_value_count: int
    missing_leg_count: int
    publication_ready: bool
    source_bundle_id: str = ""
    source_content_sha256: str = ""
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "context_snapshot_id": self.context_snapshot_id,
            "required_leg_count": self.required_leg_count,
            "present_leg_count": self.present_leg_count,
            "road_validated_leg_count": self.road_validated_leg_count,
            "fallback_leg_count": self.fallback_leg_count,
            "invalid_value_count": self.invalid_value_count,
            "missing_leg_count": self.missing_leg_count,
            "publication_ready": self.publication_ready,
            "source_bundle_id": self.source_bundle_id,
            "source_content_sha256": self.source_content_sha256,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "validated", "road_validated"}


def _parse_geometry(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, list | tuple):
        raw_points = value
    else:
        text = _safe_str(value)
        if not text:
            return ()
        try:
            raw_points = json.loads(text)
        except Exception:
            return ()
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        lat = _safe_float(point[0])
        lon = _safe_float(point[1])
        if lat is not None and lon is not None:
            points.append((lat, lon))
    return tuple(points)


def _normalize_entity_id(value: Any) -> str:
    return route_anchor_key(value)


def _positive_or_none(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field_name} must be positive when present")
    return float(value)


def _nonnegative_or_none(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{field_name} must be nonnegative when present")
    return float(value)


@dataclass(frozen=True)
class RouteMatrixCell:
    """One directed solver travel cell with route provenance."""

    origin_id: str
    destination_id: str
    distance_m: float | None
    duration_s: float | None
    route_leg_id: str = ""
    road_validated: bool = False
    fallback_used: bool = False
    provider: str = "unknown"
    query_hash: str = ""
    context_snapshot_id: str = ""
    geometry: tuple[tuple[float, float], ...] = ()
    routing_profile: str = "driving"
    routing_status: str = "unknown"
    geometry_source: str = "unknown"
    distance_source: str = "unknown"
    duration_source: str = "unknown"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        origin_id = _normalize_entity_id(self.origin_id)
        destination_id = _normalize_entity_id(self.destination_id)
        if not origin_id or not destination_id:
            raise ValueError("route matrix cell origin_id and destination_id are required")
        object.__setattr__(self, "origin_id", origin_id)
        object.__setattr__(self, "destination_id", destination_id)
        value_validator = _nonnegative_or_none if origin_id == destination_id else _positive_or_none
        object.__setattr__(self, "distance_m", value_validator(self.distance_m, "distance_m"))
        object.__setattr__(self, "duration_s", value_validator(self.duration_s, "duration_s"))
        if self.fallback_used and self.road_validated:
            raise ValueError("fallback route matrix cells cannot be road validated")
        if not self.route_leg_id:
            object.__setattr__(self, "route_leg_id", f"leg_{origin_id}_{destination_id}")
        if not self.query_hash:
            object.__setattr__(
                self,
                "query_hash",
                _stable_hash(
                    {
                        "origin_id": origin_id,
                        "destination_id": destination_id,
                        "distance_m": self.distance_m,
                        "duration_s": self.duration_s,
                        "provider": self.provider,
                        "context_snapshot_id": self.context_snapshot_id,
                    }
                ),
            )

    def require_duration_s(self) -> float:
        if self.duration_s is None:
            raise RouteMatrixNotPublicationEligible(
                f"route matrix cell lacks duration_s: {self.origin_id!r} -> {self.destination_id!r}"
            )
        return float(self.duration_s)

    def require_distance_m(self) -> float:
        if self.distance_m is None:
            raise RouteMatrixNotPublicationEligible(
                f"route matrix cell lacks distance_m: {self.origin_id!r} -> {self.destination_id!r}"
            )
        return float(self.distance_m)

    def require_publication_eligible(self) -> None:
        if not self.road_validated or self.fallback_used:
            raise RouteMatrixNotPublicationEligible(
                f"route matrix cell is not publication eligible: {self.origin_id!r} -> {self.destination_id!r}"
            )
        self.require_duration_s()
        self.require_distance_m()

    def to_leg_result(self) -> RouteLegResult:
        return RouteLegResult(
            origin_id=self.origin_id,
            destination_id=self.destination_id,
            geometry=self.geometry,
            distance_m=self.distance_m,
            duration_s=self.duration_s,
            routing_status=self.routing_status,
            provider=self.provider,
            routing_profile=self.routing_profile,
            geometry_source=self.geometry_source,
            distance_source=self.distance_source,
            duration_source=self.duration_source,
            road_validated=self.road_validated,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            query_hash=self.query_hash,
        )


@dataclass(frozen=True)
class RouteMatrix:
    """Directed route evidence indexed by stable entity IDs."""

    matrix_id: str
    context_snapshot_id: str
    entity_ids: tuple[str, ...]
    cells: Mapping[tuple[str, str], RouteMatrixCell] = field(default_factory=dict)
    source_bundle_id: str = ""
    source_content_sha256: str = ""

    def __post_init__(self) -> None:
        entity_ids = tuple(_normalize_entity_id(value) for value in self.entity_ids if _normalize_entity_id(value))
        normalized_cells: dict[tuple[str, str], RouteMatrixCell] = {}
        for key, cell in self.cells.items():
            origin_id, destination_id = (_normalize_entity_id(key[0]), _normalize_entity_id(key[1]))
            if origin_id != cell.origin_id or destination_id != cell.destination_id:
                cell = replace(cell, origin_id=origin_id, destination_id=destination_id)
            normalized_cells[(origin_id, destination_id)] = cell
            entity_ids += tuple(value for value in (origin_id, destination_id) if value not in entity_ids)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "cells", normalized_cells)

    @property
    def empty(self) -> bool:
        return not bool(self.cells)

    def cell(self, origin_id: str, destination_id: str) -> RouteMatrixCell:
        origin = _normalize_entity_id(origin_id)
        destination = _normalize_entity_id(destination_id)
        if origin and origin == destination:
            return RouteMatrixCell(
                origin_id=origin,
                destination_id=destination,
                distance_m=0.0,
                duration_s=0.0,
                road_validated=True,
                fallback_used=False,
                provider="deterministic_identity",
                context_snapshot_id=self.context_snapshot_id,
                routing_status="identity_zero_distance",
                geometry_source="identity",
                distance_source="identity",
                duration_source="identity",
            )
        try:
            return self.cells[(origin, destination)]
        except KeyError as exc:
            raise RouteMatrixCellMissing(origin, destination) from exc

    def duration_minutes(self, origin_id: str, destination_id: str, *, strict: bool = False) -> float:
        cell = self.cell(origin_id, destination_id)
        if strict:
            cell.require_publication_eligible()
        return cell.require_duration_s() / 60.0

    def distance_meters(self, origin_id: str, destination_id: str, *, strict: bool = False) -> float:
        cell = self.cell(origin_id, destination_id)
        if strict:
            cell.require_publication_eligible()
        return cell.require_distance_m()

    def leg(self, origin_id: str, destination_id: str, *, strict: bool = False) -> RouteLegResult:
        cell = self.cell(origin_id, destination_id)
        if strict:
            cell.require_publication_eligible()
        return cell.to_leg_result()

    def require_road_validated(self, sequence: Sequence[str] | None = None) -> None:
        cells = self._cells_for_sequence(sequence) if sequence is not None else tuple(self.cells.values())
        if not cells:
            raise RouteMatrixMissing("route matrix contains no cells")
        for cell in cells:
            cell.require_publication_eligible()

    def _cells_for_sequence(self, sequence: Sequence[str]) -> tuple[RouteMatrixCell, ...]:
        normalized = tuple(_normalize_entity_id(value) for value in sequence)
        return tuple(self.cell(left, right) for left, right in zip(normalized[:-1], normalized[1:], strict=False))


def required_pairs_from_sequences(sequences: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    """Return ordered, de-duplicated directed pairs required by route sequences."""

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sequence in sequences:
        normalized = tuple(_normalize_entity_id(value) for value in sequence if _normalize_entity_id(value))
        for pair in zip(normalized[:-1], normalized[1:], strict=False):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return tuple(pairs)


def validate_route_matrix(
    matrix: RouteMatrix,
    *,
    required_sequences: Sequence[Sequence[str]] = (),
    require_publication_ready: bool = False,
) -> RouteMatrixValidationReport:
    """Validate route matrix cells and optional required solver sequences."""

    required_pairs = required_pairs_from_sequences(required_sequences)
    pairs_to_check = required_pairs or tuple(matrix.cells)
    errors: list[str] = []
    warnings: list[str] = []
    present = 0
    road_validated = 0
    fallback = 0
    invalid = 0
    missing = 0
    for origin_id, destination_id in pairs_to_check:
        try:
            cell = matrix.cell(origin_id, destination_id)
        except RouteMatrixCellMissing:
            missing += 1
            errors.append(f"missing_cell:{origin_id}->{destination_id}")
            continue
        present += 1
        if cell.road_validated:
            road_validated += 1
        if cell.fallback_used:
            fallback += 1
            errors.append(f"fallback_cell:{origin_id}->{destination_id}")
        if not cell.road_validated:
            errors.append(f"not_road_validated:{origin_id}->{destination_id}")
        if cell.distance_m is None or cell.duration_s is None or cell.distance_m <= 0 or cell.duration_s <= 0:
            invalid += 1
            errors.append(f"invalid_distance_or_duration:{origin_id}->{destination_id}")
    if not pairs_to_check:
        warnings.append("route_matrix_has_no_checked_cells")
    publication_ready = bool(pairs_to_check) and missing == 0 and fallback == 0 and invalid == 0 and road_validated == present
    if require_publication_ready and not publication_ready:
        errors.append("route_matrix_not_publication_ready")
    return RouteMatrixValidationReport(
        matrix_id=matrix.matrix_id,
        context_snapshot_id=matrix.context_snapshot_id,
        required_leg_count=int(len(pairs_to_check)),
        present_leg_count=int(present),
        road_validated_leg_count=int(road_validated),
        fallback_leg_count=int(fallback),
        invalid_value_count=int(invalid),
        missing_leg_count=int(missing),
        publication_ready=publication_ready,
        source_bundle_id=matrix.source_bundle_id,
        source_content_sha256=matrix.source_content_sha256,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def route_matrix_to_frame(matrix: RouteMatrix) -> pd.DataFrame:
    """Serialize route matrix cells to a DataFrame."""

    rows = []
    for origin_id, destination_id in sorted(matrix.cells):
        cell = matrix.cells[(origin_id, destination_id)]
        rows.append(
            {
                "matrix_id": matrix.matrix_id,
                "source_bundle_id": matrix.source_bundle_id,
                "source_content_sha256": matrix.source_content_sha256,
                "context_snapshot_id": cell.context_snapshot_id or matrix.context_snapshot_id,
                "origin_id": cell.origin_id,
                "destination_id": cell.destination_id,
                "route_leg_id": cell.route_leg_id,
                "geometry": json.dumps(cell.geometry, separators=(",", ":")),
                "distance_m": cell.distance_m,
                "duration_s": cell.duration_s,
                "provider": cell.provider,
                "routing_profile": cell.routing_profile,
                "routing_status": cell.routing_status,
                "geometry_source": cell.geometry_source,
                "distance_source": cell.distance_source,
                "duration_source": cell.duration_source,
                "road_validated": cell.road_validated,
                "fallback_used": cell.fallback_used,
                "fallback_reason": cell.fallback_reason or "",
                "query_hash": cell.query_hash,
            }
        )
    return pd.DataFrame(rows)


def write_validated_route_matrix_artifacts(
    matrix: RouteMatrix,
    output_dir: str | Path,
    *,
    required_sequences: Sequence[Sequence[str]] = (),
    require_publication_ready: bool = False,
    matrix_filename: str = "production_validated_route_matrix.csv",
    report_filename: str = "production_validated_route_matrix_report.json",
) -> RouteMatrixValidationReport:
    """Write matrix CSV and validation report artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = validate_route_matrix(
        matrix,
        required_sequences=required_sequences,
        require_publication_ready=require_publication_ready,
    )
    route_matrix_to_frame(matrix).to_csv(output_path / matrix_filename, index=False)
    (output_path / report_filename).write_text(json.dumps(report.to_record(), indent=2), encoding="utf-8")
    if require_publication_ready and not report.publication_ready:
        raise RouteMatrixNotPublicationEligible("route matrix is not publication ready; inspect validation report")
    return report


def build_validated_route_matrix_from_cache(
    path: Path,
    context_snapshot_id: str,
    output_dir: str | Path,
    *,
    required_sequences: Sequence[Sequence[str]] = (),
    require_publication_ready: bool = False,
    source_bundle_id: str = "",
    expected_source_sha256: str = "",
) -> tuple[RouteMatrix, RouteMatrixValidationReport]:
    """Load route evidence, write matrix artifacts, and validate publication readiness."""

    matrix = load_route_matrix_from_cache(
        path,
        context_snapshot_id,
        source_bundle_id=source_bundle_id,
        expected_source_sha256=expected_source_sha256,
    )
    report = write_validated_route_matrix_artifacts(
        matrix,
        output_dir,
        required_sequences=required_sequences,
        require_publication_ready=require_publication_ready,
    )
    return matrix, report


def load_route_matrix_from_cache(
    path: Path,
    context_snapshot_id: str,
    *,
    source_bundle_id: str = "",
    expected_source_sha256: str = "",
) -> RouteMatrix:
    """Load a content-addressed RouteMatrix from route evidence."""

    matrix_path = Path(path)
    if not matrix_path.exists():
        raise RouteMatrixMissing(f"route matrix cache not found: {matrix_path}")
    source_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if expected_source_sha256 and source_sha256.lower() != str(expected_source_sha256).lower():
        raise RouteMatrixNotPublicationEligible(
            "route cache SHA-256 does not match the frozen evidence manifest"
        )
    matrix_id = f"route_matrix_{_stable_hash({'source_sha256': source_sha256, 'context_snapshot_id': context_snapshot_id})}"
    frame = pd.read_csv(matrix_path)
    if frame.empty:
        return RouteMatrix(
            matrix_id=matrix_id,
            context_snapshot_id=str(context_snapshot_id),
            entity_ids=(),
            cells={},
            source_bundle_id=str(source_bundle_id),
            source_content_sha256=source_sha256,
        )
    cells: dict[tuple[str, str], RouteMatrixCell] = {}
    for _, row in frame.iterrows():
        origin_id = _safe_str(row.get("origin_id")) or _safe_str(row.get("origin_label"))
        destination_id = _safe_str(row.get("destination_id")) or _safe_str(row.get("destination_label"))
        if not origin_id or not destination_id:
            continue
        row_context = _safe_str(row.get("context_snapshot_id"), str(context_snapshot_id))
        provider = _safe_str(row.get("provider"), _safe_str(row.get("routing_source"), "route_matrix_csv"))
        cell = RouteMatrixCell(
            origin_id=origin_id,
            destination_id=destination_id,
            distance_m=_safe_float(row.get("distance_m")),
            duration_s=_safe_float(row.get("duration_s")),
            route_leg_id=_safe_str(row.get("route_option_id"), _safe_str(row.get("route_leg_id"))),
            road_validated=_safe_bool(row.get("road_validated")),
            fallback_used=_safe_bool(row.get("fallback_used")),
            provider=provider,
            query_hash=_safe_str(row.get("query_hash")),
            context_snapshot_id=row_context,
            geometry=_parse_geometry(row.get("geometry")),
            routing_profile=_safe_str(row.get("routing_profile"), "driving"),
            routing_status=_safe_str(row.get("routing_status"), "ok"),
            geometry_source=_safe_str(row.get("geometry_source"), "route_matrix_geometry"),
            distance_source=_safe_str(row.get("distance_source"), "route_matrix_distance"),
            duration_source=_safe_str(row.get("duration_source"), "route_matrix_duration"),
            fallback_reason=_safe_str(row.get("fallback_reason")) or None,
        )
        cells[(cell.origin_id, cell.destination_id)] = cell
    return RouteMatrix(
        matrix_id=matrix_id,
        context_snapshot_id=str(context_snapshot_id),
        entity_ids=(),
        cells=cells,
        source_bundle_id=str(source_bundle_id),
        source_content_sha256=source_sha256,
    )


def build_route_matrix_from_context(bundle: Any) -> RouteMatrix:
    """Build a matrix from a loaded DatasetBundle's route_options table."""

    routes = bundle.table("route_options")
    context_snapshot_id = str(bundle.context_snapshot_id)
    cells: dict[tuple[str, str], RouteMatrixCell] = {}
    for _, row in routes.iterrows():
        cell = RouteMatrixCell(
            origin_id=_safe_str(row.get("origin_id")),
            destination_id=_safe_str(row.get("destination_id")),
            distance_m=_safe_float(row.get("distance_m")),
            duration_s=_safe_float(row.get("duration_s")),
            route_leg_id=_safe_str(row.get("route_option_id")),
            road_validated=_safe_bool(row.get("road_validated")),
            fallback_used=_safe_bool(row.get("fallback_used")),
            provider=_safe_str(row.get("routing_source"), _safe_str(row.get("provider"), "context_route_options")),
            query_hash=_safe_str(row.get("query_hash")),
            context_snapshot_id=_safe_str(row.get("context_snapshot_id"), context_snapshot_id),
            geometry=_parse_geometry(row.get("geometry")),
            routing_profile=_safe_str(row.get("routing_profile"), "driving"),
            routing_status=_safe_str(row.get("routing_status"), "context_route_option"),
            geometry_source=_safe_str(row.get("geometry_source"), "context_route_geometry"),
            distance_source=_safe_str(row.get("distance_source"), "context_route_distance"),
            duration_source=_safe_str(row.get("duration_source"), "context_route_duration"),
            fallback_reason=_safe_str(row.get("fallback_reason")) or None,
        )
        cells[(cell.origin_id, cell.destination_id)] = cell
    return RouteMatrix(
        matrix_id=f"context_route_matrix_{context_snapshot_id}",
        context_snapshot_id=context_snapshot_id,
        entity_ids=(),
        cells=cells,
    )


def geodesic_fallback_matrix(entity_points: Mapping[str, tuple[float, float]]) -> RouteMatrix:
    """Build an explicit non-publication matrix from geodesic speed proxies."""

    cells: dict[tuple[str, str], RouteMatrixCell] = {}
    normalized_points = {
        _normalize_entity_id(entity_id): (float(point[0]), float(point[1]))
        for entity_id, point in entity_points.items()
        if _normalize_entity_id(entity_id)
    }
    for origin_id, origin_point in normalized_points.items():
        for destination_id, destination_point in normalized_points.items():
            if origin_id == destination_id:
                continue
            distance_m = float(geodesic(origin_point, destination_point).km * 1000.0)
            duration_s = float(distance_m / 1000.0 * 1.25 / 38.0 * 3600.0)
            cell = RouteMatrixCell(
                origin_id=origin_id,
                destination_id=destination_id,
                distance_m=distance_m,
                duration_s=duration_s,
                route_leg_id=f"fallback_{origin_id}_{destination_id}",
                road_validated=False,
                fallback_used=True,
                provider="geodesic_proxy",
                context_snapshot_id="geodesic_fallback",
                geometry=(origin_point, destination_point),
                routing_status="fallback_geodesic_proxy",
                geometry_source="straight_waypoints",
                distance_source="geodesic_proxy",
                duration_source="geodesic_speed_proxy",
                fallback_reason="explicit_demo_geodesic_fallback",
            )
            cells[(cell.origin_id, cell.destination_id)] = cell
    return RouteMatrix(
        matrix_id=f"geodesic_fallback_{_stable_hash(sorted(normalized_points))}",
        context_snapshot_id="geodesic_fallback",
        entity_ids=tuple(normalized_points),
        cells=cells,
    )


def route_minutes_from_matrix(matrix: RouteMatrix, origin_id: str, destination_id: str) -> float:
    """Return solver-ready minutes for one route matrix cell."""

    return matrix.duration_minutes(origin_id, destination_id)


def route_result_for_sequence(
    matrix: RouteMatrix,
    sequence: tuple[str, ...],
    *,
    route_id: str | None = None,
    strict: bool = False,
    solver_feasible: bool = False,
    schedule_feasible: bool = False,
    dataset_snapshot_valid: bool = False,
) -> RouteResult:
    """Create a RouteResult from the same cells consumed by solvers."""

    normalized = tuple(_normalize_entity_id(value) for value in sequence)
    legs = tuple(matrix.leg(left, right, strict=strict) for left, right in zip(normalized[:-1], normalized[1:], strict=False))
    return RouteResult(
        route_id=route_id or f"route_{matrix.matrix_id}_{_stable_hash(normalized)}",
        legs=legs,
        solver_feasible=solver_feasible,
        schedule_feasible=schedule_feasible,
        dataset_snapshot_valid=dataset_snapshot_valid,
    )


def _publication_mode(mode: str) -> bool:
    return str(mode).lower() in {"publication", "strict", "final", "research"}


@dataclass(frozen=True)
class SolverRouteMatrixAdapter:
    """Solver-facing access to route matrix durations with strict-mode gates."""

    route_matrix: RouteMatrix
    mode: str = "publication"

    @property
    def strict(self) -> bool:
        return _publication_mode(self.mode)

    def travel_minutes(self, origin_id: str, destination_id: str) -> float:
        return self.route_matrix.duration_minutes(origin_id, destination_id, strict=self.strict)

    def distance_m(self, origin_id: str, destination_id: str) -> float:
        return self.route_matrix.distance_meters(origin_id, destination_id, strict=self.strict)

    def assert_publication_ready(self, sequence: Sequence[str] | None = None) -> None:
        self.route_matrix.require_road_validated(sequence)

    def route_result(
        self,
        sequence: Sequence[str],
        *,
        route_id: str | None = None,
        solver_feasible: bool = False,
        schedule_feasible: bool = False,
        dataset_snapshot_valid: bool = False,
    ) -> RouteResult:
        return route_result_for_sequence(
            self.route_matrix,
            tuple(str(value) for value in sequence),
            route_id=route_id,
            strict=self.strict,
            solver_feasible=solver_feasible,
            schedule_feasible=schedule_feasible,
            dataset_snapshot_valid=dataset_snapshot_valid,
        )
