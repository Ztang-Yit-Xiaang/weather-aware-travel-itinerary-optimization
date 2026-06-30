"""Load and validate clean-clone research data snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .schemas import DatasetBundle, DatasetValidationReport

CATALOG_TABLES = (
    "poi_entities",
    "poi_observations",
    "poi_features",
    "feature_provenance",
    "hotel_entities",
    "source_audit",
)
CONTEXT_TABLES = ("weather_scenarios", "route_options")
DEFAULT_CONTEXT_SNAPSHOT_ID = "context_static_demo_2026_06"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot table is missing: {path}")
    return pd.read_csv(path)


def load_dataset_bundle(
    catalog_snapshot_id: str = "california_v1",
    *,
    context_snapshot_id: str | None = None,
    root: str | Path | None = None,
) -> DatasetBundle:
    """Load a stable catalog snapshot and its default context tables."""

    base = Path(root) if root is not None else _repo_root()
    snapshot_dir = base / "data" / "snapshots" / str(catalog_snapshot_id)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_context_id = str(context_snapshot_id or manifest.get("context_snapshot_id") or DEFAULT_CONTEXT_SNAPSHOT_ID)

    tables: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {"manifest.json": _sha256(manifest_path)}
    for filename in sorted((manifest.get("files") or {}).keys()):
        path = snapshot_dir / str(filename)
        if path.exists():
            file_hashes[str(filename)] = _sha256(path)
    for table_name in (*CATALOG_TABLES, *CONTEXT_TABLES):
        filename = f"{table_name}.csv"
        path = snapshot_dir / filename
        tables[table_name] = _read_csv(path)
        file_hashes[filename] = _sha256(path)

    return DatasetBundle(
        catalog_snapshot_id=str(manifest.get("catalog_snapshot_id") or catalog_snapshot_id),
        context_snapshot_id=resolved_context_id,
        snapshot_dir=snapshot_dir,
        manifest=manifest,
        tables=tables,
        file_hashes=file_hashes,
    )


def _missing_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(frame.columns))


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].astype(str).str.lower().isin({"true", "1", "yes"})


def validate_dataset_bundle(bundle: DatasetBundle) -> DatasetValidationReport:
    """Validate enough snapshot structure to use it as a Phase 0 fallback."""

    errors: list[str] = []
    warnings: list[str] = []
    counts = {name: len(frame) for name, frame in bundle.tables.items()}

    for name in CATALOG_TABLES:
        if name not in bundle.tables:
            errors.append(f"missing catalog table: {name}")
    for name in CONTEXT_TABLES:
        if name not in bundle.tables:
            errors.append(f"missing context table: {name}")

    entities = bundle.tables.get("poi_entities", pd.DataFrame())
    missing = _missing_columns(
        entities,
        {"poi_id", "canonical_name", "latitude", "longitude", "entity_type", "canonical_city"},
    )
    if missing:
        errors.append(f"poi_entities missing columns: {missing}")
    elif entities.empty:
        errors.append("poi_entities is empty")
    else:
        if entities["poi_id"].duplicated().any():
            errors.append("poi_entities contains duplicate poi_id values")
        lat = pd.to_numeric(entities["latitude"], errors="coerce")
        lon = pd.to_numeric(entities["longitude"], errors="coerce")
        invalid_coords = lat.isna().any() or lon.isna().any()
        invalid_coords = invalid_coords or (~lat.between(-90, 90)).any() or (~lon.between(-180, 180)).any()
        if invalid_coords:
            errors.append("poi_entities contains invalid coordinates")

    feature_provenance = bundle.tables.get("feature_provenance", pd.DataFrame())
    features = bundle.tables.get("poi_features", pd.DataFrame())
    if "source_coverage_score" not in features.columns:
        errors.append("poi_features missing source_coverage_score")
    if "data_confidence" in features.columns:
        warnings.append("poi_features still contains legacy data_confidence; use source_coverage_score for new work")
    if not entities.empty and not feature_provenance.empty and "poi_id" in feature_provenance.columns:
        missing_provenance = set(entities["poi_id"].astype(str)) - set(feature_provenance["poi_id"].astype(str))
        if missing_provenance:
            errors.append(f"feature_provenance missing poi_id rows: {sorted(missing_provenance)[:5]}")
    else:
        errors.append("feature_provenance is empty or missing poi_id")

    weather = bundle.tables.get("weather_scenarios", pd.DataFrame())
    missing_weather = _missing_columns(weather, {"context_snapshot_id", "valid_time", "retrieved_at", "source"})
    if missing_weather:
        errors.append(f"weather_scenarios missing columns: {missing_weather}")
    elif not weather["context_snapshot_id"].astype(str).eq(bundle.context_snapshot_id).all():
        errors.append("weather_scenarios context_snapshot_id does not match bundle context")

    routes = bundle.tables.get("route_options", pd.DataFrame())
    missing_routes = _missing_columns(
        routes,
        {
            "route_option_id",
            "context_snapshot_id",
            "origin_id",
            "destination_id",
            "geometry_source",
            "distance_source",
            "duration_source",
            "road_validated",
        },
    )
    if missing_routes:
        errors.append(f"route_options missing columns: {missing_routes}")
    elif not routes.empty:
        if not routes["context_snapshot_id"].astype(str).eq(bundle.context_snapshot_id).all():
            errors.append("route_options context_snapshot_id does not match bundle context")
        if not _bool_series(routes, "road_validated").all():
            warnings.append("route_options contains non-road-validated fallback legs; final comparisons must gate them")

    observations = bundle.tables.get("poi_observations", pd.DataFrame())
    if "source_name" in observations.columns:
        private_sources = [
            value
            for value in observations["source_name"].dropna().astype(str).unique().tolist()
            if "yelp" in value.lower()
        ]
        if private_sources:
            errors.append(f"fallback snapshot includes private Yelp-derived sources: {private_sources}")

    can_optimize = not errors and counts.get("poi_entities", 0) > 0 and counts.get("poi_features", 0) > 0
    final_comparison_eligible = can_optimize and not routes.empty and _bool_series(routes, "road_validated").all()

    return DatasetValidationReport(
        catalog_snapshot_id=bundle.catalog_snapshot_id,
        context_snapshot_id=bundle.context_snapshot_id,
        can_optimize=bool(can_optimize),
        final_comparison_eligible=bool(final_comparison_eligible),
        errors=tuple(errors),
        warnings=tuple(warnings),
        table_counts=counts,
    )
