"""Load and validate clean-clone research data snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .context import (
    CONTEXT_TABLES,
    DEFAULT_CONTEXT_SNAPSHOT_ID,
    load_context_bundle,
    read_csv_table,
    read_manifest,
    sha256_file,
)
from .schemas import CatalogBundle, DatasetBundle, DatasetValidationReport

CATALOG_TABLES = (
    "poi_entities",
    "poi_observations",
    "poi_features",
    "feature_provenance",
    "hotel_entities",
    "source_audit",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return read_csv_table(path)


def _table_names(manifest: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_tables = manifest.get(key) or [f"{table}.csv" for table in default]
    names = []
    for value in raw_tables:
        text = str(value)
        names.append(text[:-4] if text.endswith(".csv") else text)
    return tuple(names)


def load_catalog_bundle(
    catalog_snapshot_id: str = "california_v1",
    *,
    root: str | Path | None = None,
) -> CatalogBundle:
    """Load stable catalog tables from data/snapshots/<catalog_snapshot_id>."""

    base = Path(root) if root is not None else _repo_root()
    snapshot_dir = base / "data" / "snapshots" / str(catalog_snapshot_id)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    table_names = _table_names(manifest, "catalog_tables", CATALOG_TABLES)

    tables: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {"manifest.json": _sha256(manifest_path)}
    for filename in sorted((manifest.get("files") or {}).keys()):
        path = snapshot_dir / str(filename)
        if path.exists():
            file_hashes[str(filename)] = _sha256(path)
    for table_name in table_names:
        filename = f"{table_name}.csv"
        path = snapshot_dir / filename
        tables[table_name] = _read_csv(path)
        file_hashes[filename] = _sha256(path)

    return CatalogBundle(
        catalog_snapshot_id=str(manifest.get("catalog_snapshot_id") or catalog_snapshot_id),
        snapshot_dir=snapshot_dir,
        manifest=manifest,
        tables=tables,
        file_hashes=file_hashes,
    )


def load_dataset_bundle(
    catalog_snapshot_id: str = "california_v1",
    *,
    context_snapshot_id: str | None = None,
    root: str | Path | None = None,
) -> DatasetBundle:
    """Load a stable catalog snapshot and its default context tables."""

    catalog = load_catalog_bundle(catalog_snapshot_id, root=root)
    resolved_context_id = str(
        context_snapshot_id
        or catalog.manifest.get("default_context_snapshot_id")
        or catalog.manifest.get("context_snapshot_id")
        or DEFAULT_CONTEXT_SNAPSHOT_ID
    )
    context = load_context_bundle(
        resolved_context_id,
        root=root,
        legacy_snapshot_dir=catalog.snapshot_dir,
        legacy_manifest=catalog.manifest,
    )
    return DatasetBundle(catalog=catalog, context=context)


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

    catalog_files = bundle.catalog.manifest.get("files", {})
    if isinstance(catalog_files, dict):
        for filename, expected_hash in catalog_files.items():
            actual_hash = bundle.catalog.file_hashes.get(str(filename))
            if actual_hash and expected_hash and actual_hash != expected_hash:
                errors.append(f"catalog manifest hash mismatch: {filename}")

    context_files = bundle.context.manifest.get("files", {})
    if isinstance(context_files, dict):
        for filename, expected_hash in context_files.items():
            actual_hash = bundle.context.file_hashes.get(str(filename))
            if actual_hash and expected_hash and actual_hash != expected_hash:
                errors.append(f"context manifest hash mismatch: {filename}")

    if bundle.context.legacy_combined_snapshot:
        warnings.append("context snapshot loaded from legacy combined catalog snapshot")

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
