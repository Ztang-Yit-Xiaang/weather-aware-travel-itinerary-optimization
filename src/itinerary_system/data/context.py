"""Time-sensitive context snapshot loading helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .schemas import ContextBundle

CONTEXT_TABLES = ("weather_scenarios", "route_options")
DEFAULT_CONTEXT_SNAPSHOT_ID = "context_static_demo_2026_06"


class SnapshotLoadError(RuntimeError):
    """Raised when a catalog or context snapshot cannot be loaded."""


class SnapshotTableMissing(SnapshotLoadError, FileNotFoundError):
    """Raised when a manifest-required snapshot table is absent."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a local snapshot file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    """Read a snapshot manifest as a JSON object."""

    if not path.exists():
        raise SnapshotTableMissing(f"Snapshot manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SnapshotLoadError(f"Snapshot manifest must be a JSON object: {path}")
    return payload


def read_csv_table(path: Path) -> pd.DataFrame:
    """Read a required snapshot CSV table."""

    if not path.exists():
        raise SnapshotTableMissing(f"Snapshot table is missing: {path}")
    return pd.read_csv(path)


def _table_names(manifest: dict, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_tables = manifest.get("context_tables") or [f"{table}.csv" for table in default]
    names = []
    for value in raw_tables:
        text = str(value)
        names.append(text[:-4] if text.endswith(".csv") else text)
    return tuple(names)


def _load_tables(source_dir: Path, table_names: tuple[str, ...]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    tables: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {}
    for table_name in table_names:
        filename = f"{table_name}.csv"
        path = source_dir / filename
        tables[table_name] = read_csv_table(path)
        file_hashes[filename] = sha256_file(path)
    return tables, file_hashes


def load_context_bundle(
    context_snapshot_id: str = DEFAULT_CONTEXT_SNAPSHOT_ID,
    *,
    root: str | Path | None = None,
    legacy_snapshot_dir: str | Path | None = None,
    legacy_manifest: dict | None = None,
) -> ContextBundle:
    """Load a context snapshot, preferring the separated context directory."""

    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    context_dir = base / "data" / "contexts" / str(context_snapshot_id)
    manifest_path = context_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        table_names = _table_names(manifest, CONTEXT_TABLES)
        tables, file_hashes = _load_tables(context_dir, table_names)
        file_hashes["manifest.json"] = sha256_file(manifest_path)
        return ContextBundle(
            context_snapshot_id=str(manifest.get("context_snapshot_id") or context_snapshot_id),
            context_dir=context_dir,
            manifest=manifest,
            tables=tables,
            file_hashes=file_hashes,
        )

    if legacy_snapshot_dir is None:
        raise SnapshotTableMissing(f"Context manifest is missing: {manifest_path}")

    legacy_dir = Path(legacy_snapshot_dir)
    manifest = dict(legacy_manifest or {})
    if "context_tables" not in manifest:
        raise SnapshotTableMissing(f"Context manifest is missing: {manifest_path}")
    table_names = _table_names(manifest, CONTEXT_TABLES)
    tables, file_hashes = _load_tables(legacy_dir, table_names)
    return ContextBundle(
        context_snapshot_id=str(context_snapshot_id),
        context_dir=legacy_dir,
        manifest={
            "context_schema_version": "legacy-combined-snapshot",
            "context_snapshot_id": str(context_snapshot_id),
            "legacy_combined_snapshot": True,
            "context_tables": [f"{table}.csv" for table in table_names],
            "files": {filename: manifest.get("files", {}).get(filename, "") for filename in file_hashes},
        },
        tables=tables,
        file_hashes=file_hashes,
        legacy_combined_snapshot=True,
    )
