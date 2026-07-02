"""Dataclasses for stable catalog and time-sensitive context snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CatalogBundle:
    """Loaded stable catalog tables for one reproducible research snapshot."""

    catalog_snapshot_id: str
    snapshot_dir: Path
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    file_hashes: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Catalog table not loaded: {name}")
        return self.tables[name].copy()


@dataclass(frozen=True)
class ContextBundle:
    """Loaded time-sensitive context tables for one reproducible research snapshot."""

    context_snapshot_id: str
    context_dir: Path
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    file_hashes: dict[str, str] = field(default_factory=dict)
    legacy_combined_snapshot: bool = False

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Context table not loaded: {name}")
        return self.tables[name].copy()


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded catalog and context tables for one reproducible research snapshot."""

    catalog: CatalogBundle
    context: ContextBundle

    @property
    def catalog_snapshot_id(self) -> str:
        return self.catalog.catalog_snapshot_id

    @property
    def context_snapshot_id(self) -> str:
        return self.context.context_snapshot_id

    @property
    def snapshot_dir(self) -> Path:
        return self.catalog.snapshot_dir

    @property
    def context_dir(self) -> Path:
        return self.context.context_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return self.catalog.manifest

    @property
    def context_manifest(self) -> dict[str, Any]:
        return self.context.manifest

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {**self.catalog.tables, **self.context.tables}

    @property
    def file_hashes(self) -> dict[str, str]:
        output = {f"catalog/{key}": value for key, value in self.catalog.file_hashes.items()}
        output.update({f"context/{key}": value for key, value in self.context.file_hashes.items()})
        for key, value in self.catalog.file_hashes.items():
            output.setdefault(key, value)
        for key, value in self.context.file_hashes.items():
            output.setdefault(key, value)
        return output

    def table(self, name: str) -> pd.DataFrame:
        if name in self.catalog.tables:
            return self.catalog.table(name)
        if name in self.context.tables:
            return self.context.table(name)
        raise KeyError(f"Dataset table not loaded: {name}")


@dataclass(frozen=True)
class DatasetValidationReport:
    """Validation result for a catalog/context bundle."""

    catalog_snapshot_id: str
    context_snapshot_id: str
    can_optimize: bool
    final_comparison_eligible: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    table_counts: dict[str, int] = field(default_factory=dict)
