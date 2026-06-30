"""Dataclasses for stable catalog and time-sensitive context snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded catalog and context tables for one reproducible research snapshot."""

    catalog_snapshot_id: str
    context_snapshot_id: str
    snapshot_dir: Path
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    file_hashes: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Snapshot table not loaded: {name}")
        return self.tables[name].copy()


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

