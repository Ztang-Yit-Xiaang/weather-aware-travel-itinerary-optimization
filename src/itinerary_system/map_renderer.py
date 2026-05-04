"""Map rendering compatibility wrapper."""

from __future__ import annotations

from pathlib import Path

from ._legacy import import_legacy_module
from .config import TripConfig


def build_map(context: dict, config: TripConfig, output_path: str | Path | None = None):
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")
    merged_context = dict(context)
    merged_context["RUN_LIVE_APIS"] = bool(config.get("enrichment", "run_live_apis", False))
    return blueprint_trip_map.build_production_trip_map(
        merged_context,
        output_path=output_path,
        run_live_routing=bool(config.get("enrichment", "run_live_apis", False)),
    )
