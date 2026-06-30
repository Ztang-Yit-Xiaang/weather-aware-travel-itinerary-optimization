"""Road-route cache helpers for Phase 0 validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .models import RouteLegResult

ROAD_ROUTE_CACHE_FILENAME = "production_road_route_cache.csv"


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


def _safe_float(value: Any, default: float = math.nan) -> float:
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


def route_anchor_key(label: Any) -> str:
    text = _safe_str(label).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


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
        if math.isfinite(lat) and math.isfinite(lon):
            points.append((lat, lon))
    return tuple(points)


@dataclass(frozen=True)
class RoadRouteCache:
    """Validated route-leg cache keyed by route anchor labels or IDs."""

    frame: pd.DataFrame
    source_path: Path | None = None

    @classmethod
    def from_csv(cls, path: str | Path) -> "RoadRouteCache":
        cache_path = Path(path)
        if not cache_path.exists():
            return cls(pd.DataFrame(), source_path=cache_path)
        frame = pd.read_csv(cache_path)
        return cls(_normalize_cache_frame(frame), source_path=cache_path)

    @property
    def empty(self) -> bool:
        return self.frame.empty

    def lookup_leg(self, left: dict[str, Any], right: dict[str, Any]) -> RouteLegResult | None:
        if self.frame.empty:
            return None
        origin_keys = {
            route_anchor_key(left.get("id")),
            route_anchor_key(left.get("label")),
        }
        destination_keys = {
            route_anchor_key(right.get("id")),
            route_anchor_key(right.get("label")),
        }
        origin_keys.discard("")
        destination_keys.discard("")
        if not origin_keys or not destination_keys:
            return None
        matches = self.frame[
            self.frame["origin_key"].isin(origin_keys)
            & self.frame["destination_key"].isin(destination_keys)
            & self.frame["road_validated_bool"]
        ]
        if matches.empty:
            return None
        row = matches.iloc[0]
        geometry = _parse_geometry(row.get("geometry"))
        if not geometry:
            left_lat = _safe_float(left.get("latitude"))
            left_lon = _safe_float(left.get("longitude"))
            right_lat = _safe_float(right.get("latitude"))
            right_lon = _safe_float(right.get("longitude"))
            if all(math.isfinite(value) for value in [left_lat, left_lon, right_lat, right_lon]):
                geometry = ((left_lat, left_lon), (right_lat, right_lon))
        if not geometry:
            return None
        return RouteLegResult(
            origin_id=_safe_str(row.get("origin_id"), _safe_str(left.get("id"))),
            destination_id=_safe_str(row.get("destination_id"), _safe_str(right.get("id"))),
            geometry=geometry,
            distance_m=_safe_float(row.get("distance_m"), None),
            duration_s=_safe_float(row.get("duration_s"), None),
            routing_status=_safe_str(row.get("routing_status"), "ok"),
            provider=_safe_str(row.get("provider"), "road_route_cache"),
            routing_profile=_safe_str(row.get("routing_profile"), "driving"),
            geometry_source=_safe_str(row.get("geometry_source"), "road_route_cache_geometry"),
            distance_source=_safe_str(row.get("distance_source"), "road_route_cache_distance"),
            duration_source=_safe_str(row.get("duration_source"), "road_route_cache_duration"),
            road_validated=True,
            fallback_used=False,
            fallback_reason=None,
            query_hash=_safe_str(row.get("query_hash")),
        )


def _normalize_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    for column in [
        "origin_id",
        "destination_id",
        "origin_label",
        "destination_label",
        "road_validated",
        "geometry",
        "distance_m",
        "duration_s",
        "provider",
        "routing_profile",
        "routing_status",
        "geometry_source",
        "distance_source",
        "duration_source",
        "query_hash",
    ]:
        if column not in output.columns:
            output[column] = ""
    output["origin_key"] = output.apply(
        lambda row: route_anchor_key(row.get("origin_id")) or route_anchor_key(row.get("origin_label")),
        axis=1,
    )
    output["destination_key"] = output.apply(
        lambda row: route_anchor_key(row.get("destination_id")) or route_anchor_key(row.get("destination_label")),
        axis=1,
    )
    output["road_validated_bool"] = output["road_validated"].map(_safe_bool)
    return output


def load_road_route_cache(output_dir: str | Path, config: Any | None = None) -> RoadRouteCache:
    """Load the configured or default production road-route cache."""

    output_path = Path(output_dir)
    configured_path = None
    if config is not None:
        try:
            configured_path = _safe_str(config.get("routing", "road_route_cache_path", ""))
        except Exception:
            configured_path = ""
    candidates: list[Path] = []
    if configured_path:
        configured = Path(configured_path)
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.extend([output_path / configured, configured])
    candidates.append(output_path / ROAD_ROUTE_CACHE_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            return RoadRouteCache.from_csv(candidate)
    return RoadRouteCache(pd.DataFrame(), source_path=candidates[0] if candidates else None)
