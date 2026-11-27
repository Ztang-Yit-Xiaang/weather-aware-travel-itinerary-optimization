"""Equal-area, land-masked interest density and exact raster mass regions.

The mask is cartographic evidence, not a walking-access or route certificate.
Boundary cells retain their supported land area and published polygons are clipped.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import shapely
from matplotlib.path import Path as PolygonPath
from matplotlib.transforms import Bbox
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

EARTH_RADIUS_M = 6_371_008.8
MASS_LEVELS = (0.5, 0.8, 0.95)


def _immutable_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable_json(item) for item in value)
    return value


@lru_cache(maxsize=2)
def _decoded_support(payload: bytes):
    """Cache immutable cartographic input only; never cache personal inference."""
    geometry = json.loads(payload)
    if geometry.get("type") != "FeatureCollection" or not geometry.get("features"):
        raise ValueError("interest_support_invalid")
    geometry["_compiled_polygons"] = tuple(
        (kind, Polygon(np.asarray(polygon[0]), [np.asarray(hole) for hole in polygon[1:]]))
        for kind, polygon in _rings(geometry)
    )
    return _immutable_json(geometry)


def load_land_support(repository_root: Path) -> tuple[Mapping, dict]:
    directory = repository_root / "configs" / "interest_regions"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "interest-land-support-v1" or manifest.get("file") != "land_support.geojson":
        raise ValueError("interest_support_invalid")
    source = directory / "land_support.geojson"
    if source.is_symlink() or source.stat().st_size > 5_000_000:
        raise ValueError("interest_support_invalid")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("sha256"):
        raise ValueError("interest_support_hash_mismatch")
    return _decoded_support(payload), manifest


def _project(points: np.ndarray, center: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    lon0, lat0 = np.radians(center)
    if not inverse:
        lon, lat = np.radians(points).T
        delta = lon - lon0
        k = np.sqrt(2 / (1 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(delta)))
        return EARTH_RADIUS_M * np.column_stack(
            (
                k * np.cos(lat) * np.sin(delta),
                k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(delta)),
            )
        )
    x, y = points.T
    rho = np.hypot(x, y)
    c = 2 * np.arcsin(np.minimum(1, rho / (2 * EARTH_RADIUS_M)))
    safe = np.where(rho == 0, 1, rho)
    lat = np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c) * np.cos(lat0) / safe)
    lon = lon0 + np.arctan2(x * np.sin(c), rho * np.cos(lat0) * np.cos(c) - y * np.sin(lat0) * np.sin(c))
    lat = np.where(rho == 0, lat0, lat)
    return np.degrees(np.column_stack((lon, lat)))


def _rings(geometry: Mapping):
    for feature in geometry["features"]:
        shape = feature["geometry"]
        polygons = shape["coordinates"] if shape["type"] == "MultiPolygon" else [shape["coordinates"]]
        for polygon in polygons:
            yield feature["properties"]["kind"], polygon


def land_mask(points: np.ndarray, geometry: Mapping, coverage: list[float]) -> np.ndarray:
    """Test sourced land minus sourced inland water and explicit restricted polygons."""
    points = np.asarray(points, dtype=float)
    window = Bbox.from_extents(*(points.min(axis=0) - 1e-8), *(points.max(axis=0) + 1e-8))
    land = np.zeros(len(points), dtype=bool)
    excluded = land.copy()
    for kind, polygon in _rings(geometry):
        exterior = PolygonPath(polygon[0])
        if not exterior.get_extents().overlaps(window):
            continue
        inside = exterior.clip_to_bbox(window).contains_points(points)
        for hole in polygon[1:]:
            inside &= ~PolygonPath(hole).clip_to_bbox(window).contains_points(points)
        if kind == "land":
            land |= inside
        elif kind in {"water", "restricted"}:
            excluded |= inside
        else:
            raise ValueError("interest_support_kind_invalid")
    west, south, east, north = coverage
    return (
        land
        & ~excluded
        & (points[:, 0] >= west)
        & (points[:, 0] <= east)
        & (points[:, 1] >= south)
        & (points[:, 1] <= north)
    )


def _local_support(geometry: Mapping, coverage: list[float], window) -> object:
    """Intersect sourced land, water, restrictions and coverage without raster erosion."""
    land, excluded = [], []
    source_polygons = geometry.get("_compiled_polygons")
    if source_polygons is None:
        source_polygons = (
            (kind, Polygon(np.asarray(polygon[0]), [np.asarray(hole) for hole in polygon[1:]]))
            for kind, polygon in _rings(geometry)
        )
    west, south, east, north = window.bounds
    for kind, feature in source_polygons:
        left, bottom, right, top = feature.bounds
        if right < west or left > east or top < south or bottom > north:
            continue
        try:
            clipped = feature.intersection(window)
        except shapely.errors.GEOSException as exc:
            raise ValueError("interest_support_geometry_invalid") from exc
        if not clipped.is_valid:
            raise ValueError("interest_support_geometry_invalid")
        if kind == "land":
            land.append(clipped)
        elif kind in {"water", "restricted"}:
            excluded.append(clipped)
        else:
            raise ValueError("interest_support_kind_invalid")
    return shapely.union_all(land).difference(shapely.union_all(excluded)).intersection(shapely.box(*coverage))


def _clipped_polygons(mask, x, y, center, support) -> list:
    coordinates = _raster_polygons(mask, x, y, center)
    if not coordinates:
        return []
    clipped = shape({"type": "MultiPolygon", "coordinates": coordinates}).intersection(support)
    parts, polygons = [clipped], []
    while parts:
        part = parts.pop()
        if part.geom_type == "Polygon" and part.area > 0:
            polygons.append(part)
        elif part.geom_type in {"MultiPolygon", "GeometryCollection"}:
            parts.extend(part.geoms)
    return mapping(MultiPolygon(polygons))["coordinates"]


def _raster_polygons(mask: np.ndarray, x: np.ndarray, y: np.ndarray, center: np.ndarray) -> list:
    """Union disjoint grid cells, retaining holes and disconnected components."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return []
    cells = shapely.box(x[cols], y[rows], x[cols + 1], y[rows + 1])
    union = shapely.coverage_union_all(cells)
    geographic = shapely.transform(union, lambda coordinates: _project(coordinates, center, inverse=True))
    if geographic.geom_type == "Polygon":
        geographic = MultiPolygon([geographic])
    return mapping(geographic)["coordinates"]


def build_interest_surface(
    coordinates: np.ndarray,
    probability_draws: np.ndarray,
    *,
    geometry: dict,
    manifest: dict,
    bandwidth_m: float = 1000.0,
    grid_size: int = 128,
    search_center: np.ndarray | None = None,
    search_radius_m: float | None = None,
) -> dict:
    points = np.asarray(coordinates, dtype=float)
    draws = np.asarray(probability_draws, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 1 <= len(points) <= 500
        or draws.ndim != 2
        or draws.shape[0] != len(points)
        or draws.shape[1] < 1
        or not np.isfinite(points).all()
        or not np.isfinite(draws).all()
        or np.any((draws < 0) | (draws > 1))
        or np.any(np.abs(points[:, 0]) > 180)
        or np.any(np.abs(points[:, 1]) > 85)
        or not 250 <= bandwidth_m <= 5000
        or isinstance(grid_size, bool)
        or not isinstance(grid_size, int)
        or not 32 <= grid_size <= 128
    ):
        raise ValueError("interest_surface_input_invalid")
    center = points.mean(axis=0)
    if search_center is not None:
        search_center = np.asarray(search_center, dtype=float)
        if (
            search_center.shape != (2,)
            or not np.isfinite(search_center).all()
            or np.any(np.abs(search_center) > [180, 85])
            or isinstance(search_radius_m, bool)
            or not isinstance(search_radius_m, (int, float))
            or not np.isfinite(search_radius_m)
            or not 1000 <= search_radius_m <= 25000
        ):
            raise ValueError("interest_search_area_invalid")
    elif search_radius_m is not None:
        raise ValueError("interest_search_area_invalid")
    projected = _project(points, center)
    low, high = projected.min(axis=0) - 3 * bandwidth_m, projected.max(axis=0) + 3 * bandwidth_m
    if np.max(high - low) > 250_000:
        raise ValueError("interest_surface_context_too_wide")
    x, y = np.linspace(low[0], high[0], grid_size + 1), np.linspace(low[1], high[1], grid_size + 1)
    xx, yy = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    corners = _project(np.array([low, [low[0], high[1]], high, [high[0], low[1]]]), center, inverse=True)
    window = shapely.box(*(corners.min(axis=0) - 0.01), *(corners.max(axis=0) + 0.01))
    geographic_support = _local_support(geometry, manifest["coverage_bbox"], window)
    projected_support = shapely.transform(geographic_support, lambda coordinates: _project(coordinates, center))
    if search_center is not None:
        local_center = _project(search_center[None, :], center)[0]
        projected_support = projected_support.intersection(
            shapely.Point(local_center).buffer(search_radius_m, quad_segs=64)
        )
        geographic_support = shapely.transform(
            projected_support, lambda coordinates: _project(coordinates, center, inverse=True)
        )
    shapely.prepare(projected_support)
    cells = shapely.box(
        np.tile(x[:-1], grid_size),
        np.repeat(y[:-1], grid_size),
        np.tile(x[1:], grid_size),
        np.repeat(y[1:], grid_size),
    )
    cell_area = float((x[1] - x[0]) * (y[1] - y[0]))
    areas = np.zeros(len(grid))
    full = shapely.covers(projected_support, cells)
    areas[full] = cell_area
    boundary = shapely.intersects(projected_support, cells) & ~full
    clipped_cells = shapely.intersection(cells[boundary], projected_support)
    areas[boundary] = shapely.area(clipped_cells)
    boundary_centers = shapely.get_coordinates(shapely.centroid(clipped_cells))
    nonempty = ~shapely.is_empty(clipped_cells)
    grid[np.flatnonzero(boundary)[nonempty]] = boundary_centers
    supported = (areas > 1e-8).reshape(xx.shape)
    if not supported.any():
        raise ValueError("interest_land_support_empty")
    # Separable Gaussians avoid constructing a grid-by-candidate distance matrix.
    kx = np.exp(-(((x[:-1] + x[1:])[:, None] / 2 - projected[None, :, 0]) ** 2) / (2 * bandwidth_m**2))
    ky = np.exp(-(((y[:-1] + y[1:])[:, None] / 2 - projected[None, :, 1]) ** 2) / (2 * bandwidth_m**2))
    edge_indices = np.flatnonzero(boundary & (areas > 1e-8))
    edge_distance = np.sum((grid[edge_indices, None, :] - projected[None, :, :]) ** 2, axis=2)
    edge_kernels = np.exp(-edge_distance / (2 * bandwidth_m**2))
    nominal_edges = ky[edge_indices // grid_size] * kx[edge_indices % grid_size]
    totals = np.einsum("yx,yi,xi->i", areas.reshape(xx.shape), ky, kx, optimize=False)
    totals += np.einsum("ij,i->j", edge_kernels - nominal_edges, areas[edge_indices])
    totals = np.maximum(totals, 0)
    # Never give a distant Gaussian tail the entire mass after land clipping.
    supported_points = shapely.covers(geographic_support, shapely.points(points))
    nearest_distance = cKDTree(grid[supported.ravel()]).query(projected)[0]
    supported_points &= nearest_distance <= 3 * bandwidth_m
    supported_points &= totals > 0
    restricted_draws = draws * supported_points[:, None]
    draw_totals = np.einsum("i,ij->j", totals, restricted_draws)
    if np.any(draw_totals <= 0):
        raise ValueError("interest_supported_probability_empty")
    weights = (restricted_draws / draw_totals).mean(axis=1)
    field = np.einsum("yi,xi,i->yx", ky, kx, weights, optimize=False)
    field.ravel()[edge_indices] = np.einsum("ij,j->i", edge_kernels, weights)
    masses = field * areas.reshape(xx.shape)
    masses /= masses.sum()
    density = np.divide(masses.ravel(), areas, out=np.zeros_like(areas), where=areas > 0).reshape(xx.shape) * 1_000_000
    order = np.argsort(-density.ravel(), kind="stable")
    cumulative = np.cumsum(masses.ravel()[order])
    regions = []
    for level in MASS_LEVELS:
        threshold = float(density.ravel()[order[min(np.searchsorted(cumulative, level), len(order) - 1)]])
        mask = supported & (density >= threshold)
        ties = supported & (density == threshold)
        regions.append(
            {
                "type": "Feature",
                "properties": {
                    "mass_level": level,
                    "achieved_mass": float(masses[mask].sum()),
                    "threshold_density_per_km2": threshold,
                    "boundary_cell_mass": float(masses[ties].max()),
                    "tie_cell_count": int(np.count_nonzero(ties)),
                    "threshold_tie_mass": float(masses[ties].sum()),
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": _clipped_polygons(mask, x, y, center, geographic_support),
                },
            }
        )
    # Fixed physical density scale, independent of zoom and viewport.
    density_reference = 1_000_000 / (2 * math.pi * bandwidth_m**2)
    # A fixed logarithmic scale retains low-density detail as the catalog grows.
    fractions = np.log1p(1000 * density / density_reference) / math.log1p(1000)
    bins = np.minimum(11, np.floor(fractions * 12)).astype(int)
    bins[masses == 0] = -1
    surfaces = []
    for band in range(12):
        mask = supported & (bins == band)
        if mask.any():
            surfaces.append(
                {
                    "type": "Feature",
                    "properties": {
                        "density_fraction": (band + 0.5) / 12,
                    },
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": _clipped_polygons(mask, x, y, center, geographic_support),
                    },
                }
            )
    return {
        "status": "ready",
        "algorithm_id": "masked-posterior-mixture-partial-cells-v2",
        "crs": "local_lambert_azimuthal_equal_area_sphere",
        "projection_center": center.tolist(),
        "bandwidth_m": bandwidth_m,
        "grid_size": grid_size,
        "cell_area_m2": cell_area,
        "support_clipping": "polygon_intersection_partial_cell_area",
        "mixture_normalization": "per_draw_global_after_support_clipping",
        "support_domain_sha256": manifest["sha256"],
        "support_source": manifest.get("source", "provided support"),
        "support_note": "Land/water clipping; nearby terrain and itinerary feasibility are not certified.",
        "normalization_domain": "supported_route_context_candidates",
        "supported_candidate_count": int(supported_points.sum()),
        "excluded_candidate_count": int((~supported_points).sum()),
        "candidate_support": supported_points.tolist(),
        "integrated_mass": float(masses.sum()),
        "mass_levels": list(MASS_LEVELS),
        "density_reference_per_km2": density_reference,
        "color_scale_id": "fixed_log1p_1000_gaussian_peak_v1",
        "density_geojson": {"type": "FeatureCollection", "features": surfaces},
        "regions_geojson": {"type": "FeatureCollection", "features": regions},
    }
