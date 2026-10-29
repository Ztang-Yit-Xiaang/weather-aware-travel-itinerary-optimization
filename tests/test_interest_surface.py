from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath

from itinerary_system.interest_surface import (
    _project,
    _raster_polygons,
    build_interest_surface,
    land_mask,
    load_land_support,
)


def rectangle(west, south, east, north):
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def support():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "land"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [rectangle(-0.06, -0.04, -0.004, 0.04)],
                        [rectangle(0.004, -0.04, 0.06, 0.04)],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"kind": "water"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        rectangle(0.025, 0.006, 0.035, 0.016),
                    ],
                },
            },
        ],
    }


MANIFEST = {"sha256": "a" * 64, "coverage_bbox": [-0.1, -0.1, 0.1, 0.1], "source": "synthetic geometry"}


def contains(shape, points):
    inside = np.zeros(len(points), dtype=bool)
    for polygon in shape["coordinates"]:
        selected = PolygonPath(polygon[0]).contains_points(points)
        for hole in polygon[1:]:
            selected &= ~PolygonPath(hole).contains_points(points)
        inside |= selected
    return inside


def test_equal_area_projection_round_trip_including_origin():
    origin = np.array([-120.0, 36.0])
    points = np.array([origin, [-119.7, 34.4], [-122.4, 37.8]])
    np.testing.assert_allclose(_project(_project(points, origin), origin, inverse=True), points, atol=1e-10)


def test_mass_regions_are_nested_disconnected_and_exclude_water():
    points = np.array([[-0.02, 0.0], [0.02, 0.0]])
    result = build_interest_surface(
        points, np.full((2, 64), 0.5), geometry=support(), manifest=MANIFEST, bandwidth_m=750
    )
    assert abs(result["integrated_mass"] - 1) < 1e-12
    gx, gy = np.meshgrid(np.linspace(-0.055, 0.055, 120), np.linspace(-0.035, 0.035, 90))
    probe = np.column_stack((gx.ravel(), gy.ravel()))
    land = land_mask(probe, support(), MANIFEST["coverage_bbox"])
    previous = np.zeros(len(probe), dtype=bool)
    for feature in result["regions_geojson"]["features"]:
        props = feature["properties"]
        assert (
            props["mass_level"]
            <= props["achieved_mass"]
            <= props["mass_level"] + props["boundary_cell_mass"] * props["tie_cell_count"] + 1e-12
        )
        inside = contains(feature["geometry"], probe)
        assert not np.any(inside & ~land)
        assert not np.any(previous & ~inside)
        assert len(feature["geometry"]["coordinates"]) >= 2
        previous = inside
    assert result == build_interest_surface(
        points, np.full((2, 64), 0.5), geometry=support(), manifest=MANIFEST, bandwidth_m=750
    )


def test_cell_union_preserves_holes_and_diagonal_components():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    polygon = _raster_polygons(mask, np.arange(6.0), np.arange(6.0), np.array([0.0, 0.0]))
    assert len(polygon) == 1 and len(polygon[0]) == 2
    diagonal = _raster_polygons(np.eye(3, dtype=bool), np.arange(4.0), np.arange(4.0), np.array([0.0, 0.0]))
    assert len(diagonal) == 3


def test_restricted_terrain_is_removed_and_unknown_coverage_does_not_fill():
    geometry = support()
    geometry["features"].append(
        {
            "type": "Feature",
            "properties": {"kind": "restricted"},
            "geometry": {"type": "Polygon", "coordinates": [rectangle(-0.03, -0.01, -0.01, 0.01)]},
        }
    )
    points = np.array([[-0.02, 0], [0, 0], [0.02, 0], [0.03, 0.01], [1.0, 1.0]])
    np.testing.assert_array_equal(
        land_mask(points, geometry, MANIFEST["coverage_bbox"]), [False, False, True, False, False]
    )
    with pytest.raises(ValueError, match="support_empty|probability_empty"):
        build_interest_surface(np.array([[1.0, 1.0]]), np.array([[0.5]]), geometry=geometry, manifest=MANIFEST)


def test_zero_probability_has_explicit_failure():
    with pytest.raises(ValueError, match="probability_empty"):
        build_interest_surface(np.array([[-0.02, 0]]), np.zeros((1, 32)), geometry=support(), manifest=MANIFEST)


def test_coastal_point_is_retained_without_extending_the_region_into_water():
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "land"},
                "geometry": {"type": "Polygon", "coordinates": [rectangle(0, -0.04, 0.07, 0.04)]},
            }
        ],
    }
    points = np.array([[0.00005, 0], [0.04, 0]])
    result = build_interest_surface(
        points, np.full((2, 32), 0.5), geometry=geometry, manifest=MANIFEST, bandwidth_m=500, grid_size=32
    )
    assert result["support_clipping"] == "polygon_intersection_partial_cell_area"
    assert contains(result["regions_geojson"]["features"][-1]["geometry"], points).all()
    gx, gy = np.meshgrid(np.linspace(-0.01, 0.06, 151), np.linspace(-0.02, 0.02, 121))
    probe = np.column_stack((gx.ravel(), gy.ravel()))
    for feature in result["density_geojson"]["features"] + result["regions_geojson"]["features"]:
        assert not np.any(contains(feature["geometry"], probe) & (probe[:, 0] < 0))


def test_clipped_posterior_mixture_mass_matches_independent_spatial_integration():
    from scipy.stats import qmc

    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "land"},
                "geometry": {"type": "Polygon", "coordinates": [rectangle(0, -0.04, 0.07, 0.04)]},
            }
        ],
    }
    points = np.array([[0.00005, 0], [0.04, 0]])
    draws = np.array([[0.9, 0.05], [0.1, 0.95]])
    width = 500
    result = build_interest_surface(
        points, draws, geometry=geometry, manifest=MANIFEST, bandwidth_m=width, grid_size=96
    )
    center = points.mean(axis=0)
    projected = _project(points, center)
    low, high = projected.min(axis=0) - 3 * width, projected.max(axis=0) + 3 * width
    probe = low + qmc.Sobol(2, scramble=True, seed=42).random_base2(18) * (high - low)
    geographic = _project(probe, center, inverse=True)
    support_mask = land_mask(geographic, geometry, MANIFEST["coverage_bbox"])
    # This oracle evaluates the posterior draws directly at independent samples.
    # It neither reuses raster cell weights nor normalizes individual kernels.
    kernels = np.exp(-np.sum((probe[:, None, :] - projected[None, :, :]) ** 2, axis=2) / (2 * width**2))
    unnormalized = (kernels @ draws) * support_mask[:, None]
    sampled_mass = (unnormalized / unnormalized.sum(axis=0)).mean(axis=1)
    for feature in result["regions_geojson"]["features"]:
        integrated = sampled_mass[contains(feature["geometry"], geographic)].sum()
        assert abs(integrated - feature["properties"]["achieved_mass"]) < 0.012, (
            integrated,
            feature["properties"]["achieved_mass"],
        )


def test_sparse_catalog_keeps_multiple_density_shades_on_a_fixed_scale():
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "land"},
                "geometry": {"type": "Polygon", "coordinates": [rectangle(-1, -1, 1, 1)]},
            }
        ],
    }
    gx, gy = np.meshgrid(np.linspace(-0.16, 0.16, 5), np.linspace(-0.12, 0.12, 4))
    points = np.column_stack((gx.ravel(), gy.ravel()))
    manifest = {**MANIFEST, "coverage_bbox": [-1, -1, 1, 1]}
    result = build_interest_surface(points, np.full((20, 32), 0.5), geometry=geometry, manifest=manifest)
    assert result["color_scale_id"] == "fixed_log1p_1000_gaussian_peak_v1"
    assert len(result["density_geojson"]["features"]) >= 4
    fractions = [feature["properties"]["density_fraction"] for feature in result["density_geojson"]["features"]]
    assert fractions == sorted(fractions)


def test_static_support_cache_is_immutable_and_rechecks_source_hash(tmp_path):
    import hashlib
    import json

    directory = tmp_path / "configs/interest_regions"
    directory.mkdir(parents=True)
    payload = json.dumps(support()).encode()
    path = directory / "land_support.geojson"
    path.write_bytes(payload)
    manifest = {
        **MANIFEST,
        "schema_version": "interest-land-support-v1",
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    first, _ = load_land_support(tmp_path)
    second, _ = load_land_support(tmp_path)
    assert first is second
    with pytest.raises(TypeError):
        first["features"][0]["properties"]["kind"] = "water"
    path.write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_land_support(tmp_path)
