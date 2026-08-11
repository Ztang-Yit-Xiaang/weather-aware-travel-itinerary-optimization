from __future__ import annotations

import ast
from pathlib import Path

from itinerary_system import (
    blueprint_core,
    blueprint_day_plans,
    blueprint_render_layers,
    blueprint_render_panels,
    blueprint_render_primitives,
    blueprint_renderer,
)
from itinerary_system._legacy import import_legacy_module

REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_BLUEPRINT_CALLABLES = frozenset(
    {
        "_city_hotel_catalog",
        "_city_poi_catalog",
        "_coerce_days_by_city",
        "_profile_config",
        "_select_hotel_for_city_plan",
        "_sequence_between",
        "_stanford_transition_stop",
        "_transition_drive_minutes",
        "_trip_sequence_with_pass_through",
        "_unique_in_order",
    }
)

RENDERER_MODULES = (
    blueprint_day_plans,
    blueprint_render_primitives,
    blueprint_render_layers,
    blueprint_render_panels,
    blueprint_renderer,
)

RENDERER_CONSTANTS = frozenset(
    {
        "DAY_COLORS",
        "FASTEST_ROUTE_COLOR",
        "FULL_SCENE_OFFSET_LAT_STEP",
        "FULL_SCENE_OFFSET_LON_STEP",
        "MATRIX_DEFAULT_ROUTE_KEY",
        "MATRIX_METHOD_STYLES",
        "MATRIX_PROFILE_STYLE_OFFSETS",
        "MATRIX_ROUTE_OFFSETS_ENABLED",
        "ROUTE_CONTEXT_PANE",
        "ROUTE_CORE_PANE",
        "ROUTE_TOP_PANE",
        "SCENIC_CA1_COLOR",
    }
)

LEGACY_ONLY_CALLABLES = frozenset(
    {
        "_add_browser_road_routing",
        "_add_comparison_route_layer",
        "_add_full_scene_persistence_guard",
        "_add_layer_control_overlap_guard",
        "_california_corridor_path",
        "_catalog_lookup_for_names",
        "_city_detail_layer_label",
        "_first_existing",
        "_parse_name_list",
        "_select_greedy_day_stops",
        "_select_greedy_pass_through_day_stops",
        "build_greedy_baseline_day_plan",
    }
)


def test_live_legacy_blueprint_surface_is_explicit_and_callable():
    blueprint = import_legacy_module("blueprint_trip_map")

    missing = sorted(name for name in CORE_BLUEPRINT_CALLABLES if not callable(getattr(blueprint, name, None)))

    assert missing == []
    assert all(getattr(blueprint, name) is getattr(blueprint_core, name) for name in CORE_BLUEPRINT_CALLABLES)
    assert callable(blueprint.build_production_trip_map)
    assert not hasattr(blueprint_core, "build_production_trip_map")
    assert blueprint.CITY_COORDS is blueprint_core.CITY_COORDS
    assert isinstance(blueprint.CITY_COORDS, dict)
    assert blueprint.CITY_COORDS


def test_renderer_surface_is_package_owned_and_legacy_aliases_are_identical():
    blueprint = import_legacy_module("blueprint_trip_map")
    renderer_callables = {
        name: value
        for module in RENDERER_MODULES
        for name, value in vars(module).items()
        if callable(value) and getattr(value, "__module__", "") == module.__name__
    }

    assert len(renderer_callables) == 57
    assert set(renderer_callables).isdisjoint(LEGACY_ONLY_CALLABLES)
    assert all(getattr(blueprint, name) is value for name, value in renderer_callables.items())
    assert all(
        getattr(blueprint, name) is getattr(blueprint_render_primitives, name)
        for name in RENDERER_CONSTANTS
    )
    assert blueprint.build_production_trip_map is blueprint_renderer.build_production_trip_map


def test_notebook_facade_and_package_sources_enforce_the_renderer_boundary():
    notebook_path = REPO_ROOT / "notebook" / "blueprint_trip_map.py"
    notebook_tree = ast.parse(notebook_path.read_text(encoding="utf-8"))
    notebook_defs = {
        node.name
        for node in notebook_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert notebook_defs == LEGACY_ONLY_CALLABLES

    map_renderer_source = (REPO_ROOT / "src" / "itinerary_system" / "map_renderer.py").read_text(
        encoding="utf-8"
    )
    assert 'import_legacy_module("blueprint_trip_map")' not in map_renderer_source
    assert "blueprint_renderer.build_production_trip_map(" in map_renderer_source

    package_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "src" / "itinerary_system").glob("*.py")
    )
    assert 'import_legacy_module("blueprint_trip_map")' not in package_sources