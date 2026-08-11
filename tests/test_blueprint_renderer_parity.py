from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import math
import re
import shutil
from pathlib import Path

import pandas as pd

from itinerary_system import blueprint_render_panels as panels
from itinerary_system import blueprint_render_primitives as primitives
from itinerary_system import blueprint_renderer, map_renderer
from itinerary_system import blueprint_renderer_sections as sections
from itinerary_system import blueprint_route_selector as selector

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUTPUTS = REPO_ROOT / "results" / "outputs"

# E3.C4 reviewed baseline: the only change from E3.C3 is the mobile-only
# selector containment rule. Data, route-debug, and Leaflet contracts stay frozen.
EXPECTED_HTML_SHA256 = "a06583549a135688e62d663ff5c6197074e96f3a0bad57d5cb791f37273fc2bb"
EXPECTED_DAY_PLAN_SHA256 = "6bb4a3a40d76a07ba62e02bf055fd40fece8853b09fa76be07cffe16b7f88e27"
EXPECTED_ROUTE_DEBUG_SHA256 = "b723926ceb77887660a7730104e3e2ebc891ca668ab33f4b43fe9d8f9444ff7e"
EXPECTED_LEAFLET_COUNTS = {
    "polyline": 1105,
    "circle_marker": 418,
    "marker": 1173,
    "feature_group": 162,
    "geo_json": 0,
}
EXPECTED_TOKEN_PRESENCE = {
    "Route Selector": True,
    "Balanced": True,
    "Fastest inter-city route": True,
    "Scenic CA-1 / PCH route": True,
    "Must-go candidates": True,
    "Selected hotel/base pins": False,
    "Interest preview": True,
    "Route diagnostics": False,
}


def _normalize_folium_ids(text: str) -> str:
    text = re.sub(r"_[0-9a-f]{32}", "_FOLIUM_ID", text)
    return re.sub(r"(?<![A-Za-z0-9])[0-9a-f]{32}(?![A-Za-z0-9])", "FOLIUM_ID", text)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _baseline_trip() -> dict:
    row = pd.read_csv(SOURCE_OUTPUTS / "production_hierarchical_gurobi_plan.csv").iloc[0]
    return {
        "gateway_start": str(row["gateway_start"]),
        "gateway_end": str(row["gateway_end"]),
        "city_sequence": ast.literal_eval(str(row["city_sequence"])),
        "days_by_city": ast.literal_eval(str(row["days_by_city"])),
        "pass_through_cities": ast.literal_eval(str(row["pass_through_cities"])),
    }


def _copy_renderer_inputs(output_dir: Path) -> None:
    output_dir.mkdir()
    for source in SOURCE_OUTPUTS.iterdir():
        if source.is_file() and source.suffix.lower() in {".csv", ".json", ".jsonl"}:
            shutil.copy2(source, output_dir / source.name)


def test_renderer_geometry_helpers_preserve_empty_duplicate_and_nonfinite_contracts():
    assert primitives._dedupe_route_points([]) == []
    assert primitives._dedupe_route_points(
        [[1, 2], [1, 2], [1.00000001, 2.00000001], [3, 4]]
    ) == [[1.0, 2.0], [3.0, 4.0]]
    assert primitives._route_bounds([]) == []
    assert primitives._route_midpoint([]) is None
    assert primitives._route_arrow_samples([]) == []
    assert math.isnan(primitives._finite_float(None))
    assert math.isnan(primitives._finite_float(float("nan")))
    assert math.isnan(primitives._finite_float(float("inf")))
    assert primitives._finite_float("12.5") == 12.5
    assert primitives._finite_float("bad", default=-1.0) == -1.0
    assert primitives._matrix_route_offset("unknown", "unknown", float("nan")) == 0.0


def test_renderer_offline_route_edges_never_call_network(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("offline renderer attempted a live request")

    monkeypatch.setattr(primitives.requests, "get", fail_network)

    assert primitives._fetch_osrm_route([], {}, run_live=False) == ([], "single-point")
    assert primitives._fetch_osrm_route([[1, 2]], {}, run_live=False) == ([[1, 2]], "single-point")

    points = [[37.0, -122.0], [37.2, -121.8]]
    key = primitives._route_cache_key(points)
    cached_path = [[37.0, -122.0], [37.1, -121.9], [37.2, -121.8]]
    assert primitives._fetch_osrm_route(
        points,
        {key: {"mode": "osrm", "path": cached_path}},
        run_live=False,
    ) == (cached_path, "cached-osrm")

    fallback_path, fallback_mode = primitives._fetch_osrm_route(points, {}, run_live=False)
    assert fallback_path[0] == points[0]
    assert fallback_path[-1] == points[-1]
    assert fallback_mode in {"straight-line-fallback", "curated-corridor-fallback"}


def test_renderer_html_and_missing_artifact_fallbacks_are_safe(tmp_path):
    assert primitives._escape("<script>&") == "&lt;script&gt;&amp;"
    missing = tmp_path / "missing"
    fallback = pd.DataFrame([{"day": 1, "attraction_name": "Fallback"}])
    returned = panels._load_default_hierarchical_gurobi_day_plan(missing, fallback)
    assert returned is fallback
    assert "not generated yet" in panels._build_method_comparison_html(pd.DataFrame())


def test_renderer_state_is_request_scoped_and_section_functions_stay_bounded(tmp_path):
    def state_kwargs(suffix):
        return {
            "context": {},
            "output_path": tmp_path / f"{suffix}.html",
            "output_dir": tmp_path / suffix,
            "cache_path": tmp_path / f"{suffix}.json",
            "route_cache": {},
            "run_live": False,
            "route_debug_rows": [],
            "route_debug_registry": [],
            "trip": {},
            "city_sequence": [],
            "main_trip_days": 0,
            "profile_day_plans": {},
            "day_plan_df": pd.DataFrame(),
            "profile_day_plan_df": pd.DataFrame(),
            "use_route_matrix": False,
            "fastest_sequence": [],
            "scenic_sequence": [],
            "fit_points": [],
            "all_points": [],
            "trip_map": object(),
        }

    first = sections.RendererBuildState(**state_kwargs("first"))
    second = sections.RendererBuildState(**state_kwargs("second"))
    first.fastest_path.append([1.0, 2.0])
    first.comparison_layers["Methods"] = [object()]

    assert second.fastest_path == []
    assert second.comparison_layers == {}
    assert first.route_debug_rows is not second.route_debug_rows
    assert first.route_debug_registry is not second.route_debug_registry

    function_limits = {}
    for module in (blueprint_renderer, sections, selector):
        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_limits[f"{module.__name__}.{node.name}"] = node.end_lineno - node.lineno + 1

    panel_tree = ast.parse(Path(panels.__file__).read_text(encoding="utf-8"))
    route_control = next(
        node
        for node in ast.walk(panel_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_add_route_debug_controls"
    )
    function_limits[f"{panels.__name__}.{route_control.name}"] = route_control.end_lineno - route_control.lineno + 1

    oversized_functions = {name: size for name, size in function_limits.items() if size >= 300}
    assert oversized_functions == {}

    section_tree = ast.parse(Path(sections.__file__).read_text(encoding="utf-8"))
    mutable_module_assignments = [
        node
        for node in section_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
    ]
    assert mutable_module_assignments == []


def test_renderer_orchestrator_has_one_explicit_ordered_section_pipeline():
    tree = ast.parse(Path(blueprint_renderer.__file__).read_text(encoding="utf-8"))
    renderer = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    calls = [
        node.func.id
        for statement in renderer.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls == [
        "_prepare_renderer_state",
        "_add_context_route_section",
        "_add_hotel_layer_section",
        "_add_must_go_layer_section",
        "_add_nature_layer_section",
        "_add_profile_route_section",
        "_add_comparison_layer_section",
        "_add_dashboard_section",
        "_finalize_renderer",
    ]


def test_map_renderer_calls_package_renderer_without_legacy_import(monkeypatch, tmp_path):
    class Config:
        def get(self, *_args):
            return _args[-1]

    class FakeMap:
        def __init__(self):
            self.bounds = None
            self.saved = None

        def fit_bounds(self, points, padding):
            self.bounds = (points, padding)

        def save(self, path):
            self.saved = Path(path)
            self.saved.write_text("<html></html>", encoding="utf-8")

    fake_map = FakeMap()
    day_plan = pd.DataFrame(
        [
            {
                "day": 1,
                "latitude": 34.0,
                "longitude": -118.0,
                "route_start_latitude": 34.0,
                "route_start_longitude": -118.0,
                "route_end_latitude": 34.1,
                "route_end_longitude": -118.1,
            }
        ]
    )
    calls = []

    def fake_build(context, output_path=None, run_live_routing=None):
        calls.append((context, output_path, run_live_routing))
        return fake_map, day_plan, output_path

    monkeypatch.setattr(map_renderer.blueprint_renderer, "build_production_trip_map", fake_build)
    monkeypatch.setattr(map_renderer, "prepare_map_dashboard_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(map_renderer.map_exporter, "export_map_artifacts", lambda *_args, **_kwargs: {})

    output_path = tmp_path / "figures" / "map.html"
    context = {
        "OUTPUT_DIR": tmp_path / "outputs",
        "FIGURE_DIR": output_path.parent,
        "CANONICAL_TRIP_DAYS": 1,
        "CANONICAL_TOTAL_BUDGET": 100.0,
    }
    output_path.parent.mkdir()
    (tmp_path / "outputs").mkdir()

    returned_map, returned_plan, returned_path = map_renderer.build_map(
        context,
        Config(),
        output_path=output_path,
    )

    assert calls and calls[0][1] == output_path
    assert returned_map is fake_map
    assert returned_plan is day_plan
    assert returned_path == output_path
    assert fake_map.saved == output_path

def test_package_renderer_matches_pre_migration_render_signature(tmp_path):
    output_dir = tmp_path / "outputs"
    figure_dir = tmp_path / "figures"
    _copy_renderer_inputs(output_dir)
    figure_dir.mkdir()

    context = {
        "PROJECT_ROOT": REPO_ROOT,
        "OUTPUT_DIR": output_dir,
        "FIGURE_DIR": figure_dir,
        "best_hierarchical_trip": _baseline_trip(),
        "RUN_LIVE_APIS": False,
        "MAP_REFRESH_ROAD_GEOMETRY": False,
        "MAP_ROUTE_ONLY_DEBUG_VIEW": False,
        "MAP_BALANCED_ONLY_DEFAULT_VIEW": True,
        "SHOW_CONTEXT_ROUTES_BY_DEFAULT": False,
        "SHOW_COMPARISON_LAYERS_BY_DEFAULT": False,
        "SHOW_TRAVELER_OVERVIEWS_BY_DEFAULT": False,
        "SHOW_SELECTED_RESULT_BY_DEFAULT": False,
    }
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        _map, day_plan, html_path = blueprint_renderer.build_production_trip_map(
            context,
            output_path=figure_dir / "package_renderer.html",
            run_live_routing=False,
        )

    html = Path(html_path).read_text(encoding="utf-8")
    normalized_html = _normalize_folium_ids(html)
    day_plan_csv = day_plan.to_csv(index=False, lineterminator="\n")
    route_debug = pd.read_csv(output_dir / "production_map_route_debug.csv")
    route_debug_csv = _normalize_folium_ids(route_debug.to_csv(index=False, lineterminator="\n"))

    assert _sha256(normalized_html) == EXPECTED_HTML_SHA256
    assert len(html.encode()) == 8_976_620
    assert len(day_plan) == 7
    assert _sha256(day_plan_csv) == EXPECTED_DAY_PLAN_SHA256
    assert len(route_debug) == 335
    assert _sha256(route_debug_csv) == EXPECTED_ROUTE_DEBUG_SHA256
    assert {
        "polyline": html.count("L.polyline("),
        "circle_marker": html.count("L.circleMarker("),
        "marker": html.count("L.marker("),
        "feature_group": html.count("L.featureGroup("),
        "geo_json": html.count("L.geoJson("),
    } == EXPECTED_LEAFLET_COUNTS
    assert {token: token in html for token in EXPECTED_TOKEN_PRESENCE} == EXPECTED_TOKEN_PRESENCE
