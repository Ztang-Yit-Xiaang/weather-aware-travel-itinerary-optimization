"""Validate lightweight and modular dashboard export artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated dashboard map artifacts.")
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR), help="Directory containing map figures.")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing production output CSVs."
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_js_assignment(path: Path, marker: str) -> Any:
    text = path.read_text(encoding="utf-8")
    index = text.find(marker)
    if index < 0:
        raise ValueError(f"missing JS marker: {marker}")
    start = index + len(marker)
    return json.JSONDecoder().raw_decode(text[start:].lstrip())[0]


def read_js_global(path: Path, global_name: str) -> Any:
    return read_js_assignment(path, f"window.{global_name} = ")


def read_js_global_map(path: Path, global_name: str, key: str) -> Any:
    key_text = json.dumps(key)
    return read_js_assignment(path, f"window.{global_name}[{key_text}] = ")


def check_exists(path: Path, label: str, errors: list[str]) -> bool:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        errors.append(f"Empty {label}: {path}")
        return False
    return True


def check_relative_asset(path_text: str, source: Path, errors: list[str]) -> bool:
    candidate = Path(path_text)
    if candidate.is_absolute():
        errors.append(f"Asset path must be relative, got absolute path in {source}: {path_text}")
        return False
    if ".." in candidate.parts:
        errors.append(f"Asset path must not escape dashboard folder in {source}: {path_text}")
        return False
    return True


def route_has_geometry(route_data: Any) -> bool:
    if isinstance(route_data, dict):
        features = route_data.get("features")
        if isinstance(features, list) and features:
            return True
        geometry = route_data.get("geometry")
        if isinstance(geometry, dict) and geometry.get("coordinates"):
            return True
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def validate(figure_dir: Path, output_dir: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    dashboard_root = figure_dir / "full_interactive_dashboard"
    assets = dashboard_root / "assets"

    lightweight = figure_dir / "lightweight_share_map.html"
    index_html = dashboard_root / "index.html"
    research_html = dashboard_root / "research.html"
    customer_html = dashboard_root / "customer.html"
    style_css = assets / "style.css"
    map_js = assets / "map_controls.js"
    dashboard_js = assets / "dashboard.js"
    data_loader_js = assets / "data_loader.js"
    route_index_path = assets / "route_index.json"
    route_index_js_path = assets / "route_index.js"
    metrics_path = assets / "dashboard_metrics.json"
    metrics_js_path = assets / "dashboard_metrics.js"
    debug_path = assets / "debug_summary.json"
    debug_js_path = assets / "debug_summary.js"
    interest_path = assets / "interest_preview.json"
    interest_js_path = assets / "interest_preview.js"
    playback_path = assets / "playback_data.json"
    playback_js_path = assets / "playback_data.js"
    city_path = assets / "city_details.json"
    city_js_path = assets / "city_details.js"
    selected_hotels_path = assets / "selected_hotels.json"
    selected_hotels_js_path = assets / "selected_hotels.js"
    hotel_path = assets / "hotel_choices.json"
    hotel_js_path = assets / "hotel_choices.js"
    nature_path = assets / "nature_explore.json"
    nature_js_path = assets / "nature_explore.js"
    evaluation_html = dashboard_root / "evaluation.html"
    evaluation_path = assets / "evaluation_metrics.json"
    evaluation_js_path = assets / "evaluation_metrics.js"
    size_report = output_dir / "production_map_artifact_size_report.csv"

    for path, label in [
        (lightweight, "lightweight share map"),
        (index_html, "full dashboard index"),
        (research_html, "research dashboard HTML"),
        (customer_html, "customer dashboard HTML"),
        (style_css, "dashboard CSS"),
        (map_js, "map controls JS"),
        (dashboard_js, "dashboard JS"),
        (data_loader_js, "dashboard data loader JS"),
        (route_index_path, "route index JSON"),
        (route_index_js_path, "route index JS fallback"),
        (metrics_path, "dashboard metrics JSON"),
        (metrics_js_path, "dashboard metrics JS fallback"),
        (debug_path, "debug summary JSON"),
        (debug_js_path, "debug summary JS fallback"),
        (interest_path, "interest preview JSON"),
        (interest_js_path, "interest preview JS fallback"),
        (playback_path, "playback data JSON"),
        (playback_js_path, "playback data JS fallback"),
        (city_path, "city details JSON"),
        (city_js_path, "city details JS fallback"),
        (selected_hotels_path, "selected hotel JSON"),
        (selected_hotels_js_path, "selected hotel JS fallback"),
        (hotel_path, "hotel choices JSON"),
        (hotel_js_path, "hotel choices JS fallback"),
        (nature_path, "nature explore JSON"),
        (nature_js_path, "nature explore JS fallback"),
        (evaluation_html, "evaluation dashboard HTML"),
        (evaluation_path, "evaluation metrics JSON"),
        (evaluation_js_path, "evaluation metrics JS fallback"),
    ]:
        check_exists(path, label, errors)

    if lightweight.exists():
        html = lightweight.read_text(encoding="utf-8", errors="replace")
        if "L.map" not in html:
            errors.append(f"Lightweight map does not initialize Leaflet with L.map: {lightweight}")
        if "share-map-data" not in html:
            errors.append(f"Lightweight map does not embed selected route data: {lightweight}")

    if index_html.exists():
        html = index_html.read_text(encoding="utf-8", errors="replace")
        if '<div id="map"' not in html:
            errors.append(f'Dashboard index is missing <div id="map">: {index_html}')
        if "diagnostic-panel" not in html:
            errors.append(f"Dashboard index is missing diagnostic panel: {index_html}")
        if "assets/data_loader.js" not in html:
            errors.append(f"Dashboard index is missing assets/data_loader.js: {index_html}")
        if "Load optional layers" in html:
            errors.append(f"Dashboard index contains stale milestone-0 optional-layer button: {index_html}")
        if "alert(" in html:
            errors.append(f"Dashboard index should not use alert() for milestone-1 dashboard behavior: {index_html}")
        for token in ["Saved optimized route", "Preview only route", "evaluation.html"]:
            if token not in html:
                errors.append(f"Dashboard index missing required truth-first UI label {token!r}: {index_html}")
        for token in [
            'data-map-layer-toggle="terrain"',
            'data-map-layer-toggle="roads"',
            'data-map-layer-toggle="labels"',
            'data-map-layer-toggle="weather_risk"',
        ]:
            if token not in html:
                errors.append(f"Dashboard index missing wired upper-right map layer control {token!r}: {index_html}")
        if "checked readonly" in html:
            errors.append(f"Dashboard index still contains decorative readonly layer checkboxes: {index_html}")
        for token in ["dashboard-mode-toggle", 'data-mode-section="customer"', 'data-mode-section="research"']:
            if token not in html:
                errors.append(f"Dashboard index missing one-page mode switch token {token!r}: {index_html}")

    if research_html.exists():
        html = research_html.read_text(encoding="utf-8", errors="replace")
        for token in [
            'data-dashboard-mode="research"',
            "Research/Test dashboard",
            "dashboard-mode-toggle",
            "customer-trip-controls",
            "route-selector",
            "evaluation.html",
            "Debug summary",
        ]:
            if token not in html:
                errors.append(f"Research dashboard missing required research-mode token {token!r}: {research_html}")

    if customer_html.exists():
        html = customer_html.read_text(encoding="utf-8", errors="replace")
        for token in [
            'data-dashboard-mode="customer"',
            "Customer trip planner",
            "dashboard-mode-toggle",
            "Plan your trip",
            "customer-trip-controls",
            "Trip summary",
            'data-mode-section="research"',
        ]:
            if token not in html:
                errors.append(f"Customer dashboard missing required customer-mode token {token!r}: {customer_html}")

    if evaluation_html.exists():
        html = evaluation_html.read_text(encoding="utf-8", errors="replace")
        for token in ["Method Evaluation Dashboard", "assets/evaluation_metrics.js", "Method Comparison", "Solver Tradeoffs"]:
            if token not in html:
                errors.append(f"Evaluation dashboard missing required token {token!r}: {evaluation_html}")

    if style_css.exists():
        css = style_css.read_text(encoding="utf-8", errors="replace")
        if "#map" not in css or "height: 100vh" not in css:
            errors.append(f"Dashboard CSS does not give #map a full viewport height: {style_css}")

    if map_js.exists():
        js = map_js.read_text(encoding="utf-8", errors="replace")
        for token in [
            'L.map("map"',
            "drawDefaultRoute",
            "renderRouteSelector",
            "toggleRoute",
            "runQuickAction",
            "loadOptionalLayers",
            "zoomVisibleRoutes",
            "playRouteAnimation",
            "pauseRouteAnimation",
            "setPlaybackStop",
            "drawSelectedHotels",
            "toggleSelectedHotels",
            "toggleHotelCandidates",
            "drawLivePreviewRoute",
            "clearLivePreviewRoute",
            "Preview only - rerun pipeline to save",
            "initializeBaseMapLayers",
            "bindMapLayerToggles",
            "buildLabelLayer",
            "buildWeatherRiskLayer",
            "data-map-layer-toggle",
            "numbered-route-stop",
            "renderCustomerControls(index)",
            "window.dashboardMap",
            "dashboardLogError",
        ]:
            if token not in js:
                errors.append(f"Dashboard JS missing required diagnostic/init token {token!r}: {map_js}")
        if "alert(" in js:
            errors.append(f"Dashboard JS should not use alert() for normal optional-layer status: {map_js}")

    if dashboard_js.exists():
        js = dashboard_js.read_text(encoding="utf-8", errors="replace")
        for token in [
            "renderActiveStopDetail",
            "renderCityDetails",
            "renderHotelChoices",
            "setPreferredHotel",
            "window.localStorage",
            "renderNatureExplore",
            "data-show-nature-layer",
            "bindDashboardShellControls",
            "normalizeInterestBarValues",
            "data-interest-axis",
            "previewScore",
            "Build live preview route",
            "buildLivePreviewRoute",
            "window.setTimeout(buildLivePreviewRoute",
            "Browser preview is approximate",
            "data-toggle-hotel-candidates",
            "renderCustomerControls",
            "customer_visible",
            "dashboard-mode-toggle",
            "applyDashboardMode",
        ]:
            if token not in js:
                errors.append(f"Dashboard panel JS missing required V2 token {token!r}: {dashboard_js}")
        if "alert(" in js:
            errors.append(f"Dashboard panel JS should not use alert(): {dashboard_js}")

    if data_loader_js.exists():
        js = data_loader_js.read_text(encoding="utf-8", errors="replace")
        for token in [
            "dashboardIsFileMode",
            "loadScriptOnce",
            "loadDashboardAsset",
            "loadRouteIndex",
            "loadDashboardMetrics",
            "loadDebugSummary",
            "loadInterestPreview",
            "loadPlaybackData",
            "loadCityDetails",
            "loadSelectedHotels",
            "loadHotelChoices",
            "loadNatureExplore",
            "loadRouteGeoJson",
            "loadPoiJson",
            "[dashboard-loader]",
            "[dashboard-assets]",
            "[dashboard-error]",
        ]:
            if token not in js:
                errors.append(f"Data loader missing required token {token!r}: {data_loader_js}")

    route_index: dict[str, Any] = {}
    if route_index_path.exists():
        try:
            route_index = read_json(route_index_path)
        except Exception as exc:
            errors.append(f"Invalid route_index.json: {route_index_path}: {exc}")
        if isinstance(route_index, dict):
            if route_index.get("contract_version") != "core-route-index-v1":
                errors.append(
                    f"route_index.json must declare contract_version='core-route-index-v1': {route_index_path}"
                )
            for stale_key in ["default_route", "pois"]:
                if stale_key in route_index:
                    errors.append(f"route_index.json contains stale top-level key {stale_key!r}: {route_index_path}")
            routes = route_index.get("routes", [])
            if not routes:
                errors.append(f"route_index.json must contain at least one route: {route_index_path}")
            if not any(isinstance(route, dict) and route.get("default") for route in routes):
                errors.append(f"route_index.json must contain at least one default route: {route_index_path}")

            for route in routes:
                if not isinstance(route, dict):
                    errors.append(f"Route index record must be an object: {route_index_path}")
                    continue
                route_id = str(route.get("id", ""))
                if "path" in route:
                    errors.append(f"Route {route_id or '<missing id>'} contains stale key 'path': {route_index_path}")
                for required_key in [
                    "id",
                    "label",
                    "default",
                    "optional",
                    "family",
                    "selector_group",
                    "geojson",
                    "geojson_js",
                    "pois",
                    "pois_js",
                    "customer_visible",
                    "research_only",
                    "customer_control_group",
                ]:
                    if required_key not in route:
                        errors.append(f"Route {route_id or '<missing id>'} missing {required_key}: {route_index_path}")

                route_path_text = str(route.get("geojson", ""))
                route_js_text = str(route.get("geojson_js", ""))
                for path_text, label in [(route_path_text, "route GeoJSON"), (route_js_text, "route JS fallback")]:
                    if not check_relative_asset(path_text, route_index_path, errors):
                        continue
                    if not check_exists(dashboard_root / path_text, label, errors):
                        continue
                route_path = dashboard_root / route_path_text
                route_js_path = dashboard_root / route_js_text
                if route_path.exists():
                    try:
                        route_data = read_json(route_path)
                        if not route_has_geometry(route_data):
                            errors.append(f"Route GeoJSON has no features or coordinates: {route_path}")
                    except Exception as exc:
                        errors.append(f"Invalid route GeoJSON JSON: {route_path}: {exc}")
                if route_js_path.exists() and route_id:
                    try:
                        route_data = read_js_global_map(route_js_path, "DASHBOARD_ROUTES", route_id)
                        if not route_has_geometry(route_data):
                            errors.append(f"Route JS fallback has no features or coordinates: {route_js_path}")
                    except Exception as exc:
                        errors.append(f"Invalid route JS fallback: {route_js_path}: {exc}")

                poi_path_text = str(route.get("pois", ""))
                poi_js_text = str(route.get("pois_js", ""))
                poi_id = str(route.get("pois_id") or route_id)
                for path_text, label in [(poi_path_text, "POI JSON"), (poi_js_text, "POI JS fallback")]:
                    if not check_relative_asset(path_text, route_index_path, errors):
                        continue
                    if not check_exists(dashboard_root / path_text, label, errors):
                        continue
                poi_path = dashboard_root / poi_path_text
                poi_js_path = dashboard_root / poi_js_text
                is_default_route = bool(route.get("default"))
                if poi_path.exists():
                    try:
                        poi_data = read_json(poi_path)
                        if not isinstance(poi_data, list):
                            errors.append(f"POI JSON should be a list: {poi_path}")
                        elif not poi_data:
                            warnings.append(f"POI JSON is empty: {poi_path}")
                        elif is_default_route and not any(point.get("lat") and point.get("lon") for point in poi_data):
                            errors.append(f"Default route POI JSON has no playable stop coordinates: {poi_path}")
                        elif is_default_route and not any(
                            safe_float(point.get("display_utility")) > 0 for point in poi_data
                        ):
                            errors.append(f"Default route POI JSON has no nonzero display_utility: {poi_path}")
                        if is_default_route and poi_data:
                            required_stop_fields = {
                                "description",
                                "why_selected",
                                "expected_duration_minutes",
                                "weather_risk",
                                "source_confidence",
                                "type",
                                "city_or_anchor",
                            }
                            missing = required_stop_fields.difference(poi_data[0])
                            if missing:
                                errors.append(f"Default route POI JSON missing rich stop fields {sorted(missing)}: {poi_path}")
                    except Exception as exc:
                        errors.append(f"Invalid POI JSON: {poi_path}: {exc}")
                if poi_js_path.exists():
                    try:
                        poi_data = read_js_global_map(poi_js_path, "DASHBOARD_POIS", poi_id)
                        if not isinstance(poi_data, list):
                            errors.append(f"POI JS fallback should define a list: {poi_js_path}")
                    except Exception as exc:
                        errors.append(f"Invalid POI JS fallback: {poi_js_path}: {exc}")

    if route_index_js_path.exists():
        try:
            js_index = read_js_global(route_index_js_path, "DASHBOARD_ROUTE_INDEX")
            if not isinstance(js_index, dict) or not js_index.get("routes"):
                errors.append(f"route_index.js fallback does not define routes: {route_index_js_path}")
            elif route_index and js_index != route_index:
                errors.append(f"route_index.js fallback disagrees with route_index.json: {route_index_js_path}")
        except Exception as exc:
            errors.append(f"Invalid route_index.js fallback: {route_index_js_path}: {exc}")

    if metrics_path.exists():
        try:
            metrics_json = read_json(metrics_path)
        except Exception as exc:
            errors.append(f"Invalid dashboard_metrics.json: {metrics_path}: {exc}")
        else:
            if not isinstance(metrics_json, dict):
                errors.append(f"dashboard_metrics.json should define an object: {metrics_path}")
    if metrics_js_path.exists():
        try:
            metrics_js = read_js_global(metrics_js_path, "DASHBOARD_METRICS")
            if metrics_path.exists():
                try:
                    if metrics_js != read_json(metrics_path):
                        errors.append(f"dashboard_metrics.js disagrees with dashboard_metrics.json: {metrics_js_path}")
                except Exception:
                    pass
        except Exception as exc:
            errors.append(f"Invalid dashboard_metrics.js fallback: {metrics_js_path}: {exc}")

    for json_path, js_path, global_name, label in [
        (debug_path, debug_js_path, "DASHBOARD_DEBUG_SUMMARY", "debug summary"),
        (interest_path, interest_js_path, "DASHBOARD_INTEREST_PREVIEW", "interest preview"),
        (playback_path, playback_js_path, "DASHBOARD_PLAYBACK_DATA", "playback data"),
        (city_path, city_js_path, "DASHBOARD_CITY_DETAILS", "city details"),
        (selected_hotels_path, selected_hotels_js_path, "DASHBOARD_SELECTED_HOTELS", "selected hotels"),
        (hotel_path, hotel_js_path, "DASHBOARD_HOTEL_CHOICES", "hotel choices"),
        (nature_path, nature_js_path, "DASHBOARD_NATURE_EXPLORE", "nature explore"),
        (evaluation_path, evaluation_js_path, "DASHBOARD_EVALUATION_METRICS", "evaluation metrics"),
    ]:
        if json_path.exists():
            try:
                json_data = read_json(json_path)
            except Exception as exc:
                errors.append(f"Invalid {label} JSON: {json_path}: {exc}")
                json_data = None
        else:
            json_data = None
        if js_path.exists():
            try:
                js_data = read_js_global(js_path, global_name)
                if json_data is not None and js_data != json_data:
                    errors.append(f"{js_path.name} disagrees with {json_path.name}: {js_path}")
            except Exception as exc:
                errors.append(f"Invalid {label} JS fallback: {js_path}: {exc}")

    if playback_path.exists():
        try:
            playback_data = read_json(playback_path)
            route_stops = next(iter((playback_data.get("routes") or {}).values()), {}).get("stops", [])
            if route_stops:
                required_playback_fields = {
                    "description",
                    "why_selected",
                    "expected_duration_minutes",
                    "weather_risk",
                    "source_confidence",
                    "type",
                    "city_or_anchor",
                }
                missing = required_playback_fields.difference(route_stops[0])
                if missing:
                    errors.append(f"Playback stop records missing rich fields {sorted(missing)}: {playback_path}")
        except Exception as exc:
            errors.append(f"Invalid playback rich-stop check: {playback_path}: {exc}")

    if evaluation_path.exists():
        try:
            evaluation_data = read_json(evaluation_path)
            found = {str(method.get("method")) for method in evaluation_data.get("methods", [])}
            required = {
                "hierarchical_gurobi_pipeline",
                "hierarchical_greedy_baseline",
                "hierarchical_bandit_gurobi_repair",
            }
            missing = required.difference(found)
            if missing:
                errors.append(f"Evaluation metrics missing methods {sorted(missing)}: {evaluation_path}")
            for key in [
                "comparison_score",
                "route_distance_km",
                "route_time_minutes",
                "nature_score",
                "weather_risk",
                "selected_nature_stops",
                "runtime_seconds",
            ]:
                if not any(field.get("key") == key for field in evaluation_data.get("chart_fields", [])):
                    errors.append(f"Evaluation metrics missing chart field {key}: {evaluation_path}")
        except Exception as exc:
            errors.append(f"Invalid evaluation metrics JSON: {evaluation_path}: {exc}")

    if not size_report.exists():
        errors.append(f"Missing artifact size report. Regenerate map artifacts to create: {size_report}")
    elif size_report.stat().st_size == 0:
        errors.append(f"Artifact size report is empty: {size_report}")

    return warnings, errors


def main() -> int:
    args = parse_args()
    warnings, errors = validate(Path(args.figure_dir), Path(args.output_dir))
    if warnings:
        print("Dashboard export warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Dashboard export validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Dashboard export validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
