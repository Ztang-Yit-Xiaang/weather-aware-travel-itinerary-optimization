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


def validate(figure_dir: Path, output_dir: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    dashboard_root = figure_dir / "full_interactive_dashboard"
    assets = dashboard_root / "assets"

    lightweight = figure_dir / "lightweight_share_map.html"
    index_html = dashboard_root / "index.html"
    style_css = assets / "style.css"
    map_js = assets / "map_controls.js"
    dashboard_js = assets / "dashboard.js"
    data_loader_js = assets / "data_loader.js"
    route_index_path = assets / "route_index.json"
    route_index_js_path = assets / "route_index.js"
    metrics_path = assets / "dashboard_metrics.json"
    metrics_js_path = assets / "dashboard_metrics.js"
    size_report = output_dir / "production_map_artifact_size_report.csv"

    for path, label in [
        (lightweight, "lightweight share map"),
        (index_html, "full dashboard index"),
        (style_css, "dashboard CSS"),
        (map_js, "map controls JS"),
        (dashboard_js, "dashboard JS"),
        (data_loader_js, "dashboard data loader JS"),
        (route_index_path, "route index JSON"),
        (route_index_js_path, "route index JS fallback"),
        (metrics_path, "dashboard metrics JSON"),
        (metrics_js_path, "dashboard metrics JS fallback"),
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

    if style_css.exists():
        css = style_css.read_text(encoding="utf-8", errors="replace")
        if "#map" not in css or "height: 100vh" not in css:
            errors.append(f"Dashboard CSS does not give #map a full viewport height: {style_css}")

    if map_js.exists():
        js = map_js.read_text(encoding="utf-8", errors="replace")
        for token in ['L.map("map"', "loadOptionalLayers", "dashboardLogError"]:
            if token not in js:
                errors.append(f"Dashboard JS missing required diagnostic/init token {token!r}: {map_js}")
        if "alert(" in js:
            errors.append(f"Dashboard JS should not use alert() for normal optional-layer status: {map_js}")

    if data_loader_js.exists():
        js = data_loader_js.read_text(encoding="utf-8", errors="replace")
        for token in [
            "dashboardIsFileMode",
            "loadScriptOnce",
            "loadDashboardAsset",
            "loadRouteIndex",
            "loadDashboardMetrics",
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
                for required_key in ["id", "label", "default", "optional", "geojson", "geojson_js", "pois", "pois_js"]:
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
                if poi_path.exists():
                    try:
                        poi_data = read_json(poi_path)
                        if not isinstance(poi_data, list):
                            errors.append(f"POI JSON should be a list: {poi_path}")
                        elif not poi_data:
                            warnings.append(f"POI JSON is empty: {poi_path}")
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
        except Exception as exc:
            errors.append(f"Invalid route_index.js fallback: {route_index_js_path}: {exc}")

    if metrics_path.exists():
        try:
            read_json(metrics_path)
        except Exception as exc:
            errors.append(f"Invalid dashboard_metrics.json: {metrics_path}: {exc}")
    if metrics_js_path.exists():
        try:
            read_js_global(metrics_js_path, "DASHBOARD_METRICS")
        except Exception as exc:
            errors.append(f"Invalid dashboard_metrics.js fallback: {metrics_js_path}: {exc}")

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
