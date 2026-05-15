"""Lightweight and modular map exports for production itinerary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TripConfig


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _route_name(row: pd.Series) -> str:
    return str(row.get("attraction_name", row.get("name", row.get("hotel_name", "Stop"))))


def _selected_rows(route_df: pd.DataFrame, max_routes: int = 1) -> pd.DataFrame:
    if route_df is None or route_df.empty:
        return pd.DataFrame(columns=["name", "latitude", "longitude"])
    frame = route_df.copy()
    if "route_key" in frame.columns and max_routes > 0:
        route_keys = list(frame["route_key"].dropna().astype(str).unique())[:max_routes]
        if route_keys:
            frame = frame[frame["route_key"].astype(str).isin(route_keys)].copy()
    sort_cols = [column for column in ["trip_days", "day", "stop_order"] if column in frame.columns]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame.reset_index(drop=True)


def _point_records(route_df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in route_df.iterrows():
        lat = _safe_float(row.get("latitude", row.get("hotel_latitude", 0.0)))
        lon = _safe_float(row.get("longitude", row.get("hotel_longitude", 0.0)))
        if not lat or not lon:
            continue
        records.append(
            {
                "name": _route_name(row),
                "city": str(row.get("city", row.get("overnight_city", ""))),
                "lat": lat,
                "lon": lon,
                "category": str(row.get("category", "")),
                "nature_region": str(row.get("nature_region", "")),
                "interest_adjusted_value": _safe_float(
                    row.get("interest_adjusted_value", row.get("final_poi_value", 0.0))
                ),
                "interest_fit": _safe_float(row.get("interest_fit", 0.0)),
                "interest_delta": _safe_float(row.get("interest_delta", 0.0)),
                "nature_score": _safe_float(row.get("nature_score", 0.0)),
                "scenic_score": _safe_float(row.get("scenic_score", 0.0)),
                "weather_sensitivity": _safe_float(row.get("weather_sensitivity", 0.0)),
                "seasonality_risk": _safe_float(row.get("seasonality_risk", 0.0)),
                "park_type": str(row.get("park_type", "")),
                "reason_selected": str(row.get("reason_selected", "")),
                "why_not_selected": str(row.get("why_not_selected", "")),
            }
        )
    return records


def _route_geojson(points: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for index, point in enumerate(points, start=1):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                "properties": {**point, "stop_order": index},
            }
        )
    if len(points) >= 2:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[point["lon"], point["lat"]] for point in points]},
                "properties": {"name": "Selected route", "role": "selected_route"},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _dashboard_metrics(points: list[dict[str, Any]], config: TripConfig) -> dict[str, Any]:
    values = [point["interest_adjusted_value"] for point in points]
    return {
        "trip_days": int(config.get("trip", "trip_days", 7)),
        "scenario": str(config.get("trip", "scenario", "california_coast")),
        "interest_profile": str(
            config.get("trip", "interest_profile", config.get("interest", "mode", "balanced_interest"))
        ),
        "selected_stop_count": len(points),
        "interest_adjusted_utility": round(float(sum(values)), 4),
        "default_route_label": "Selected route",
    }


def _json_payload(data: Any) -> str:
    return json.dumps(data, indent=2).replace("</", "<\\/")


def _write_text_asset(path: Path, text: str, written: list[Path]) -> None:
    path.write_text(text, encoding="utf-8")
    written.append(path)


def _write_json_asset(path: Path, data: Any, written: list[Path]) -> None:
    _write_text_asset(path, _json_payload(data), written)


def _global_assignment(global_name: str, data: Any) -> str:
    return f"window.{global_name} = {_json_payload(data)};\n"


def _global_map_assignment(global_name: str, key: str, data: Any) -> str:
    return (
        f"window.{global_name} = window.{global_name} || {{}};\n"
        f"window.{global_name}[{json.dumps(key)}] = {_json_payload(data)};\n"
    )


def _write_lightweight_map(path: Path, points: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    center = [points[0]["lat"], points[0]["lon"]] if points else [36.7783, -119.4179]
    payload = json.dumps({"points": points, "metrics": metrics, "center": center}).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lightweight Share Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Inter, system-ui, sans-serif; }}
    .share-panel {{ position: absolute; z-index: 900; top: 18px; left: 18px; width: 280px; background: #fff; border: 1px solid #dbe3ea; box-shadow: 0 12px 30px rgba(15,23,42,.18); border-radius: 8px; padding: 12px; }}
    .share-panel h1 {{ font-size: 16px; margin: 0 0 6px; }}
    .share-panel p {{ color: #475569; font-size: 12px; line-height: 1.4; margin: 4px 0; }}
    .share-warning {{ display: none; position: absolute; z-index: 1000; right: 18px; top: 18px; max-width: 360px; padding: 12px; border-radius: 8px; border: 1px solid #f59e0b; background: #fffbeb; color: #78350f; font-size: 13px; box-shadow: 0 12px 30px rgba(15,23,42,.16); }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="share-warning" class="share-warning"></div>
  <section class="share-panel">
    <h1>Nature-Aware Route</h1>
    <p><b>Scenario:</b> {metrics.get("scenario", "")}</p>
    <p><b>Interest:</b> {metrics.get("interest_profile", "")}</p>
    <p><b>Stops:</b> {metrics.get("selected_stop_count", 0)}</p>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script id="share-map-data" type="application/json">{payload}</script>
  <script>
    function showShareWarning(message) {{
      const warning = document.getElementById('share-warning');
      warning.textContent = message;
      warning.style.display = 'block';
      console.error('[share-map-error]', message);
    }}

    function initShareMap() {{
      const payload = JSON.parse(document.getElementById('share-map-data').textContent);
      if (!window.L) {{
        showShareWarning('Leaflet failed to load. Check your network connection or regenerate the map with local Leaflet assets.');
        return;
      }}
      if (!payload.points || payload.points.length === 0) {{
        showShareWarning('No selected route data found. Regenerate route assets first.');
      }}
      const map = L.map('map', {{ preferCanvas: true }}).setView(payload.center, 7);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
      const latLngs = [];
      payload.points.forEach((point, index) => {{
        const latLng = [point.lat, point.lon];
        latLngs.push(latLng);
        L.circleMarker(latLng, {{ radius: 7, color: '#166534', fillColor: '#2A9D8F', fillOpacity: 0.9, weight: 2 }})
          .bindPopup(`<b>${{index + 1}}. ${{point.name}}</b><br>${{point.city}}<br>${{point.nature_region || point.category}}`)
          .addTo(map);
      }});
      if (latLngs.length > 1) {{
        L.polyline(latLngs, {{ color: '#0f766e', weight: 4, opacity: 0.82 }}).addTo(map);
        map.fitBounds(latLngs, {{ padding: [40, 40] }});
      }}
    }}

    window.addEventListener('DOMContentLoaded', initShareMap);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_full_dashboard(
    root: Path, geojson: dict[str, Any], points: list[dict[str, Any]], metrics: dict[str, Any]
) -> list[Path]:
    assets = root / "assets"
    routes_dir = assets / "routes"
    pois_dir = assets / "pois"
    routes_dir.mkdir(parents=True, exist_ok=True)
    pois_dir.mkdir(parents=True, exist_ok=True)
    written = []

    route_id = "selected_route"
    poi_id = "selected_pois"
    route_json = "assets/routes/selected_route.geojson"
    route_js = "assets/routes/selected_route.js"
    poi_json = "assets/pois/selected_pois.json"
    poi_js = "assets/pois/selected_pois.js"

    style_path = assets / "style.css"
    _write_text_asset(
        style_path,
        """html,
body {
  height: 100%;
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#map {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100vh;
  z-index: 0;
  background: #e5edf2;
}

.dashboard-panel {
  position: absolute;
  z-index: 1000;
  top: 18px;
  left: 18px;
  width: 320px;
  background: #fff;
  border: 1px solid #dbe3ea;
  border-radius: 8px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, .18);
  padding: 12px;
}

.dashboard-panel h1 {
  font-size: 17px;
  margin: 0 0 8px;
}

.dashboard-panel button {
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  border-radius: 6px;
  padding: 6px 8px;
  cursor: pointer;
}

.dashboard-panel button:disabled {
  color: #94a3b8;
  cursor: default;
}

.metric {
  font-size: 12px;
  color: #475569;
  margin: 4px 0;
}

.status {
  min-height: 18px;
  margin-top: 8px;
  color: #0f766e;
  font-size: 12px;
}

.route-selector {
  margin-top: 8px;
  color: #334155;
  font-size: 12px;
}

.route-selector div {
  margin: 3px 0;
}

.diagnostic-panel {
  display: none;
  position: absolute;
  z-index: 1100;
  right: 18px;
  top: 18px;
  max-width: 420px;
  border-radius: 8px;
  border: 1px solid #f59e0b;
  background: #fffbeb;
  color: #78350f;
  box-shadow: 0 14px 32px rgba(15, 23, 42, .20);
  padding: 12px;
  font-size: 13px;
  line-height: 1.45;
}

.diagnostic-panel.error {
  border-color: #ef4444;
  background: #fef2f2;
  color: #7f1d1d;
}

.diagnostic-panel h2 {
  margin: 0 0 6px;
  font-size: 14px;
}

.diagnostic-panel code {
  word-break: break-all;
}
""",
        written,
    )

    route_path = routes_dir / "selected_route.geojson"
    _write_json_asset(route_path, geojson, written)

    route_js_path = routes_dir / "selected_route.js"
    _write_text_asset(route_js_path, _global_map_assignment("DASHBOARD_ROUTES", route_id, geojson), written)

    poi_path = pois_dir / "selected_pois.json"
    _write_json_asset(poi_path, points, written)

    poi_js_path = pois_dir / "selected_pois.js"
    _write_text_asset(poi_js_path, _global_map_assignment("DASHBOARD_POIS", poi_id, points), written)

    metrics_path = assets / "dashboard_metrics.json"
    _write_json_asset(metrics_path, metrics, written)

    metrics_js_path = assets / "dashboard_metrics.js"
    _write_text_asset(metrics_js_path, _global_assignment("DASHBOARD_METRICS", metrics), written)

    route_index = {
        "routes": [
            {
                "id": route_id,
                "label": "Selected Route",
                "default": True,
                "optional": False,
                "geojson": route_json,
                "geojson_js": route_js,
                "pois": poi_json,
                "pois_js": poi_js,
                "pois_id": poi_id,
            }
        ]
    }

    route_index_path = assets / "route_index.json"
    _write_json_asset(route_index_path, route_index, written)

    route_index_js_path = assets / "route_index.js"
    _write_text_asset(route_index_js_path, _global_assignment("DASHBOARD_ROUTE_INDEX", route_index), written)

    data_loader = assets / "data_loader.js"
    _write_text_asset(
        data_loader,
        """const DASHBOARD_HINT = 'If fallback assets are missing, run: python scripts/serve_dashboard.py';

function dashboardIsFileMode() {
  return window.location.protocol === 'file:';
}

function dashboardLogInit(message, extra) {
  console.log('[dashboard-init]', message, extra || '');
}

function dashboardLogLoader(message, extra) {
  console.log('[dashboard-loader]', message, extra || '');
}

function dashboardLogAssets(message, extra) {
  console.log('[dashboard-assets]', message, extra || '');
}

function dashboardLogError(message, extra) {
  console.error('[dashboard-error]', message, extra || '');
}

function showDashboardDiagnostic(level, checkName, assetUrl, detail) {
  const panel = document.getElementById('diagnostic-panel');
  if (!panel) return;
  panel.classList.toggle('error', level === 'error');
  panel.style.display = 'block';
  panel.innerHTML = `
    <h2>${level === 'error' ? 'Dashboard Error' : 'Dashboard Notice'}</h2>
    <div><b>Check:</b> ${checkName}</div>
    <div><b>Asset URL:</b> <code>${assetUrl || 'n/a'}</code></div>
    <div><b>Detail:</b> ${detail || 'No detail available.'}</div>
  `;
}

function loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[data-dashboard-src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.dataset.dashboardSrc = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

function readFallbackGlobal(fallback) {
  const value = window[fallback.globalName];
  if (fallback.key !== undefined && value) {
    return value[fallback.key];
  }
  return value;
}

async function loadFallbackAsset(kind, fallback) {
  if (!fallback || !fallback.script || !fallback.globalName) {
    throw new Error(`No JS fallback configured for ${kind}`);
  }
  dashboardLogLoader(`script fallback ${fallback.script}`);
  try {
    await loadScriptOnce(fallback.script);
  } catch (error) {
    const detail = `Both JSON fetch and JS fallback loading failed. Missing file: ${fallback.script}. Regenerate the dashboard export.`;
    showDashboardDiagnostic('error', `${kind} JS fallback loading`, fallback.script, detail);
    throw error;
  }
  const payload = readFallbackGlobal(fallback);
  if (payload === undefined || payload === null) {
    const detail = `JS fallback loaded but did not define ${fallback.globalName}. Regenerate the dashboard export.`;
    showDashboardDiagnostic('error', `${kind} JS fallback global`, fallback.script, detail);
    throw new Error(detail);
  }
  return payload;
}

async function fetchJsonAsset(kind, path) {
  dashboardLogAssets(`fetch ${path}`);
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadDashboardAsset(kind, path, fallbackGlobalName) {
  const fallback = typeof fallbackGlobalName === 'string'
    ? { script: path.replace(/\\.geojson$|\\.json$/, '.js'), globalName: fallbackGlobalName }
    : fallbackGlobalName;

  if (dashboardIsFileMode()) {
    showDashboardDiagnostic(
      'warning',
      `${kind} file mode`,
      path,
      'This dashboard was opened through file://. JSON fetch is blocked in this mode. Using JS fallback assets instead. If fallback assets are missing, run: python scripts/serve_dashboard.py'
    );
    return loadFallbackAsset(kind, fallback);
  }

  try {
    return await fetchJsonAsset(kind, path);
  } catch (error) {
    dashboardLogError(`${kind} JSON fetch failed`, error);
    return loadFallbackAsset(kind, fallback);
  }
}

async function loadRouteIndex() {
  return loadDashboardAsset('route index', 'assets/route_index.json', {
    script: 'assets/route_index.js',
    globalName: 'DASHBOARD_ROUTE_INDEX'
  });
}

async function loadDashboardMetrics() {
  return loadDashboardAsset('dashboard metrics', 'assets/dashboard_metrics.json', {
    script: 'assets/dashboard_metrics.js',
    globalName: 'DASHBOARD_METRICS'
  });
}

async function loadRouteGeoJson(routeRecord) {
  return loadDashboardAsset(`route ${routeRecord.id}`, routeRecord.geojson, {
    script: routeRecord.geojson_js,
    globalName: 'DASHBOARD_ROUTES',
    key: routeRecord.id
  });
}

async function loadPoiJson(routeRecord) {
  if (!routeRecord.pois && !routeRecord.pois_js) return [];
  const key = routeRecord.pois_id || routeRecord.id || 'selected_pois';
  return loadDashboardAsset(`POIs ${key}`, routeRecord.pois, {
    script: routeRecord.pois_js,
    globalName: 'DASHBOARD_POIS',
    key
  });
}

window.dashboardIsFileMode = dashboardIsFileMode;
window.loadDashboardAsset = loadDashboardAsset;
window.loadRouteIndex = loadRouteIndex;
window.loadDashboardMetrics = loadDashboardMetrics;
window.loadRouteGeoJson = loadRouteGeoJson;
window.loadPoiJson = loadPoiJson;
window.showDashboardDiagnostic = showDashboardDiagnostic;
window.dashboardLogInit = dashboardLogInit;
window.dashboardLogLoader = dashboardLogLoader;
window.dashboardLogAssets = dashboardLogAssets;
window.dashboardLogError = dashboardLogError;
""",
        written,
    )

    map_controls = assets / "map_controls.js"
    _write_text_asset(
        map_controls,
        """let dashboardMap = null;
let selectedRouteIndex = null;
let selectedLayerGroup = null;
let optionalLayerGroup = null;

function setDashboardStatus(message, isError = false) {
  const target = document.getElementById('dashboard-status');
  if (!target) return;
  target.textContent = message;
  target.style.color = isError ? '#b91c1c' : '#0f766e';
}

function assertMapContainerReady() {
  if (!window.L) {
    showDashboardDiagnostic('error', 'Leaflet loaded', 'https://unpkg.com/leaflet/dist/leaflet.js', 'window.L is missing.');
    throw new Error('Leaflet is not loaded');
  }
  const container = document.getElementById('map');
  if (!container) {
    showDashboardDiagnostic('error', 'map container exists', '#map', 'Missing <div id="map">.');
    throw new Error('Map container missing');
  }
  const rect = container.getBoundingClientRect();
  if (rect.height <= 0 || rect.width <= 0) {
    showDashboardDiagnostic('error', 'map container has nonzero height', '#map', `Measured ${rect.width}x${rect.height}.`);
    throw new Error('Map container has zero size');
  }
  return container;
}

function routeLineStyle(isOptional) {
  return {
    color: isOptional ? '#64748b' : '#0f766e',
    weight: isOptional ? 3 : 4,
    opacity: isOptional ? 0.55 : 0.86,
    dashArray: isOptional ? '6 6' : null
  };
}

function pointToMarker(point, index, isOptional) {
  const latLng = [Number(point.lat), Number(point.lon)];
  const marker = L.circleMarker(latLng, {
    radius: isOptional ? 5 : 7,
    color: isOptional ? '#475569' : '#166534',
    fillColor: isOptional ? '#94a3b8' : '#2A9D8F',
    fillOpacity: 0.9,
    weight: 2
  });
  marker.bindPopup(`<b>${index + 1}. ${point.name || 'Stop'}</b><br>${point.city || ''}<br>${point.nature_region || point.category || ''}`);
  return marker;
}

function geoJsonPointToMarker(feature, latlng) {
  return L.circleMarker(latlng, {
    radius: 7,
    color: '#166534',
    fillColor: '#2A9D8F',
    fillOpacity: 0.9,
    weight: 2
  });
}

function bindPopup(feature, layer) {
  const p = feature.properties || {};
  layer.bindPopup(`<b>${p.stop_order || ''} ${p.name || 'Route'}</b><br>${p.city || ''}<br>${p.nature_region || p.category || ''}`);
}

function drawRouteGeometry(map, route, group, isOptional) {
  if (!route || !Array.isArray(route.features) || route.features.length === 0) return null;
  const layer = L.geoJSON(route, {
    filter: feature => feature.geometry && feature.geometry.type !== 'Point',
    pointToLayer: geoJsonPointToMarker,
    onEachFeature: bindPopup,
    style: () => routeLineStyle(isOptional)
  });
  layer.addTo(group || map);
  return layer;
}

function drawPoiMarkers(group, pois, isOptional) {
  const latLngs = [];
  (pois || []).forEach((point, index) => {
    if (!Number.isFinite(Number(point.lat)) || !Number.isFinite(Number(point.lon))) return;
    const latLng = [Number(point.lat), Number(point.lon)];
    latLngs.push(latLng);
    pointToMarker(point, index, isOptional).addTo(group);
  });
  return latLngs;
}

function fitLayerGroup(map, group, fallbackLatLngs) {
  try {
    const bounds = group.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40] });
      return;
    }
  } catch (error) {
    dashboardLogError('fitBounds failed', error);
  }
  if (fallbackLatLngs.length > 1) {
    map.fitBounds(fallbackLatLngs, { padding: [40, 40] });
  }
}

function renderRouteSelector(routes) {
  const target = document.getElementById('route-selector');
  if (!target) return;
  target.innerHTML = '';
  routes.forEach(route => {
    const row = document.createElement('div');
    row.textContent = `${route.default ? 'Default' : route.optional ? 'Optional' : 'Route'}: ${route.label || route.id}`;
    target.appendChild(row);
  });
}

function defaultRoute(index) {
  const routes = Array.isArray(index.routes) ? index.routes : [];
  return routes.find(route => route.default) || routes[0];
}

async function drawRouteRecord(routeRecord, isOptional, targetGroup) {
  const group = targetGroup || L.layerGroup().addTo(dashboardMap);
  const route = await loadRouteGeoJson(routeRecord);
  let pois = [];
  try {
    pois = await loadPoiJson(routeRecord);
  } catch (error) {
    dashboardLogError(`POI load failed for ${routeRecord.id}`, error);
  }
  drawRouteGeometry(dashboardMap, route, group, isOptional);
  const markerLatLngs = drawPoiMarkers(group, pois, isOptional);
  return { group, markerLatLngs };
}

async function initMap() {
  dashboardLogInit('start');
  try {
    assertMapContainerReady();
    const map = L.map("map", { preferCanvas: true }).setView([36.5, -119.5], 6);
    dashboardMap = map;
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    const index = await loadRouteIndex();
    const metrics = await loadDashboardMetrics();
    window.dashboardMetrics = metrics;
    window.dashboardRouteIndex = index;
    selectedRouteIndex = index;

    if (window.renderDashboardMetrics) {
      window.renderDashboardMetrics(metrics);
    }
    renderRouteSelector(index.routes || []);

    const selected = defaultRoute(index);
    if (!selected) {
      showDashboardDiagnostic('error', 'default route exists', 'assets/route_index.json', 'No default route assets found. Regenerate dashboard assets.');
      return;
    }

    const drawn = await drawRouteRecord(selected, false);
    selectedLayerGroup = drawn.group;
    fitLayerGroup(map, selectedLayerGroup, drawn.markerLatLngs);
    setDashboardStatus(`Loaded default route: ${selected.label || selected.id}`);
    dashboardLogInit('complete');
  } catch (error) {
    dashboardLogError('initialization failed', error);
    if (!document.getElementById('diagnostic-panel')?.textContent) {
      showDashboardDiagnostic('error', 'dashboard initialization', window.location.href, error.message);
    }
  }
}

async function loadOptionalLayers() {
  if (!dashboardMap) {
    setDashboardStatus('Map is not initialized yet.', true);
    return;
  }
  if (!selectedRouteIndex) {
    selectedRouteIndex = await loadRouteIndex();
  }
  const optionalRoutes = (selectedRouteIndex.routes || []).filter(route => route.optional);
  if (!optionalRoutes.length) {
    setDashboardStatus('No optional layer assets found. Regenerate dashboard assets.', true);
    return;
  }
  if (optionalLayerGroup) {
    optionalLayerGroup.clearLayers();
  } else {
    optionalLayerGroup = L.layerGroup().addTo(dashboardMap);
  }
  let loaded = 0;
  for (const route of optionalRoutes) {
    try {
      await drawRouteRecord(route, true, optionalLayerGroup);
      loaded += 1;
    } catch (error) {
      dashboardLogError(`optional layer failed for ${route.id}`, error);
    }
  }
  setDashboardStatus(loaded ? `Loaded ${loaded} optional layers` : 'No optional layer assets found. Regenerate dashboard assets.', loaded === 0);
  renderRouteSelector(selectedRouteIndex.routes || []);
}

window.loadOptionalLayers = loadOptionalLayers;
window.addEventListener('DOMContentLoaded', initMap);
""",
        written,
    )

    dashboard_js = assets / "dashboard.js"
    _write_text_asset(
        dashboard_js,
        """function renderDashboardMetrics(metrics) {
  const target = document.getElementById('metrics');
  if (!target) return;
  target.innerHTML = `
    <div class="metric"><b>Scenario:</b> ${metrics.scenario}</div>
    <div class="metric"><b>Interest:</b> ${metrics.interest_profile}</div>
    <div class="metric"><b>Stops:</b> ${metrics.selected_stop_count}</div>
    <div class="metric"><b>Interest utility:</b> ${metrics.interest_adjusted_utility}</div>
  `;
}

window.renderDashboardMetrics = renderDashboardMetrics;
""",
        written,
    )

    index_path = root / "index.html"
    _write_text_asset(
        index_path,
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Full Interactive Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
  <link rel="stylesheet" href="assets/style.css" />
</head>
<body>
  <div id="map"></div>
  <div id="diagnostic-panel" class="diagnostic-panel" role="status" aria-live="polite"></div>
  <section class="dashboard-panel">
    <h1>Full Interactive Dashboard</h1>
    <div id="metrics"></div>
    <button id="optional-layer-button" type="button">Load optional layers</button>
    <div id="dashboard-status" class="status"></div>
    <div id="route-selector" class="route-selector"></div>
  </section>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <script src="assets/data_loader.js"></script>
  <script src="assets/dashboard.js"></script>
  <script src="assets/map_controls.js"></script>
  <script>
    document.getElementById('optional-layer-button').addEventListener('click', () => window.loadOptionalLayers());
  </script>
</body>
</html>
""",
        written,
    )
    return written


def _artifact_row(
    path: Path, artifact_type: str, layer_count: int, route_count: int, marker_count: int, notes: str
) -> dict[str, Any]:
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return {
        "artifact_path": str(path),
        "artifact_type": artifact_type,
        "file_size_mb": round(size_mb, 4),
        "layer_count": int(layer_count),
        "route_count": int(route_count),
        "marker_count": int(marker_count),
        "notes": notes,
    }


def _feature_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(data, dict) and isinstance(data.get("features"), list):
        return len(data["features"])
    if isinstance(data, list):
        return len(data)
    return 0


def export_map_artifacts(
    route_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    figure_dir: str | Path,
    config: TripConfig,
) -> dict[str, Path]:
    """Export small share and modular dashboard map artifacts."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    max_routes = int(config.get("map_export", "lightweight_max_routes", 1))
    selected = _selected_rows(route_df, max_routes=max_routes)
    points = _point_records(selected)
    geojson = _route_geojson(points)
    metrics = _dashboard_metrics(points, config)
    mode = str(config.get("map_export", "mode", "both")).lower()

    artifacts: dict[str, Path] = {}
    report_rows = []
    if mode in {"both", "lightweight", "share"}:
        share_path = figure_dir / "lightweight_share_map.html"
        _write_lightweight_map(share_path, points, metrics)
        artifacts["lightweight_share_map"] = share_path
        report_rows.append(
            _artifact_row(
                share_path,
                "lightweight_share_map",
                1,
                1 if len(points) > 1 else 0,
                len(points),
                "selected route only; no comparison or debug layers",
            )
        )

    if mode in {"both", "full", "dashboard"}:
        dashboard_root = figure_dir / "full_interactive_dashboard"
        written = _write_full_dashboard(dashboard_root, geojson, points, metrics)
        artifacts["full_interactive_dashboard"] = dashboard_root / "index.html"
        for path in written:
            if path.name == "index.html":
                artifact_type = "full_dashboard_index"
                route_count = 1 if len(points) > 1 else 0
                marker_count = len(points)
                notes = "modular_dashboard_entrypoint"
            elif path.suffix == ".geojson":
                artifact_type = "route_geojson"
                route_count = 1
                marker_count = max(0, _feature_count(path) - 1)
                notes = "json_fetch_asset;selected_default_route"
            elif "routes" in path.parts and path.suffix == ".js":
                artifact_type = "route_js_fallback"
                route_count = 1
                marker_count = max(0, _feature_count(path.with_suffix(".geojson")) - 1)
                notes = "file_mode_js_fallback;selected_default_route"
            elif "pois" in path.parts and path.suffix == ".json":
                artifact_type = "poi_json"
                route_count = 0
                marker_count = _feature_count(path)
                notes = "json_fetch_asset;selected_default_route"
            elif "pois" in path.parts and path.suffix == ".js":
                artifact_type = "poi_js_fallback"
                route_count = 0
                marker_count = _feature_count(path.with_suffix(".json"))
                notes = "file_mode_js_fallback;selected_default_route"
            elif path.suffix == ".json":
                artifact_type = "dashboard_json"
                route_count = 0
                marker_count = 0
                notes = "json_fetch_asset"
            elif path.suffix == ".css":
                artifact_type = "dashboard_css"
                route_count = 0
                marker_count = 0
                notes = "dashboard_style"
            elif path.suffix == ".js":
                artifact_type = "dashboard_js"
                route_count = 0
                marker_count = 0
                notes = (
                    "file_mode_js_fallback"
                    if path.name in {"route_index.js", "dashboard_metrics.js"}
                    else "dashboard_behavior"
                )
            else:
                artifact_type = "full_interactive_dashboard_asset"
                route_count = 0
                marker_count = 0
                notes = "externalized modular dashboard asset"
            report_rows.append(_artifact_row(path, artifact_type, 1, route_count, marker_count, notes))

    monolith_path = figure_dir / "production_hierarchical_trip_map.html"
    if monolith_path.exists():
        report_rows.append(
            _artifact_row(
                monolith_path,
                "legacy_monolithic_folium_html",
                0,
                0,
                0,
                "legacy compatibility artifact; not the preferred share export",
            )
        )

    pd.DataFrame(report_rows).to_csv(output_dir / "production_map_artifact_size_report.csv", index=False)
    return artifacts
