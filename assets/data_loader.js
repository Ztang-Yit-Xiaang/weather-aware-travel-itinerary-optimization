const DASHBOARD_HINT = 'If fallback assets are missing, run: python scripts/serve_dashboard.py';
let dashboardFileModeNoticeShown = false;

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
  if (level !== 'error') {
    dashboardLogLoader(`${checkName}: ${detail || ''}`, assetUrl || '');
    return;
  }
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
    ? { script: path.replace(/\.geojson$|\.json$/, '.js'), globalName: fallbackGlobalName }
    : fallbackGlobalName;

  if (dashboardIsFileMode()) {
    if (!dashboardFileModeNoticeShown) {
      dashboardFileModeNoticeShown = true;
      showDashboardDiagnostic(
        'warning',
        `${kind} file mode`,
        path,
        'This dashboard was opened through file://. JSON fetch is blocked in this mode. Using JS fallback assets instead. If fallback assets are missing, run: python scripts/serve_dashboard.py'
      );
    }
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

async function loadDebugSummary() {
  return loadDashboardAsset('debug summary', 'assets/debug_summary.json', {
    script: 'assets/debug_summary.js',
    globalName: 'DASHBOARD_DEBUG_SUMMARY'
  });
}

async function loadInterestPreview() {
  return loadDashboardAsset('interest preview', 'assets/interest_preview.json', {
    script: 'assets/interest_preview.js',
    globalName: 'DASHBOARD_INTEREST_PREVIEW'
  });
}

async function loadPlaybackData() {
  return loadDashboardAsset('playback data', 'assets/playback_data.json', {
    script: 'assets/playback_data.js',
    globalName: 'DASHBOARD_PLAYBACK_DATA'
  });
}

async function loadCityDetails() {
  return loadDashboardAsset('city details', 'assets/city_details.json', {
    script: 'assets/city_details.js',
    globalName: 'DASHBOARD_CITY_DETAILS'
  });
}

async function loadSelectedHotels() {
  return loadDashboardAsset('selected hotels', 'assets/selected_hotels.json', {
    script: 'assets/selected_hotels.js',
    globalName: 'DASHBOARD_SELECTED_HOTELS'
  });
}

async function loadHotelChoices() {
  return loadDashboardAsset('hotel choices', 'assets/hotel_choices.json', {
    script: 'assets/hotel_choices.js',
    globalName: 'DASHBOARD_HOTEL_CHOICES'
  });
}

async function loadNatureExplore() {
  return loadDashboardAsset('nature explore', 'assets/nature_explore.json', {
    script: 'assets/nature_explore.js',
    globalName: 'DASHBOARD_NATURE_EXPLORE'
  });
}

async function loadNatureSiteRoutes() {
  return loadDashboardAsset('nature site routes', 'assets/nature_site_routes.json', {
    script: 'assets/nature_site_routes.js',
    globalName: 'DASHBOARD_NATURE_SITE_ROUTES'
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
  const key = routeRecord.id;
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
window.loadDebugSummary = loadDebugSummary;
window.loadInterestPreview = loadInterestPreview;
window.loadPlaybackData = loadPlaybackData;
window.loadCityDetails = loadCityDetails;
window.loadSelectedHotels = loadSelectedHotels;
window.loadHotelChoices = loadHotelChoices;
window.loadNatureExplore = loadNatureExplore;
window.loadNatureSiteRoutes = loadNatureSiteRoutes;
window.loadRouteGeoJson = loadRouteGeoJson;
window.loadPoiJson = loadPoiJson;
window.showDashboardDiagnostic = showDashboardDiagnostic;
window.dashboardLogInit = dashboardLogInit;
window.dashboardLogLoader = dashboardLogLoader;
window.dashboardLogAssets = dashboardLogAssets;
window.dashboardLogError = dashboardLogError;
