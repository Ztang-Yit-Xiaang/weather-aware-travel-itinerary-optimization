let dashboardMap = null;
let dashboardRouteIndex = null;
let dashboardPlaybackData = null;
let activePlaybackRouteId = null;
let playbackMarker = null;
let playbackTimer = null;
let playbackAnimationFrame = null;
let playbackStopIndex = 0;
let playbackRunning = false;
const loadedRouteLayers = new Map();
const activeRouteIds = new Set();
let selectedHotelLayer = null;
let selectedHotelsVisible = true;
let hotelCandidatesVisible = false;
let livePreviewLayer = null;
let livePreviewPlaybackStops = [];
let livePreviewRouteRecord = null;
let terrainBaseLayer = null;
let roadsOverlayLayer = null;
let labelOverlayLayer = null;
let weatherRiskLayer = null;

const FAMILY_STYLES = {
  selected: { color: '#0a8f83', fill: '#18b8a5', weight: 6, dashArray: null },
  route_matrix: { color: '#2f6fdd', fill: '#7eb6ff', weight: 3, dashArray: '7 7' },
  trip_length: { color: '#7c4dd8', fill: '#bba7ff', weight: 3, dashArray: '4 7' },
  method: { color: '#f9735b', fill: '#ffba8e', weight: 3, dashArray: '2 7' },
  interest_profile: { color: '#0f766e', fill: '#5eead4', weight: 4, dashArray: '10 8' },
  context: { color: '#66788a', fill: '#aab8c4', weight: 2, dashArray: '9 8' },
  hotel: { color: '#6d4fd7', fill: '#d8cffd', weight: 1.5, dashArray: null },
  must_go: { color: '#d92652', fill: '#ff8aa4', weight: 1.5, dashArray: null },
  nature: { color: '#247a32', fill: '#6ac659', weight: 1.5, dashArray: null },
  live_preview: { color: '#f9735b', fill: '#ffb199', weight: 5, dashArray: '3 12' }
};

function setDashboardStatus(message, isError = false) {
  const target = document.getElementById('dashboard-status');
  if (!target) return;
  target.textContent = message;
  target.style.color = isError ? '#b91c1c' : '#0a6b53';
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
}

function defaultRoute(index) {
  const routes = Array.isArray(index?.routes) ? index.routes : [];
  return routes.find(route => route.default) || routes[0] || null;
}

function isPlayableRoute(routeRecord) {
  if (!routeRecord) return false;
  if (routeRecord.id === 'live_preview') return livePreviewPlaybackStops.length > 1;
  if (routeRecord.playable === false || routeRecord.marker_only === true) return false;
  return ['selected', 'route_matrix', 'trip_length', 'method', 'interest_profile'].includes(routeRecord.family);
}

function firstPlayableRouteId() {
  const routes = Array.isArray(dashboardRouteIndex?.routes) ? dashboardRouteIndex.routes : [];
  const current = routeById(activePlaybackRouteId);
  if (isPlayableRoute(current) && activeRouteIds.has(activePlaybackRouteId)) return activePlaybackRouteId;
  if (activePlaybackRouteId === 'live_preview' && livePreviewPlaybackStops.length > 1) return 'live_preview';
  const visible = routes.find(route => activeRouteIds.has(route.id) && isPlayableRoute(route));
  const fallback = visible || routes.find(route => route.default && isPlayableRoute(route)) || routes.find(isPlayableRoute);
  return fallback?.id || null;
}

function playbackStops(routeId = activePlaybackRouteId) {
  if (routeId === 'live_preview') return livePreviewPlaybackStops;
  const record = routeById(routeId);
  if (!isPlayableRoute(record)) return [];
  const fromAsset = dashboardPlaybackData?.routes?.[routeId]?.stops;
  if (Array.isArray(fromAsset) && fromAsset.length) return fromAsset;
  const loaded = loadedRouteLayers.get(routeId);
  return loaded?.pois || [];
}

function playbackSpeedMs() {
  const select = document.getElementById('playback-speed');
  return Number(select?.value || 900);
}

function updatePlaybackProgress() {
  const stops = playbackStops();
  const progress = document.getElementById('playback-progress-bar');
  const label = document.getElementById('playback-current-stop');
  const pct = stops.length > 1 ? (playbackStopIndex / (stops.length - 1)) * 100 : 0;
  if (progress) progress.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  if (label) {
    const stop = stops[playbackStopIndex];
    const ordinal = stop ? routePointOrdinal(stop, playbackStopIndex) : '';
    label.textContent = stop ? `${ordinal}/${stops.length} · ${stop.name || 'Stop'} · ${stop.city || ''}` : 'No playable route loaded.';
  }
}

function ensurePlaybackMarker(stop) {
  if (!stop || !Number.isFinite(Number(stop.lat)) || !Number.isFinite(Number(stop.lon))) return null;
  const latLng = [Number(stop.lat), Number(stop.lon)];
  if (!playbackMarker) {
    playbackMarker = L.marker(latLng, {
      icon: L.divIcon({
        className: 'playback-marker-icon',
        html: '<span class="playback-pulse" aria-hidden="true"></span>',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      }),
      zIndexOffset: 900
    }).addTo(dashboardMap);
  } else {
    playbackMarker.setLatLng(latLng);
  }
  playbackMarker.bindPopup(`<b>${stop.name || 'Stop'}</b><br>${stop.city || ''}`);
  return playbackMarker;
}

function setPlaybackStop(index, pan = false) {
  const stops = playbackStops();
  if (!stops.length) return;
  playbackStopIndex = Math.max(0, Math.min(stops.length - 1, index));
  const stop = stops[playbackStopIndex];
  ensurePlaybackMarker(stop);
  updatePlaybackProgress();
  if (window.renderActiveStopDetail) {
    window.renderActiveStopDetail(stop, routeById(activePlaybackRouteId), {
      index: playbackStopIndex,
      total: stops.length
    });
  }
  const follow = document.getElementById('playback-follow');
  if (pan && follow?.checked && stop?.lat && stop?.lon) {
    dashboardMap.panTo([Number(stop.lat), Number(stop.lon)]);
  }
}

function interpolateStop(fromStop, toStop, ratio) {
  return {
    lat: Number(fromStop.lat) + (Number(toStop.lat) - Number(fromStop.lat)) * ratio,
    lon: Number(fromStop.lon) + (Number(toStop.lon) - Number(fromStop.lon)) * ratio,
    name: toStop.name,
    city: toStop.city
  };
}

function animateToStop(nextIndex) {
  const stops = playbackStops();
  if (!playbackRunning || nextIndex >= stops.length) {
    playbackRunning = false;
    return;
  }
  const fromStop = stops[playbackStopIndex] || stops[0];
  const toStop = stops[nextIndex];
  const start = performance.now();
  const duration = playbackSpeedMs();
  function frame(now) {
    if (!playbackRunning) return;
    const ratio = Math.min(1, (now - start) / duration);
    ensurePlaybackMarker(interpolateStop(fromStop, toStop, ratio));
    if (ratio < 1) {
      playbackAnimationFrame = requestAnimationFrame(frame);
      return;
    }
    setPlaybackStop(nextIndex, true);
    playbackTimer = window.setTimeout(() => animateToStop(nextIndex + 1), Math.max(120, duration * 0.22));
  }
  playbackAnimationFrame = requestAnimationFrame(frame);
}

function playRouteAnimation() {
  const stops = playbackStops();
  if (stops.length < 2) {
    setDashboardStatus('Playback needs at least two route stops.', true);
    return;
  }
  if (playbackStopIndex >= stops.length - 1) playbackStopIndex = 0;
  playbackRunning = true;
  setPlaybackStop(playbackStopIndex, true);
  animateToStop(playbackStopIndex + 1);
  setDashboardStatus('Route playback running.');
}

function pauseRouteAnimation() {
  playbackRunning = false;
  if (playbackTimer) window.clearTimeout(playbackTimer);
  if (playbackAnimationFrame) window.cancelAnimationFrame(playbackAnimationFrame);
  setDashboardStatus('Route playback paused.');
}

function restartRouteAnimation() {
  pauseRouteAnimation();
  playbackStopIndex = 0;
  setPlaybackStop(0, true);
}

function stepPlayback(delta) {
  pauseRouteAnimation();
  setPlaybackStop(playbackStopIndex + delta, true);
}

function setActivePlaybackRoute(routeId, reset = true) {
  activePlaybackRouteId = routeId;
  if (reset) playbackStopIndex = 0;
  const summary = document.getElementById('playback-route-label');
  const record = routeById(routeId);
  if (summary) summary.textContent = record ? record.label : routeId;
  setPlaybackStop(playbackStopIndex, false);
}

function bindPlaybackControls() {
  document.getElementById('playback-play')?.addEventListener('click', playRouteAnimation);
  document.getElementById('playback-pause')?.addEventListener('click', pauseRouteAnimation);
  document.getElementById('playback-restart')?.addEventListener('click', restartRouteAnimation);
  document.getElementById('playback-prev')?.addEventListener('click', () => stepPlayback(-1));
  document.getElementById('playback-next')?.addEventListener('click', () => stepPlayback(1));
}

function routeLineStyle(routeRecord) {
  const style = FAMILY_STYLES[routeRecord?.family] || FAMILY_STYLES.selected;
  return {
    color: style.color,
    weight: style.weight,
    opacity: routeRecord?.default ? 0.88 : 0.62,
    dashArray: style.dashArray,
    lineCap: 'round',
    lineJoin: 'round',
    className: routeRecord?.default ? 'route-line route-line-default' : 'route-line'
  };
}

function markerPopup(point, index, routeRecord) {
  const candidateFamilies = ['hotel', 'nature', 'must_go', 'context'];
  const isEndpoint = Boolean(point?.is_route_endpoint);
  const isHotelNode = Boolean(point?.is_hotel_node || point?.node_kind === 'hotel');
  const selectionState = isEndpoint
    ? 'Airport endpoint - not counted as a selected POI'
    : (isHotelNode
      ? 'Selected overnight/base hotel - not counted as a selected POI'
    : (routeRecord?.family === 'live_preview'
      ? 'Preview only - not written to optimizer artifacts'
      : (candidateFamilies.includes(routeRecord?.family) ? 'Candidate/context marker - not a selected route stop' : 'Saved optimized route stop')));
  const label = routeRecord?.family === 'hotel' || routeRecord?.family === 'nature'
    ? (point.name || point.hotel_name || routeRecord.label)
    : (isEndpoint ? `${point.airport_code || 'Airport'} · ${point.name || 'Airport'}`
      : (isHotelNode ? `H · ${point.name || point.hotel_name || 'Hotel'}`
        : `${routePointOrdinal(point, index)}. ${point.name || 'Stop'}`));
  const day = point.day ? `Day ${point.day}<br>` : '';
  const displayUtility = Number(point.display_utility ?? point.interest_adjusted_value ?? point.final_poi_value);
  const finalValue = Number(point.final_poi_value);
  const interestValue = Number(point.interest_adjusted_value);
  const utility = Number.isFinite(displayUtility) ? `<br>Optimizer value: ${displayUtility.toFixed(2)}` : '';
  const finalLine = Number.isFinite(finalValue) && finalValue > 0 ? `<br>Final POI value: ${finalValue.toFixed(2)}` : '';
  const interestLine = Number.isFinite(interestValue) && interestValue > 0 && Math.abs(interestValue - finalValue) > 0.0001
    ? `<br>Interest-adjusted value: ${interestValue.toFixed(2)}`
    : '';
  const hotel = point.hotel_name ? `<br>Hotel: ${point.hotel_name}` : '';
  return `<b>${label}</b><br>${selectionState}<br>${day}${point.city || ''}<br>${point.nature_region || point.category || ''}${hotel}${utility}${finalLine}${interestLine}`;
}

function pointToMarker(point, index, routeRecord) {
  const style = FAMILY_STYLES[routeRecord?.family] || FAMILY_STYLES.selected;
  const candidateMarker = ['hotel', 'must_go', 'nature', 'context'].includes(routeRecord?.family);
  if (point?.is_route_endpoint) {
    const marker = L.marker([Number(point.lat), Number(point.lon)], {
      icon: L.divIcon({
        className: 'airport-route-endpoint',
        html: `<span>${point.airport_code || 'AP'}</span>`,
        iconSize: null,
        iconAnchor: [19, 12],
        popupAnchor: [0, -14]
      }),
      zIndexOffset: routeRecord?.default ? 720 : 430
    });
    marker.bindPopup(markerPopup(point, index, routeRecord));
    marker.on('click', () => {
      if (isPlayableRoute(routeRecord)) {
        activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
        playbackStopIndex = index;
        setPlaybackStop(index, false);
      }
    });
    return marker;
  }
  if (!candidateMarker) {
    const marker = L.marker([Number(point.lat), Number(point.lon)], {
      icon: L.divIcon({
        className: 'numbered-route-stop',
        html: `<span>${routePointOrdinal(point, index)}</span>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
        popupAnchor: [0, -16]
      }),
      zIndexOffset: routeRecord?.default ? 760 : 460
    });
    marker.bindPopup(markerPopup(point, index, routeRecord));
    marker.on('click', () => {
      if (isPlayableRoute(routeRecord)) {
        activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
        playbackStopIndex = index;
        setPlaybackStop(index, false);
      }
    });
    return marker;
  }
  const marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
    radius: candidateMarker ? 4.5 : (routeRecord?.optional ? 5 : 8),
    color: style.color,
    fillColor: style.fill,
    fillOpacity: candidateMarker ? 0.5 : (routeRecord?.optional ? 0.72 : 0.9),
    weight: candidateMarker ? 1.5 : (routeRecord?.optional ? 2 : 3),
    className: routeRecord?.family === 'nature' ? 'nature-marker' : 'route-marker'
  });
  marker.bindPopup(routeRecord?.family === 'hotel' ? hotelMarkerPopup(point, Boolean(point.selected)) : markerPopup(point, index, routeRecord));
  marker.on('mouseover', () => marker.setStyle({ radius: candidateMarker ? 6 : (routeRecord?.optional ? 7 : 10), fillOpacity: candidateMarker ? 0.76 : 1 }));
  marker.on('mouseout', () => marker.setStyle({ radius: candidateMarker ? 4.5 : (routeRecord?.optional ? 5 : 8), fillOpacity: candidateMarker ? 0.5 : (routeRecord?.optional ? 0.72 : 0.9) }));
  marker.on('click', () => {
    if (isPlayableRoute(routeRecord)) {
      activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
      playbackStopIndex = index;
      setPlaybackStop(index, false);
    } else if (window.renderActiveStopDetail) {
      window.renderActiveStopDetail(point, routeRecord, { index, total: 1 });
    }
  });
  return marker;
}

function drawRouteGeometry(group, route, routeRecord) {
  if (routeRecord?.marker_only) return null;
  if (!route || !Array.isArray(route.features) || route.features.length === 0) return null;
  return L.geoJSON(route, {
    filter: feature => feature.geometry && feature.geometry.type !== 'Point',
    style: () => routeLineStyle(routeRecord)
  }).addTo(group);
}

function drawPoiMarkers(group, pois, routeRecord) {
  const latLngs = [];
  (pois || []).forEach((point, index) => {
    if (!Number.isFinite(Number(point.lat)) || !Number.isFinite(Number(point.lon))) return;
    const latLng = [Number(point.lat), Number(point.lon)];
    latLngs.push(latLng);
    pointToMarker(point, index, routeRecord).addTo(group);
  });
  return latLngs;
}

function hotelMarkerPopup(hotel, selected = false) {
  const status = selected || hotel.selected ? 'Selected by optimizer' : 'Candidate alternative';
  const score = Number(hotel.hotel_score ?? hotel.score ?? 0);
  const rank = hotel.candidate_rank ? `<br>Rank: ${hotel.candidate_rank}` : '';
  const scoreLine = Number.isFinite(score) ? `<br>Score: ${score.toFixed(2)}` : '';
  const sourceValue = hotel.source || hotel.source_list || '';
  const source = sourceValue ? `<br>Source: ${sourceValue}` : '';
  const reason = hotel.selected_hotel_reason ? `<br>Reason: ${hotel.selected_hotel_reason}` : '';
  const distance = Number.isFinite(Number(hotel.mean_distance_to_selected_stops_km))
    ? `<br>Avg distance to selected stops: ${Number(hotel.mean_distance_to_selected_stops_km).toFixed(2)} km`
    : '';
  return `<b>${hotel.hotel_name || hotel.name || 'Hotel'}</b><br>${hotel.city || ''}<br>${status}${rank}${scoreLine}${source}${distance}${reason}`;
}

function drawSelectedHotels(payload) {
  if (selectedHotelLayer && dashboardMap.hasLayer(selectedHotelLayer)) {
    dashboardMap.removeLayer(selectedHotelLayer);
  }
  selectedHotelLayer = L.layerGroup();
  (payload?.items || []).forEach(hotel => {
    if (!Number.isFinite(Number(hotel.lat)) || !Number.isFinite(Number(hotel.lon))) return;
    const marker = L.circleMarker([Number(hotel.lat), Number(hotel.lon)], {
      radius: 8,
      color: '#6d4fd7',
      fillColor: '#d8cffd',
      fillOpacity: 0.95,
      weight: 3
    });
    marker.bindPopup(hotelMarkerPopup(hotel, true));
    marker.addTo(selectedHotelLayer);
  });
  if (selectedHotelsVisible && selectedHotelLayer.getLayers().length) {
    selectedHotelLayer.addTo(dashboardMap);
  }
  updateLayerToggleState();
}

function focusDashboardLocation(lat, lon, label = 'Selected location') {
  if (!dashboardMap || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return;
  dashboardMap.flyTo([Number(lat), Number(lon)], Math.max(dashboardMap.getZoom(), 8), { duration: 0.65 });
  L.popup({ closeButton: true, autoClose: true })
    .setLatLng([Number(lat), Number(lon)])
    .setContent(`<b>${label}</b>`)
    .openOn(dashboardMap);
}

function toggleSelectedHotels(forceVisible = null) {
  selectedHotelsVisible = forceVisible === null ? !selectedHotelsVisible : Boolean(forceVisible);
  if (!selectedHotelLayer) {
    setDashboardStatus('No selected hotel markers were exported for this route.');
    return;
  }
  if (selectedHotelsVisible) {
    if (!dashboardMap.hasLayer(selectedHotelLayer)) selectedHotelLayer.addTo(dashboardMap);
    setDashboardStatus('Selected hotel markers shown.');
  } else {
    if (dashboardMap.hasLayer(selectedHotelLayer)) dashboardMap.removeLayer(selectedHotelLayer);
    setDashboardStatus('Selected hotel markers hidden.');
  }
  updateLayerToggleState();
}

function extendBoundsFromLayer(bounds, layer) {
  if (!layer) return bounds;
  try {
    if (typeof layer.getBounds === 'function') {
      const layerBounds = layer.getBounds();
      if (layerBounds?.isValid && layerBounds.isValid()) bounds.extend(layerBounds);
      return bounds;
    }
    if (typeof layer.getLatLng === 'function') {
      bounds.extend(layer.getLatLng());
      return bounds;
    }
    if (typeof layer.eachLayer === 'function') {
      layer.eachLayer(child => extendBoundsFromLayer(bounds, child));
    }
  } catch (error) {
    dashboardLogError('layer bounds skipped', error);
  }
  return bounds;
}

function fitVisibleRoutes() {
  try {
    const groups = [...activeRouteIds].map(id => loadedRouteLayers.get(id)?.group).filter(Boolean);
    if (!groups.length) {
      setDashboardStatus('No visible routes to zoom.');
      return;
    }
    const bounds = L.latLngBounds([]);
    groups.forEach(group => extendBoundsFromLayer(bounds, group));
    if (bounds.isValid()) {
      dashboardMap.fitBounds(bounds, {
        paddingTopLeft: [430, 80],
        paddingBottomRight: [250, 120],
        maxZoom: 7
      });
      return;
    }
  } catch (error) {
    dashboardLogError('fitBounds failed', error);
  }
}

function initializeBaseMapLayers() {
  terrainBaseLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18,
    attribution: 'Tiles &copy; Esri &mdash; Sources: Esri, USGS, NOAA'
  }).addTo(dashboardMap);
  roadsOverlayLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    opacity: 0.34,
    zIndex: 220,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(dashboardMap);
}

function mapLayerToggleChecked(layerName, fallback = true) {
  const input = document.querySelector(`[data-map-layer-toggle="${layerName}"]`);
  return input ? Boolean(input.checked) : fallback;
}

function setLeafletLayerVisible(layer, visible) {
  if (!dashboardMap || !layer) return;
  if (visible && !dashboardMap.hasLayer(layer)) layer.addTo(dashboardMap);
  if (!visible && dashboardMap.hasLayer(layer)) dashboardMap.removeLayer(layer);
}

function selectedContextStops() {
  const routeId = activePlaybackRouteId || defaultRoute(dashboardRouteIndex)?.id;
  const stops = playbackStops(routeId);
  if (Array.isArray(stops) && stops.length) return stops;
  const routes = dashboardPlaybackData?.routes || {};
  const defaultId = defaultRoute(dashboardRouteIndex)?.id;
  return routes[defaultId]?.stops || [];
}

function loadedCandidatePois(routeId) {
  const loaded = loadedRouteLayers.get(routeId);
  return Array.isArray(loaded?.pois) ? loaded.pois : [];
}

function routePointOrdinal(point, index) {
  if (point?.is_route_endpoint || point?.is_airport_endpoint) return point.airport_code || 'AP';
  if (point?.is_hotel_node || point?.node_kind === 'hotel') return 'H';
  return point?.display_sequence_index
    || point?.route_sequence_index
    || point?.selected_stop_index
    || point?.stop_order
    || index + 1;
}

function labelPointHtml(point, index, selected = false) {
  const ordinal = routePointOrdinal(point, index);
  const prefix = selected && !point?.is_route_endpoint && !point?.is_airport_endpoint && !point?.is_hotel_node
    ? `${ordinal}. `
    : (point?.is_route_endpoint || point?.is_airport_endpoint || point?.is_hotel_node ? `${ordinal} ` : '');
  return `${prefix}${point.name || point.hotel_name || point.nature_region || 'Place'}`;
}

function addTextLabel(group, point, className, html) {
  if (!Number.isFinite(Number(point.lat)) || !Number.isFinite(Number(point.lon))) return;
  L.marker([Number(point.lat), Number(point.lon)], {
    icon: L.divIcon({
      className: `map-text-label ${className || ''}`,
      html,
      iconSize: null,
      iconAnchor: [-10, 8]
    }),
    interactive: false,
    zIndexOffset: className === 'selected-label' ? 520 : 260
  }).addTo(group);
}

function buildLabelLayer() {
  const group = L.layerGroup();
  selectedContextStops().filter(point => point?.is_selected_poi || point?.is_route_endpoint || point?.is_airport_endpoint || point?.is_hotel_node).slice(0, 32).forEach((point, index) => {
    addTextLabel(group, point, 'selected-label', labelPointHtml(point, index, true));
  });
  if (!activeRouteIds.has('nature_candidates')) return group;
  const naturePoints = [
    ...loadedCandidatePois('nature_candidates'),
  ];
  const seen = new Set();
  naturePoints.slice(0, 32).forEach(point => {
    const key = `${point.name || point.nature_region || ''}:${point.lat}:${point.lon}`;
    if (seen.has(key)) return;
    seen.add(key);
    addTextLabel(group, point, 'nature-label', labelPointHtml(point, 0, false));
  });
  return group;
}

function riskColor(value) {
  const risk = Number(value || 0);
  if (risk >= 0.75) return '#dc2626';
  if (risk >= 0.50) return '#f97316';
  if (risk >= 0.25) return '#eab308';
  return '#49a33f';
}

function collectWeatherRiskPoints() {
  const items = [
    ...selectedContextStops(),
    ...loadedCandidatePois('nature_candidates'),
    ...loadedCandidatePois('hotel_candidates'),
    ...(Array.isArray(window.dashboardNatureExplore?.items) ? window.dashboardNatureExplore.items : []),
    ...(Array.isArray(window.dashboardSelectedHotels?.items) ? window.dashboardSelectedHotels.items : [])
  ];
  const seen = new Set();
  return items
    .filter(point => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon)))
    .filter(point => {
      const key = `${point.name || point.hotel_name || ''}:${point.lat}:${point.lon}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function buildWeatherRiskLayer() {
  const group = L.layerGroup();
  collectWeatherRiskPoints().forEach(point => {
    const risk = Number(point.weather_risk ?? point.weather_sensitivity ?? 0.12);
    const color = riskColor(risk);
    L.circleMarker([Number(point.lat), Number(point.lon)], {
      radius: 10 + Math.min(8, risk * 8),
      color,
      fillColor: color,
      fillOpacity: 0.14,
      opacity: 0.45,
      weight: 2,
      interactive: false
    }).addTo(group);
  });
  return group;
}

function refreshMapContextLayers() {
  if (!dashboardMap) return;
  if (labelOverlayLayer && dashboardMap.hasLayer(labelOverlayLayer)) dashboardMap.removeLayer(labelOverlayLayer);
  if (weatherRiskLayer && dashboardMap.hasLayer(weatherRiskLayer)) dashboardMap.removeLayer(weatherRiskLayer);
  labelOverlayLayer = buildLabelLayer();
  weatherRiskLayer = buildWeatherRiskLayer();
  setLeafletLayerVisible(labelOverlayLayer, mapLayerToggleChecked('labels', true));
  setLeafletLayerVisible(weatherRiskLayer, mapLayerToggleChecked('weather_risk', true));
}

function updateMapLayerToggleState() {
  document.querySelectorAll('[data-map-layer-toggle]').forEach(input => {
    const action = input.dataset.mapLayerToggle;
    if (action === 'terrain') input.checked = Boolean(terrainBaseLayer && dashboardMap?.hasLayer(terrainBaseLayer));
    if (action === 'roads') input.checked = Boolean(roadsOverlayLayer && dashboardMap?.hasLayer(roadsOverlayLayer));
    if (action === 'labels') input.checked = Boolean(labelOverlayLayer && dashboardMap?.hasLayer(labelOverlayLayer));
    if (action === 'weather_risk') input.checked = Boolean(weatherRiskLayer && dashboardMap?.hasLayer(weatherRiskLayer));
  });
}

function bindMapLayerToggles() {
  document.querySelectorAll('[data-map-layer-toggle]').forEach(input => {
    input.addEventListener('change', event => {
      const action = event.target.dataset.mapLayerToggle;
      const checked = event.target.checked;
      if (action === 'terrain') setLeafletLayerVisible(terrainBaseLayer, checked);
      if (action === 'roads') setLeafletLayerVisible(roadsOverlayLayer, checked);
      if (action === 'labels') {
        if (!labelOverlayLayer) labelOverlayLayer = buildLabelLayer();
        setLeafletLayerVisible(labelOverlayLayer, checked);
      }
      if (action === 'weather_risk') {
        if (!weatherRiskLayer) weatherRiskLayer = buildWeatherRiskLayer();
        setLeafletLayerVisible(weatherRiskLayer, checked);
      }
      updateMapLayerToggleState();
    });
  });
}

function routeById(routeId) {
  if (routeId === 'live_preview') return livePreviewRouteRecord;
  const routes = Array.isArray(dashboardRouteIndex?.routes) ? dashboardRouteIndex.routes : [];
  return routes.find(route => route.id === routeId);
}

async function ensureRouteLayer(routeRecord) {
  if (loadedRouteLayers.has(routeRecord.id)) {
    return loadedRouteLayers.get(routeRecord.id);
  }
  const group = L.layerGroup();
  let route = null;
  let pois = [];
  try {
    route = await loadRouteGeoJson(routeRecord);
  } catch (error) {
    dashboardLogError(`route load failed for ${routeRecord.id}`, error);
  }
  try {
    pois = await loadPoiJson(routeRecord);
  } catch (error) {
    dashboardLogError(`POI load failed for ${routeRecord.id}`, error);
  }
  drawRouteGeometry(group, route, routeRecord);
  const markerLatLngs = drawPoiMarkers(group, pois, routeRecord);
  if (group.getLayers().length === 0 && (!markerLatLngs || markerLatLngs.length === 0)) {
    throw new Error(`No drawable assets for ${routeRecord.id}`);
  }
  const payload = { group, route: route || null, pois: pois || [], record: routeRecord };
  loadedRouteLayers.set(routeRecord.id, payload);
  return payload;
}

function updateSelectorState() {
  document.querySelectorAll('[data-route-toggle]').forEach(input => {
    input.checked = activeRouteIds.has(input.value);
  });
  updateLayerToggleState();
}

function updateLayerToggleState() {
  document.querySelectorAll('[data-layer-toggle]').forEach(input => {
    const action = input.dataset.layerToggle;
    if (action === 'selected_hotels') input.checked = selectedHotelsVisible;
    if (action === 'default_route') input.checked = activeRouteIds.has(defaultRoute(dashboardRouteIndex)?.id);
    if (action === 'hotel_candidates') input.checked = activeRouteIds.has('hotel_candidates');
    if (action === 'nature_candidates') input.checked = activeRouteIds.has('nature_candidates');
    if (action === 'live_preview') input.checked = Boolean(livePreviewLayer && dashboardMap?.hasLayer(livePreviewLayer));
  });
}

async function toggleRoute(routeId, visible, fitAfter = true) {
  const routeRecord = routeById(routeId);
  if (!routeRecord) {
    setDashboardStatus(`Route not found: ${routeId}`, true);
    return;
  }
  try {
    const payload = await ensureRouteLayer(routeRecord);
    if (visible) {
      if (!dashboardMap.hasLayer(payload.group)) payload.group.addTo(dashboardMap);
      activeRouteIds.add(routeId);
      if (isPlayableRoute(routeRecord)) setActivePlaybackRoute(routeId);
      else if (!firstPlayableRouteId()) setActivePlaybackRoute(defaultRoute(dashboardRouteIndex)?.id);
      setDashboardStatus(`Loaded ${routeRecord.label}`);
    } else {
      if (dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
      activeRouteIds.delete(routeId);
      if (activePlaybackRouteId === routeId) setActivePlaybackRoute(firstPlayableRouteId());
      setDashboardStatus(`Hidden ${routeRecord.label}`);
    }
    if (routeId === 'hotel_candidates') hotelCandidatesVisible = activeRouteIds.has(routeId);
    updateSelectorState();
    refreshMapContextLayers();
    updateMapLayerToggleState();
    if (fitAfter) fitVisibleRoutes();
  } catch (error) {
    setDashboardStatus(`Could not load ${routeRecord.label}: ${error.message}`, true);
    showDashboardDiagnostic('error', `route ${routeId} loads`, routeRecord.geojson, error.message);
  }
}

async function showOnlyRoute(routeRecord) {
  for (const routeId of [...activeRouteIds]) {
    const payload = loadedRouteLayers.get(routeId);
    if (payload && dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
    activeRouteIds.delete(routeId);
  }
  await toggleRoute(routeRecord.id, true, true);
}

function clearRoutes() {
  for (const routeId of [...activeRouteIds]) {
    const payload = loadedRouteLayers.get(routeId);
    if (payload && dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
    activeRouteIds.delete(routeId);
  }
  updateSelectorState();
  setActivePlaybackRoute(firstPlayableRouteId(), false);
  setDashboardStatus('Cleared visible routes.');
}

function drawLivePreviewRoute(points, weights = {}) {
  clearLivePreviewRoute(false);
  const drawable = (points || []).filter(point => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon)));
  if (drawable.length === 0) {
    setDashboardStatus('No preview candidates with coordinates were available.', true);
    return;
  }
  let previewStopNumber = 0;
  livePreviewPlaybackStops = drawable.map((point, index) => {
    const isEndpoint = Boolean(point.is_route_endpoint);
    if (!isEndpoint) previewStopNumber += 1;
    const stopNumber = isEndpoint ? (point.stop_order || (point.route_endpoint_side === 'start' ? 0 : 999)) : previewStopNumber;
    return {
      ...point,
      name: point.name || (isEndpoint ? point.airport_code || 'Airport endpoint' : `Preview stop ${stopNumber}`),
      city: point.city || point.nature_region || '',
      day: point.day || Math.max(1, stopNumber || index + 1),
      stop_order: stopNumber,
      type: isEndpoint ? 'airport_endpoint' : (point.type || point.category || point.park_type || (point.nature_region ? 'nature_candidate' : 'preview_candidate')),
      category: isEndpoint ? 'airport_endpoint' : (point.category || point.type || point.park_type || (point.nature_region ? 'nature_candidate' : 'preview_candidate')),
      description: point.description || `${point.name || 'Preview stop'} is a browser-only preview candidate from exported POI data.`,
      why_selected: isEndpoint
        ? 'Preview airport endpoint from the exported segment graph; not counted as a selected POI.'
        : 'Preview only - rerun pipeline to save. This stop is not written back to optimizer artifacts.',
      expected_duration_minutes: point.expected_duration_minutes || 75,
      weather_risk: Number(point.weather_risk || 0.15),
      source_confidence: Number(point.source_confidence || point.data_confidence || 0.5),
      display_utility: Number(point.preview_score || point.display_utility || point.final_poi_value || 0),
      optimization_value_source: isEndpoint ? 'configured_airport_endpoint' : 'browser_preview_score',
      route_sequence_index: isEndpoint ? 0 : stopNumber,
      display_sequence_index: isEndpoint ? (point.airport_code || 'AP') : stopNumber,
      selected_stop_index: isEndpoint ? 0 : stopNumber,
      node_kind: isEndpoint ? 'airport' : 'preview_poi',
      is_airport_endpoint: isEndpoint,
      is_selected_poi: false,
      is_hotel_node: false
    };
  });
  livePreviewRouteRecord = {
    id: 'live_preview',
    label: 'Preview only - rerun pipeline to save',
    family: 'live_preview',
    default: false,
    optional: true,
    playable: true,
    marker_only: false
  };
  livePreviewLayer = L.layerGroup();
  const latLngs = livePreviewPlaybackStops.map(point => [Number(point.lat), Number(point.lon)]);
  if (latLngs.length > 1) {
    L.polyline(latLngs, {
      color: '#f9735b',
      weight: 5,
      opacity: 0.85,
      dashArray: '3 12',
      lineCap: 'round',
      className: 'route-line live-preview-line'
    }).addTo(livePreviewLayer);
  }
  livePreviewPlaybackStops.forEach((point, index) => {
    let marker = null;
    if (point.is_route_endpoint) {
      marker = L.marker([Number(point.lat), Number(point.lon)], {
        icon: L.divIcon({
          className: 'airport-route-endpoint',
          html: `<span>${point.airport_code || 'AP'}</span>`,
          iconSize: null,
          iconAnchor: [19, 12],
          popupAnchor: [0, -14]
        }),
        zIndexOffset: 710
      });
    } else {
      marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
        radius: 6,
        color: '#c94b35',
        fillColor: '#ffb199',
        fillOpacity: 0.88,
        weight: 2
      });
    }
    const score = Number(point.preview_score);
    const label = point.is_route_endpoint ? `${point.airport_code || 'Airport'} · ${point.name || 'Airport'}` : `Preview ${routePointOrdinal(point, index)}. ${point.name || 'Candidate'}`;
    marker.bindPopup(
      `<b>${label}</b><br>${point.city || point.nature_region || ''}<br>Preview score: ${Number.isFinite(score) ? score.toFixed(2) : 'n/a'}<br>Browser preview is approximate; Run All recomputes the real optimized route.`
    );
    marker.on('click', () => {
      playbackStopIndex = index;
      setActivePlaybackRoute('live_preview', false);
      setPlaybackStop(index, false);
    });
    marker.addTo(livePreviewLayer);
  });
  livePreviewLayer.addTo(dashboardMap);
  try {
    const bounds = L.featureGroup(livePreviewLayer.getLayers()).getBounds();
    if (bounds.isValid()) {
      dashboardMap.fitBounds(bounds, {
        paddingTopLeft: [430, 80],
        paddingBottomRight: [250, 120],
        maxZoom: 7
      });
    }
  } catch (error) {
    dashboardLogError('preview route fit failed', error);
  }
  updateLayerToggleState();
  const naturePct = Math.round((Number(weights.nature) || 0) * 100);
  const sequenceStatus = weights.__sequence_feasible === false ? ' Sequence infeasible or missing segment candidates.' : ' Sequence feasible.';
  setActivePlaybackRoute('live_preview', true);
  setDashboardStatus(`Preview only - rerun pipeline to save. Browser-side route rebuilt (${naturePct}% nature).${sequenceStatus}`, weights.__sequence_feasible === false);
}

function clearLivePreviewRoute(updateStatus = true) {
  if (livePreviewLayer && dashboardMap?.hasLayer(livePreviewLayer)) {
    dashboardMap.removeLayer(livePreviewLayer);
  }
  livePreviewLayer = null;
  livePreviewPlaybackStops = [];
  livePreviewRouteRecord = null;
  if (activePlaybackRouteId === 'live_preview') setActivePlaybackRoute(firstPlayableRouteId(), true);
  updateLayerToggleState();
  if (updateStatus) setDashboardStatus('Live preview route cleared.');
}

async function showRouteGroup(groupName) {
  const routes = (dashboardRouteIndex.routes || []).filter(route => {
    const groups = route.quick_groups || [];
    return groups.includes(groupName);
  });
  if (!routes.length) {
    setDashboardStatus(`No route assets found for ${groupName}.`, true);
    return;
  }
  for (const route of routes) {
    await toggleRoute(route.id, true, false);
  }
  fitVisibleRoutes();
  setDashboardStatus(`Loaded ${routes.length} ${groupName.replace('_', '-')} route layer${routes.length === 1 ? '' : 's'}.`);
}

function runQuickAction(action) {
  if (action === 'clear') {
    clearRoutes();
  } else if (action === 'default') {
    const selected = defaultRoute(dashboardRouteIndex);
    if (selected) showOnlyRoute(selected);
  } else if (action === 'zoom_visible') {
    fitVisibleRoutes();
  } else {
    showRouteGroup(action);
  }
}

function toggleHotelCandidates(forceVisible = null) {
  const targetVisible = forceVisible === null ? !activeRouteIds.has('hotel_candidates') : Boolean(forceVisible);
  const routeRecord = routeById('hotel_candidates');
  if (!routeRecord) {
    setDashboardStatus('No hotel candidate assets found. Regenerate dashboard assets.', true);
    return;
  }
  hotelCandidatesVisible = targetVisible;
  toggleRoute('hotel_candidates', targetVisible);
}

function bindLayerToggles() {
  document.querySelectorAll('[data-layer-toggle]').forEach(input => {
    input.addEventListener('change', event => {
      const action = event.target.dataset.layerToggle;
      const checked = event.target.checked;
      if (action === 'default_route') {
        const selected = defaultRoute(dashboardRouteIndex);
        if (!selected) return;
        toggleRoute(selected.id, checked);
      } else if (action === 'selected_hotels') {
        toggleSelectedHotels(checked);
      } else if (action === 'hotel_candidates') {
        toggleHotelCandidates(checked);
      } else if (action === 'nature_candidates') {
        toggleRoute('nature_candidates', checked);
      } else if (action === 'live_preview') {
        if (!checked) clearLivePreviewRoute();
        else if (livePreviewLayer && !dashboardMap.hasLayer(livePreviewLayer)) {
          livePreviewLayer.addTo(dashboardMap);
          setActivePlaybackRoute('live_preview', true);
        } else if (!livePreviewLayer) {
          setDashboardStatus('No preview route is available yet. Drag interest bars or build a preview route first.');
          event.target.checked = false;
        }
      }
    });
  });
  updateLayerToggleState();
}

function renderQuickButtons(index) {
  const target = document.getElementById('quick-actions');
  if (!target) return;
  const routes = Array.isArray(index.routes) ? index.routes : [];
  const availableGroups = new Set(routes.flatMap(route => route.quick_groups || []));
  const buttonSpecs = [
    ['clear', 'Clear'],
    ['default', 'Saved route'],
    ['days_7', '7-day'],
    ['days_9', '9-day'],
    ['days_12', '12-day'],
    ['balanced', 'Balanced'],
    ['zoom_visible', 'Zoom visible']
  ];
  target.innerHTML = '';
  buttonSpecs.forEach(([action, label]) => {
    if (!['clear', 'default', 'zoom_visible'].includes(action) && !availableGroups.has(action)) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.dataset.quickAction = action;
    button.addEventListener('click', () => runQuickAction(action));
    target.appendChild(button);
  });
}

function renderRouteSelector(index) {
  const target = document.getElementById('route-selector');
  if (!target) return;
  const routes = Array.isArray(index.routes) ? index.routes : [];
  target.innerHTML = '';
  const groups = new Map();
  routes.forEach(route => {
    const groupName = route.selector_group || route.family || 'routes';
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(route);
  });
  groups.forEach((groupRoutes, groupName) => {
    const title = document.createElement('div');
    title.className = 'route-group-title';
    title.textContent = groupName.replaceAll('_', ' ');
    target.appendChild(title);
    groupRoutes.forEach(route => {
      const row = document.createElement('div');
      row.className = 'route-row';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = route.id;
      input.checked = Boolean(route.default);
      input.dataset.routeToggle = 'true';
      input.addEventListener('change', event => toggleRoute(route.id, event.target.checked));
      const label = document.createElement('label');
      label.append(document.createTextNode(route.label || route.id));
      const chip = document.createElement('span');
      chip.className = 'layer-chip';
      chip.textContent = route.family || 'route';
      label.append(chip);
      const meta = document.createElement('small');
      const bits = [route.trip_days ? `${route.trip_days} days` : '', route.method || '', route.profile || ''].filter(Boolean);
      meta.textContent = bits.join(' · ') || (route.optional ? 'Hidden until selected' : 'Visible on load');
      label.append(meta);
      row.append(input, label);
      target.appendChild(row);
    });
  });
}

function loadOptionalLayers() {
  const optionalRoutes = (dashboardRouteIndex?.routes || []).filter(route => route.optional);
  if (!optionalRoutes.length) {
    setDashboardStatus('No optional layer assets found. Regenerate dashboard assets.', true);
    return;
  }
  setDashboardStatus(`${optionalRoutes.length} optional layers are available in the route selector.`);
}

async function drawDefaultRoute(routeRecord) {
  await showOnlyRoute(routeRecord);
  setDashboardStatus(`Loaded default route: ${routeRecord.label || routeRecord.id}`);
}

async function initMap() {
  dashboardLogInit('start');
  try {
    assertMapContainerReady();
    dashboardMap = L.map("map", { preferCanvas: true }).setView([36.5, -119.5], 6);
    window.dashboardMap = dashboardMap;
    initializeBaseMapLayers();

    const index = await loadRouteIndex();
    const metrics = await loadDashboardMetrics();
    const debugSummary = await loadDebugSummary().catch(error => {
      dashboardLogError('debug summary load failed', error);
      return { available: false, message: error.message };
    });
    const interestPreview = await loadInterestPreview().catch(error => {
      dashboardLogError('interest preview load failed', error);
      return { available: false, message: error.message };
    });
    const playbackData = await loadPlaybackData().catch(error => {
      dashboardLogError('playback data load failed', error);
      return { available: false, routes: {}, message: error.message };
    });
    const cityDetails = await loadCityDetails().catch(error => {
      dashboardLogError('city details load failed', error);
      return { available: false, cities: [], message: error.message };
    });
    const selectedHotels = await loadSelectedHotels().catch(error => {
      dashboardLogError('selected hotels load failed', error);
      return { available: false, items: [], message: error.message };
    });
    const hotelChoices = await loadHotelChoices().catch(error => {
      dashboardLogError('hotel choices load failed', error);
      return { available: false, cities: {}, message: error.message };
    });
    const natureExplore = await loadNatureExplore().catch(error => {
      dashboardLogError('nature explore load failed', error);
      return { available: false, items: [], message: error.message };
    });
    window.dashboardMetrics = metrics;
    window.dashboardRouteIndex = index;
    window.dashboardPlaybackData = playbackData;
    window.dashboardNatureExplore = natureExplore;
    window.dashboardSelectedHotels = selectedHotels;
    dashboardRouteIndex = index;
    dashboardPlaybackData = playbackData;
    if (window.bindDashboardShellControls) {
      window.bindDashboardShellControls();
    }
    if (window.renderDashboardMetrics) {
      window.renderDashboardMetrics(metrics);
    }
    if (window.renderDebugSummary) {
      window.renderDebugSummary(debugSummary);
    }
    if (window.renderInterestPreview) {
      window.renderInterestPreview(interestPreview);
    }
    if (window.renderCityDetails) {
      window.renderCityDetails(cityDetails);
    }
    if (window.renderHotelChoices) {
      window.renderHotelChoices(hotelChoices);
    }
    if (window.renderNatureExplore) {
      window.renderNatureExplore(natureExplore);
    }
    drawSelectedHotels(selectedHotels);
    bindPlaybackControls();
    if (window.renderCustomerControls) {
      window.renderCustomerControls(index);
    }
    renderQuickButtons(index);
    renderRouteSelector(index);
    bindLayerToggles();
    bindMapLayerToggles();

    const selected = defaultRoute(index);
    if (!selected) {
      showDashboardDiagnostic('error', 'default route exists', 'assets/route_index.json', 'No default route assets found. Regenerate dashboard assets.');
      throw new Error('No default route found');
    }
    await drawDefaultRoute(selected);
    refreshMapContextLayers();
    updateMapLayerToggleState();
    if (document.body.dataset.dashboardMode === 'customer') {
      setDashboardStatus('Customer planner ready. Changes load saved artifacts or preview-only routes.');
    } else {
      loadOptionalLayers();
    }
    dashboardLogInit('complete');
  } catch (error) {
    dashboardLogError('initialization failed', error);
    if (!document.getElementById('diagnostic-panel')?.textContent) {
      showDashboardDiagnostic('error', 'dashboard initialization', window.location.href, error.message);
    }
  }
}

window.drawDefaultRoute = drawDefaultRoute;
window.renderRouteSelector = renderRouteSelector;
window.activeRouteIds = activeRouteIds;
window.toggleRoute = toggleRoute;
window.toggleSelectedHotels = toggleSelectedHotels;
window.toggleHotelCandidates = toggleHotelCandidates;
window.showOnlyRoute = showOnlyRoute;
window.setActivePlaybackRoute = setActivePlaybackRoute;
window.runQuickAction = runQuickAction;
window.showRouteGroup = showRouteGroup;
window.zoomVisibleRoutes = fitVisibleRoutes;
window.loadOptionalLayers = loadOptionalLayers;
window.drawLivePreviewRoute = drawLivePreviewRoute;
window.clearLivePreviewRoute = clearLivePreviewRoute;
window.focusDashboardLocation = focusDashboardLocation;
window.playRouteAnimation = playRouteAnimation;
window.pauseRouteAnimation = pauseRouteAnimation;
window.restartRouteAnimation = restartRouteAnimation;
window.setPlaybackStop = setPlaybackStop;
window.addEventListener('DOMContentLoaded', initMap);
