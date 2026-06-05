function renderDashboardMetrics(metrics) {
  const target = document.getElementById('metrics');
  if (!target) return;
  const subtitle = document.getElementById('dashboard-subtitle');
  if (subtitle) subtitle.textContent = metrics.scenario_label || metrics.scenario || 'Itinerary scenario';
  target.innerHTML = `
    <div class="metrics-grid">
      <div class="metric"><b>${metrics.trip_days || '7'}</b>Days</div>
      <div class="metric"><b>${metrics.selected_stop_count}</b>Stops</div>
      <div class="metric"><b>${metrics.display_utility ?? metrics.interest_adjusted_utility}</b>Optimizer value</div>
      <div class="metric"><b>${metrics.nature_optimization_enabled ? 'Active' : 'Off'}</b>Nature mode</div>
    </div>
    <div class="summary-list">
      <div><b>Scenario:</b> ${metrics.scenario}</div>
      <div><b>Interest:</b> ${metrics.interest_profile}</div>
      <div><b>Value column:</b> ${metrics.optimization_value_column || 'final_poi_value'}</div>
      <div><b>Route state:</b> ${metrics.route_state_label || 'Saved optimized route'}</div>
      <div><b>Default route:</b> ${metrics.default_route_label}</div>
      <div><b>Route method:</b> ${metrics.default_route_method || 'selected artifact'}</div>
      <div><b>Artifact freshness:</b> ${metrics.artifact_metadata_fresh ? 'fresh metadata match' : 'metadata missing/stale'}</div>
      <div><b>Anchor audit:</b> ${metrics.audit_readiness || 'missing'}${metrics.audit_missing_count ? ` · ${metrics.audit_missing_count} missing` : ''}</div>
      <div><b>Indexed layers:</b> ${metrics.route_record_count || 1}</div>
    </div>
    <div class="interest-note">Default route is computed by Python optimization using nature-aware value weights.</div>
    <div class="interest-note">${metrics.preview_state_label || 'Preview only - rerun pipeline to save'}.</div>
  `;
}

function renderObjectSummary(targetId, title, payload) {
  const target = document.getElementById(targetId);
  if (!target) return;
  if (!payload || payload.available === false) {
    target.innerHTML = `<div class="summary-list">${payload?.message || `No ${title} artifact available.`}</div>`;
    return;
  }
  const rows = Object.entries(payload)
    .filter(([key]) => !['available', 'items'].includes(key))
    .slice(0, 8)
    .map(([key, value]) => `<div><b>${key.replaceAll('_', ' ')}:</b> ${typeof value === 'object' ? JSON.stringify(value).slice(0, 160) : value}</div>`)
    .join('');
  target.innerHTML = `<div class="summary-list">${rows || `${title} loaded.`}</div>`;
}

function compactValue(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function bindDashboardShellControls() {
  const panel = document.querySelector('.dashboard-panel');
  const handle = document.getElementById('dashboard-drag-handle');
  const collapse = document.getElementById('dashboard-collapse');
  const modeToggle = document.getElementById('dashboard-mode-toggle');
  if (!panel || !handle) return;
  collapse?.addEventListener('click', event => {
    event.stopPropagation();
    panel.classList.toggle('collapsed');
    collapse.textContent = panel.classList.contains('collapsed') ? 'Expand' : 'Collapse';
  });
  const applyMode = mode => {
    const nextMode = mode === 'customer' ? 'customer' : 'research';
    document.body.dataset.dashboardMode = nextMode;
    document.body.classList.toggle('customer-dashboard', nextMode === 'customer');
    document.body.classList.toggle('research-dashboard', nextMode === 'research');
    const modeLabel = document.getElementById('dashboard-mode-label');
    if (modeLabel) {
      modeLabel.textContent = nextMode === 'customer' ? 'Customer trip planner' : 'Research/Test dashboard';
    }
    if (modeToggle) {
      modeToggle.textContent = nextMode === 'customer' ? 'Research/Test mode' : 'Customer mode';
      modeToggle.setAttribute('aria-pressed', nextMode === 'customer' ? 'true' : 'false');
    }
    if (window.dashboardMap?.invalidateSize) {
      window.setTimeout(() => window.dashboardMap.invalidateSize(), 80);
    }
  };
  applyMode(document.body.dataset.dashboardMode || 'research');
  modeToggle?.addEventListener('click', event => {
    event.stopPropagation();
    const current = document.body.dataset.dashboardMode === 'customer' ? 'customer' : 'research';
    const next = current === 'customer' ? 'research' : 'customer';
    applyMode(next);
    const status = document.getElementById('dashboard-status');
    if (status) {
      status.textContent = next === 'customer'
        ? 'Customer planner mode: choices load saved artifacts or preview-only routes.'
        : 'Research/Test mode: all comparison, debug, and evaluation controls are visible.';
      status.style.color = '#0a6b53';
    }
  });
  window.applyDashboardMode = applyMode;
  let dragState = null;
  handle.addEventListener('pointerdown', event => {
    if (event.target?.tagName === 'BUTTON') return;
    const rect = panel.getBoundingClientRect();
    dragState = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    };
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.right = 'auto';
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener('pointermove', event => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth - 8);
    const maxTop = Math.max(0, window.innerHeight - 52);
    const left = Math.max(8, Math.min(maxLeft, event.clientX - dragState.offsetX));
    const top = Math.max(8, Math.min(maxTop, event.clientY - dragState.offsetY));
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  });
  handle.addEventListener('pointerup', event => {
    if (dragState?.pointerId === event.pointerId) dragState = null;
  });
  handle.addEventListener('pointercancel', () => {
    dragState = null;
  });
}

function renderActiveStopDetail(stop, routeRecord, playbackMeta = {}) {
  const target = document.getElementById('active-stop-detail');
  if (!target) return;
  if (!stop) {
    target.innerHTML = '<div class="summary-list">Click a marker or start playback to inspect a stop.</div>';
    return;
  }
  const rows = [
    ['Route', routeRecord?.label || ''],
    ['Playback', playbackMeta.total ? `${routePointOrdinal(stop, playbackMeta.index || 0)} of ${playbackMeta.total}` : ''],
    ['Day', stop.day || ''],
    ['City', stop.city || ''],
    ['City/anchor', stop.city_or_anchor || ''],
    ['Overnight hotel', stop.hotel_name || ''],
    ['Type', stop.type || stop.category || ''],
    ['Category', stop.category || ''],
    ['Nature region', stop.nature_region || ''],
    ['Expected duration', stop.expected_duration_minutes ? `${stop.expected_duration_minutes} min` : ''],
    ['Weather risk', compactValue(stop.weather_risk)],
    ['Source confidence', compactValue(stop.source_confidence)],
    ['Optimizer value', compactValue(stop.display_utility ?? stop.interest_adjusted_value ?? stop.final_poi_value)],
    ['Final POI value', compactValue(stop.final_poi_value)],
    ['Interest-adjusted value', Number(stop.interest_adjusted_value) > 0 ? compactValue(stop.interest_adjusted_value) : ''],
    ['Value source', stop.optimization_value_source || ''],
    ['Nature score', compactValue(stop.nature_score)],
    ['Scenic score', compactValue(stop.scenic_score)],
    ['Weather sensitivity', compactValue(stop.weather_sensitivity)],
    ['Source', stop.source_list || '']
  ].filter(([, value]) => value !== '' && value !== undefined && value !== null);
  const keyRows = rows.filter(([key]) => ['City/anchor', 'Type', 'Expected duration', 'Weather risk', 'Source confidence', 'Optimizer value', 'Nature score', 'Scenic score'].includes(key));
  const imageUrl = String(stop.image_url || '').trim();
  const visual = imageUrl
    ? `<div class="stop-visual" role="img" aria-label="${escapeHtml(stop.name || 'Place')} image"><img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(stop.name || 'Place')}"></div>`
    : '<div class="stop-visual" role="img" aria-label="Scenic landscape preview"></div>';
  const sourceLink = stop.source_url || stop.website_url
    ? `<div class="detail-copy"><b>Source</b><a href="${escapeHtml(stop.source_url || stop.website_url)}" target="_blank" rel="noopener">${escapeHtml(stop.detail_source || stop.source_url || stop.website_url)}</a></div>`
    : '';
  target.innerHTML = `
    <div class="detail-card active-stop-card">
      ${visual}
      <div class="active-stop-body">
        <div class="active-stop-title-row">
          <strong>${escapeHtml(stop.name || 'Stop')}</strong>
          <span class="day-pill">Day ${stop.day || playbackMeta.index + 1 || 1}</span>
        </div>
        <div class="active-stop-grid">
          ${keyRows.map(([key, value]) => `<div><b>${escapeHtml(key)}</b>${escapeHtml(value)}</div>`).join('')}
        </div>
        <div class="detail-copy"><b>Description</b>${escapeHtml(stop.description || 'No description exported.')}</div>
        <div class="detail-copy"><b>Why selected</b>${escapeHtml(stop.why_selected || stop.reason_selected || 'Selection reason was not exported.')}</div>
        ${sourceLink}
      </div>
    </div>
  `;
}

function renderCityDetails(payload) {
  const target = document.getElementById('city-details');
  if (!target) return;
  const cities = payload?.cities || [];
  const legs = payload?.intercity_legs || [];
  if (!cities.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No city detail artifact available.'}</div>`;
    return;
  }
  target.innerHTML = cities.slice(0, 10).map(city => {
    const cityLegs = legs
      .filter(leg => leg.from === city.city || leg.to === city.city)
      .slice(0, 3)
      .map(leg => `${leg.from} → ${leg.to} (${compactValue(leg.estimated_drive_minutes, 0)} min)`);
    const stops = (city.selected_stops || []).slice(0, 5).map(stop => `
      <li>
        <button type="button" data-focus-city-stop="${stop.name || ''}" data-lat="${stop.lat || ''}" data-lon="${stop.lon || ''}">
          ${stop.day ? `D${stop.day}` : 'Stop'} · ${stop.name || 'Stop'}
        </button>
      </li>
    `).join('');
    return `
      <div class="city-card">
        <div class="city-card-header">
          <strong>${city.city}</strong>
          <button type="button" data-focus-city="${city.city}" data-lat="${city.lat || ''}" data-lon="${city.lon || ''}">Focus</button>
        </div>
        <div>Days: ${(city.days || []).join(', ') || 'n/a'}</div>
        <div>Selected hotel: ${city.selected_hotel || (city.hotels || [])[0] || 'n/a'}</div>
        <div>Hotel options: ${city.hotel_alternative_count || 0}</div>
        <div>Travel legs: ${cityLegs.join('; ') || 'n/a'}</div>
        <div>Top nature/scenic: ${compactValue(city.nature_score)} / ${compactValue(city.scenic_score)}</div>
        <ul class="city-stop-list">${stops || '<li>No selected route stops exported for this city.</li>'}</ul>
      </div>
    `;
  }).join('');
  target.querySelectorAll('[data-focus-city], [data-focus-city-stop]').forEach(button => {
    button.addEventListener('click', () => {
      const lat = Number(button.dataset.lat);
      const lon = Number(button.dataset.lon);
      if (window.focusDashboardLocation && Number.isFinite(lat) && Number.isFinite(lon)) {
        window.focusDashboardLocation(lat, lon, button.dataset.focusCity || button.dataset.focusCityStop || 'City detail');
      }
    });
  });
}

function hotelPreferenceKey(city) {
  return `dashboard-preferred-hotel:${city}`;
}

function preferredHotel(city) {
  try {
    return window.localStorage.getItem(hotelPreferenceKey(city));
  } catch {
    return '';
  }
}

function setPreferredHotel(city, hotelName) {
  try {
    window.localStorage.setItem(hotelPreferenceKey(city), hotelName);
  } catch {
    return;
  }
}

function clearPreferredHotel(city) {
  try {
    window.localStorage.removeItem(hotelPreferenceKey(city));
  } catch {
    return;
  }
}

function preferredHotelPayload(payload) {
  const cities = payload?.cities || {};
  const preferred_hotels = {};
  Object.keys(cities).forEach(city => {
    const hotel = preferredHotel(city);
    if (hotel) preferred_hotels[city] = hotel;
  });
  return { hotels: { preferred_hotels, preferred_hotel_bonus: 3.0 } };
}

function downloadPreferredHotelConfig(payload) {
  const blob = new Blob([JSON.stringify(preferredHotelPayload(payload), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'preferred_hotels_config.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderHotelChoices(payload) {
  const target = document.getElementById('hotel-choices');
  if (!target) return;
  const cities = payload?.cities || {};
  const cityNames = Object.keys(cities).sort();
  if (!cityNames.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No hotel candidates found.'}</div>`;
    return;
  }
  target.innerHTML = `
    <div class="filter-row">
      <button type="button" data-toggle-hotel-candidates="true">Show hotel candidates</button>
      <button type="button" data-export-hotel-preferences="true">Export preferences</button>
    </div>
    <div class="interest-note">Preferred hotels are reversible here and can be exported into config as a soft bonus for the next optimizer rerun.</div>
  ` + cityNames.map(city => {
    const preferred = preferredHotel(city);
    const cityHotels = cities[city] || [];
    const limited = cityHotels.length <= 1 && cityHotels.some(hotel => hotel.source === 'preview');
    const cards = cityHotels.slice(0, 6).map(hotel => {
      const isPreferred = preferred && preferred === hotel.hotel_name;
      const classes = ['hotel-card', hotel.selected ? 'selected' : '', isPreferred ? 'preferred' : ''].join(' ');
      return `
        <div class="${classes}">
          <strong>${hotel.hotel_name}</strong>
          <div>Rank ${hotel.candidate_rank || 'n/a'} · Score ${compactValue(hotel.hotel_score)}</div>
          <div>${hotel.selected ? 'Selected by optimizer' : 'Candidate alternative'} · ${hotel.source || ''}</div>
          <div>${hotel.selected_hotel_reason || ''}</div>
          <div class="hotel-actions">
            <button type="button" data-prefer-hotel="${hotel.hotel_name}" data-hotel-city="${city}">${isPreferred ? 'Preferred' : 'Prefer'}</button>
            ${isPreferred ? `<button type="button" data-clear-hotel-preference="${city}">Clear</button>` : ''}
          </div>
        </div>
      `;
    }).join('');
    const limitedNote = limited ? '<div class="interest-note">Limited hotel data: only preview fallback exported for this city. Rerun with the hotel catalog to show more options.</div>' : '';
    return `<div class="city-card"><strong>${city}</strong>${limitedNote}${cards}</div>`;
  }).join('');
  const candidateButton = target.querySelector('[data-toggle-hotel-candidates]');
  candidateButton?.addEventListener('click', () => {
    if (window.toggleHotelCandidates) {
      const nextVisible = !(window.activeRouteIds?.has && window.activeRouteIds.has('hotel_candidates'));
      window.toggleHotelCandidates(nextVisible);
      candidateButton.textContent = nextVisible ? 'Hide hotel candidates' : 'Show hotel candidates';
    }
  });
  target.querySelectorAll('[data-prefer-hotel]').forEach(button => {
    button.addEventListener('click', () => {
      setPreferredHotel(button.dataset.hotelCity, button.dataset.preferHotel);
      renderHotelChoices(payload);
    });
  });
  target.querySelectorAll('[data-clear-hotel-preference]').forEach(button => {
    button.addEventListener('click', () => {
      clearPreferredHotel(button.dataset.clearHotelPreference);
      renderHotelChoices(payload);
    });
  });
  target.querySelector('[data-export-hotel-preferences]')?.addEventListener('click', () => {
    downloadPreferredHotelConfig(payload);
  });
}

function natureMatchesFilters(item, filters) {
  if (!filters.size) return true;
  if (filters.has('national_park') && item.is_national_park) return true;
  if (filters.has('state_or_protected') && (item.is_state_park || item.is_protected_area)) return true;
  if (filters.has('scenic_viewpoint') && item.is_scenic_viewpoint) return true;
  if (filters.has('hiking') && item.is_hiking) return true;
  return false;
}

function renderNatureExplore(payload) {
  const target = document.getElementById('nature-explore');
  if (!target) return;
  const items = payload?.items || [];
  if (!items.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No nature candidates found.'}</div>`;
    return;
  }
  target.innerHTML = `
    <div class="filter-row">
      <button type="button" data-show-nature-layer="true">Show nature layer</button>
      <label><input type="checkbox" data-nature-filter="national_park"> National parks</label>
      <label><input type="checkbox" data-nature-filter="state_or_protected"> Protected</label>
      <label><input type="checkbox" data-nature-filter="scenic_viewpoint"> Scenic</label>
      <label><input type="checkbox" data-nature-filter="hiking"> Hiking</label>
    </div>
    <div id="nature-card-list"></div>
  `;
  const renderCards = () => {
    const filters = new Set([...target.querySelectorAll('[data-nature-filter]:checked')].map(input => input.dataset.natureFilter));
    const filtered = items.filter(item => natureMatchesFilters(item, filters)).slice(0, 12);
    target.querySelector('#nature-card-list').innerHTML = filtered.map(item => `
      <div class="nature-card">
        <strong>${item.name}</strong>
        <div>${item.city || item.nature_region || ''} · ${item.park_type || item.category || ''}</div>
        <div>Nature/scenic/hiking: ${compactValue(item.nature_score)} / ${compactValue(item.scenic_score)} / ${compactValue(item.hiking_score)}</div>
        <div>Weather sensitivity: ${compactValue(item.weather_sensitivity)} · Season risk: ${compactValue(item.seasonality_risk)}</div>
        <div>${item.source_list || ''}</div>
      </div>
    `).join('');
  };
  target.querySelectorAll('[data-nature-filter]').forEach(input => input.addEventListener('change', renderCards));
  target.querySelector('[data-show-nature-layer]')?.addEventListener('click', () => {
    if (window.toggleRoute) window.toggleRoute('nature_candidates', true);
  });
  renderCards();
}

function renderDebugSummary(summary) {
  renderObjectSummary('debug-summary', 'debug summary', summary);
}

function interestWeightsFromPreview(preview) {
  const fromWeights = preview?.weights || {};
  const fallback = { nature: 0.25, city: 0.25, culture: 0.25, history: 0.25 };
  return {
    nature: Number(fromWeights.nature ?? fallback.nature),
    city: Number(fromWeights.city ?? fallback.city),
    culture: Number(fromWeights.culture ?? fallback.culture),
    history: Number(fromWeights.history ?? fallback.history)
  };
}

function normalizeInterestBarValues(raw) {
  const axes = ['nature', 'city', 'culture', 'history'];
  const total = axes.reduce((sum, axis) => sum + Math.max(0, Number(raw[axis]) || 0), 0);
  if (total <= 0) return { nature: 0.25, city: 0.25, culture: 0.25, history: 0.25 };
  return Object.fromEntries(axes.map(axis => [axis, Math.max(0, Number(raw[axis]) || 0) / total]));
}

function nearestInterestProfile(preview, weights) {
  const presets = preview?.preset_weights || {};
  const threshold = Number(preview?.saved_profile_match_threshold ?? 0.12);
  let best = { profile: '', distance: Number.POSITIVE_INFINITY, threshold, weights: null };
  Object.entries(presets).forEach(([profile, presetWeights]) => {
    const distance = ['nature', 'city', 'culture', 'history']
      .reduce((sum, axis) => sum + Math.abs(Number(weights[axis] || 0) - Number(presetWeights?.[axis] || 0)), 0);
    if (distance < best.distance) best = { profile, distance, threshold, weights: presetWeights };
  });
  best.matched = Boolean(best.profile) && best.distance <= threshold;
  return best;
}

function savedInterestProfileRoute(profile) {
  const routes = Array.isArray(window.dashboardRouteIndex?.routes) ? window.dashboardRouteIndex.routes : [];
  return routes.find(route =>
    route.family === 'interest_profile'
    && String(route.interest_profile || route.profile || '').toLowerCase() === String(profile || '').toLowerCase()
  );
}

function previewGraphForWeights(preview, weights) {
  const match = nearestInterestProfile(preview, weights);
  const graphs = preview?.preview_route_graphs || {};
  return {
    match,
    graph: graphs[match.profile] || preview?.preview_route_graph || {}
  };
}

function previewItemAxis(item, axis) {
  if (axis === 'city') return Number(item.city_axis ?? item.city ?? item.city_score ?? 0);
  return Number(item[axis] ?? item[`${axis}_score`] ?? 0);
}

function previewAxisFit(item, weights) {
  return ['nature', 'city', 'culture', 'history']
    .reduce((sum, axis) => sum + (weights[axis] || 0) * Math.max(0, previewItemAxis(item, axis) || 0), 0);
}

function dominantPreviewAxis(item) {
  const axes = ['nature', 'city', 'culture', 'history'];
  return axes
    .map(axis => ({ axis, value: Math.max(0, previewItemAxis(item, axis) || 0) }))
    .sort((a, b) => b.value - a.value)[0] || { axis: 'nature', value: 0 };
}

function candidateAllowedByWeights(item, weights) {
  const dominant = dominantPreviewAxis(item);
  const dominantWeight = Number(weights[dominant.axis] || 0);
  const nonNatureFit = ['city', 'culture', 'history']
    .reduce((sum, axis) => sum + Number(weights[axis] || 0) * Math.max(0, previewItemAxis(item, axis) || 0), 0);
  const natureSignal = Math.max(
    Number(item.nature_score || 0),
    Number(item.scenic_score || 0),
    Number(item.park_bonus || 0),
    item.is_national_park ? 1 : 0
  );
  if (Number(weights.nature || 0) <= 0.001 && natureSignal >= 0.35 && nonNatureFit < 0.18) return false;
  if (dominant.value >= 0.35 && dominantWeight <= 0.001) return false;
  return previewAxisFit(item, weights) > 0.02 || dominant.value < 0.2;
}

function previewScore(item, weights, lambdas) {
  const fit = previewAxisFit(item, weights);
  const base = Math.min(Number(item.final_poi_value ?? item.interest_adjusted_value ?? 0) || 0, 120) / 120;
  const park = Number(weights.nature || 0) * Number(item.park_bonus || 0);
  const weather = Number(item.weather_sensitivity || 0) * Number(item.weather_risk || 0.15);
  const season = Number(item.seasonality_risk || 0);
  const detour = Number(item.detour_minutes || 0);
  return base
    + Number(lambdas.lambda_fit ?? 0.65) * fit
    + Number(lambdas.lambda_park ?? 0.35) * park
    - Number(lambdas.lambda_weather ?? 0.30) * weather
    - Number(lambdas.lambda_season ?? 0.20) * season
    - Number(lambdas.lambda_detour ?? 0.006) * detour;
}

function renderInterestPreview(preview) {
  const target = document.getElementById('interest-preview');
  if (!target) return;
  if (!preview || preview.available === false) {
    target.innerHTML = `<div class="summary-list">${preview?.message || 'No interest preview artifact available.'}</div>`;
    return;
  }
  const axes = ['nature', 'city', 'culture', 'history'];
  const weights = interestWeightsFromPreview(preview);
  target.innerHTML = `
    <div id="interest-bar-controls">
      ${axes.map(axis => `
        <label class="interest-axis">
          <span>${axis}</span>
          <input type="range" min="0" max="100" value="${Math.round((weights[axis] || 0) * 100)}" data-interest-axis="${axis}">
          <strong data-interest-percent="${axis}">0%</strong>
        </label>
      `).join('')}
    </div>
    <div class="interest-note">Saved optimized route stays artifact-backed. Dragging bars builds a browser preview only.</div>
    <div class="interest-note">National park candidates stay in candidate/context layers unless the selected route artifact includes them.</div>
    <div class="playback-controls">
      <button type="button" data-build-live-preview="true">Build live preview route</button>
      <button type="button" data-clear-live-preview="true">Clear preview route</button>
    </div>
    <div class="interest-note"><b>Preview only - rerun pipeline to save.</b> Browser preview is approximate; Run All recomputes the real optimized route.</div>
    <div id="interest-route-match" class="summary-list"></div>
    <div id="interest-ranking" class="interest-ranking"></div>
  `;
  let previewDebounce = null;
  let latestPreviewCandidates = [];
  const renderRanking = () => {
    const raw = Object.fromEntries(axes.map(axis => [
      axis,
      Number(target.querySelector(`[data-interest-axis="${axis}"]`)?.value || 0)
    ]));
    const normalized = normalizeInterestBarValues(raw);
    axes.forEach(axis => {
      const node = target.querySelector(`[data-interest-percent="${axis}"]`);
      if (node) node.textContent = `${Math.round(normalized[axis] * 100)}%`;
    });
    const routeMix = preview.route_mix || {};
    const profileMatch = nearestInterestProfile(preview, normalized);
    const savedProfile = savedInterestProfileRoute(profileMatch.profile);
    const routeError = axes.reduce((sum, axis) => {
      const mixValue = Number(routeMix[axis] ?? routeMix[`${axis}_score`] ?? 0);
      return sum + Math.abs((normalized[axis] || 0) - mixValue);
    }, 0);
    const routeScore = Math.max(0, 1 - routeError);
    const matchNode = target.querySelector('#interest-route-match');
    if (matchNode) {
      matchNode.innerHTML = `
        <div><b>Saved route match:</b> ${compactValue(routeScore, 2)}</div>
        <div><b>Closest saved profile:</b> ${profileMatch.profile || 'none'}${profileMatch.matched && savedProfile ? ' (available)' : ''}</div>
        <div><b>Route state:</b> ${profileMatch.matched && savedProfile ? 'Saved optimized profile route available' : 'Preview only - rerun pipeline to save'}</div>
      `;
    }
    const candidates = (preview.preview_candidates || preview.top_boosted_pois || [])
      .map(item => ({ ...item, preview_score: previewScore(item, normalized, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score)
    .filter(item => candidateAllowedByWeights(item, normalized))
    .slice(0, 8);
    latestPreviewCandidates = candidates;
    target.querySelector('#interest-ranking').innerHTML = candidates.map(item => `
      <div class="nature-card">
        <strong>${item.name}</strong>
        <div>${item.city || ''} · ${item.nature_region || item.park_type || ''}</div>
        <div>Preview score: ${compactValue(item.preview_score)} · Fit: ${compactValue(previewAxisFit(item, normalized))}</div>
      </div>
    `).join('');
  };
  const currentWeights = () => normalizeInterestBarValues(Object.fromEntries(axes.map(axis => [
    axis,
    Number(target.querySelector(`[data-interest-axis="${axis}"]`)?.value || 0)
  ])));
  const previewId = item => String(item.id || item.poi_id || item.place_id || item.name || '')
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '_');
  const itemCityText = item => `${item.name || ''} ${item.city || ''} ${item.city_or_anchor || ''} ${item.nature_region || ''} ${item.category || ''}`.toLowerCase();
  const endpointPoint = (endpoint, side) => endpoint && Number.isFinite(Number(endpoint.lat)) && Number.isFinite(Number(endpoint.lon))
    ? {
        name: endpoint.name || endpoint.code || 'Airport endpoint',
        city: endpoint.city || '',
        city_or_anchor: endpoint.name || endpoint.code || '',
        lat: Number(endpoint.lat),
        lon: Number(endpoint.lon),
        day: side === 'start' ? 1 : Number(window.dashboardMetrics?.trip_days || 0) || 1,
        stop_order: side === 'start' ? 0 : 999,
        type: 'airport_endpoint',
        category: 'airport_endpoint',
        is_route_endpoint: true,
        is_airport_endpoint: true,
        is_selected_poi: false,
        is_hotel_node: false,
        node_kind: 'airport',
        display_sequence_index: endpoint.code || 'AP',
        selected_stop_index: 0,
        route_sequence_index: 0,
        route_endpoint_side: side,
        airport_code: endpoint.code || ''
      }
    : null;
  const rankedPreviewCandidates = weights => {
    const previewItems = Array.isArray(preview.preview_candidates)
      ? preview.preview_candidates
      : (Array.isArray(preview.top_boosted_pois) ? preview.top_boosted_pois : []);
    const natureItems = Array.isArray(window.dashboardNatureExplore?.items) ? window.dashboardNatureExplore.items : [];
    const merged = [...previewItems, ...natureItems];
    const seen = new Set();
    return merged
      .filter(item => Number.isFinite(Number(item.lat)) && Number.isFinite(Number(item.lon)))
      .filter(item => {
        const key = `${item.name || ''}:${item.lat}:${item.lon}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .filter(item => candidateAllowedByWeights(item, weights))
      .map(item => ({ ...item, preview_score: previewScore(item, weights, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score);
  };
  const candidatesForSegment = (ranked, segment, seenKeys) => {
    const candidateIds = new Set((segment.candidate_ids || []).map(value => String(value).toLowerCase()));
    const allowed = (segment.allowed_cities || [])
      .map(value => String(value).toLowerCase())
      .filter(Boolean);
    return ranked.filter(item => {
      const key = `${item.name || ''}:${item.lat}:${item.lon}`.toLowerCase();
      if (seenKeys.has(key)) return false;
      const text = itemCityText(item);
      const id = previewId(item);
      if (candidateIds.has(id)) return true;
      return allowed.some(city => text.includes(city));
    });
  };
  const buildSegmentOrderedPreview = weights => {
    const ranked = rankedPreviewCandidates(weights);
    const graphSelection = previewGraphForWeights(preview, weights);
    const previewGraph = graphSelection.graph || {};
    const segments = Array.isArray(previewGraph.segments) ? previewGraph.segments : [];
    const route = [];
    const missing = [];
    const seen = new Set();
    const start = endpointPoint(previewGraph.start_endpoint, 'start');
    const end = endpointPoint(previewGraph.end_endpoint, 'end');
    if (start) route.push(start);
    if (!segments.length) {
      ranked.slice(0, Math.max(2, Math.min(10, Number(window.dashboardMetrics?.selected_stop_count || 8)))).forEach(item => route.push(item));
      if (end) route.push(end);
      return { route, sequenceFeasible: false, missingSegments: ['no segment graph exported'] };
    }
    segments.forEach(segment => {
      const limit = Math.max(1, Number(segment.stop_limit || 1));
      const picked = candidatesForSegment(ranked, segment, seen).slice(0, limit);
      if (!picked.length) {
        missing.push(`${segment.start_city || '?'} to ${segment.end_city || '?'}`);
      }
      picked.forEach(item => {
        const key = `${item.name || ''}:${item.lat}:${item.lon}`.toLowerCase();
        seen.add(key);
        route.push({
          ...item,
          segment_index: segment.segment_index,
          allowed_cities: (segment.allowed_cities || []).join(', ')
        });
      });
    });
    if (end) route.push(end);
    return { route, sequenceFeasible: missing.length === 0, missingSegments: missing };
  };
  const buildLivePreviewRoute = () => {
    const weights = currentWeights();
    const profileMatch = nearestInterestProfile(preview, weights);
    const savedProfileRoute = profileMatch.matched ? savedInterestProfileRoute(profileMatch.profile) : null;
    if (savedProfileRoute && window.showOnlyRoute) {
      window.showOnlyRoute(savedProfileRoute);
      if (window.clearLivePreviewRoute) window.clearLivePreviewRoute(false);
      const matchNode = target.querySelector('#interest-route-match');
      if (matchNode) {
        matchNode.innerHTML = `
          <div><b>Saved profile route:</b> ${savedProfileRoute.label || profileMatch.profile}</div>
          <div><b>Profile distance:</b> ${compactValue(profileMatch.distance, 2)} (threshold ${compactValue(profileMatch.threshold, 2)})</div>
          <div><b>Route state:</b> Saved optimized route artifact</div>
        `;
      }
      const status = document.getElementById('dashboard-status');
      if (status) {
        status.textContent = `Loaded saved ${profileMatch.profile} route artifact.`;
        status.style.color = '#0a6b53';
      }
      return;
    }
    const previewRoute = buildSegmentOrderedPreview(weights);
    const candidates = previewRoute.route;
    if (!candidates.filter(item => !item.is_route_endpoint).length) {
      const status = document.getElementById('dashboard-status');
      if (status) {
        status.textContent = 'No exported POI/nature candidates with coordinates are available for the live preview route.';
        status.style.color = '#b91c1c';
      }
      return;
    }
    if (window.drawLivePreviewRoute) {
      window.drawLivePreviewRoute(candidates, { ...weights, __sequence_feasible: previewRoute.sequenceFeasible });
    }
    const matchNode = target.querySelector('#interest-route-match');
    const score = candidates.reduce((sum, item) => sum + Number(item.preview_score || 0), 0);
    const natureStops = candidates.filter(item => {
      const text = `${item.name || ''} ${item.category || ''} ${item.nature_region || ''} ${item.park_type || ''}`.toLowerCase();
      return Number(item.nature_score || 0) >= 0.35
        || Number(item.scenic_score || 0) >= 0.35
        || Number(item.park_bonus || 0) > 0
        || item.is_national_park
        || /park|nature|scenic|hiking|viewpoint|yosemite|sequoia|kings canyon|joshua tree|big sur/.test(text);
    }).length;
    if (matchNode) {
      matchNode.innerHTML = `
        <div><b>Preview route score:</b> ${compactValue(score)}</div>
        <div><b>Preview stops:</b> ${candidates.filter(item => !item.is_route_endpoint).length}</div>
        <div><b>Preview nature stops:</b> ${natureStops}</div>
        <div><b>Sequence:</b> ${previewRoute.sequenceFeasible ? 'feasible' : `needs rerun / missing ${previewRoute.missingSegments.join('; ')}`}</div>
        <div><b>Route state:</b> Preview only - rerun pipeline to save</div>
      `;
    }
  };
  target.querySelector('[data-build-live-preview]')?.addEventListener('click', buildLivePreviewRoute);
  target.querySelector('[data-clear-live-preview]')?.addEventListener('click', () => {
    if (window.clearLivePreviewRoute) window.clearLivePreviewRoute();
  });
  target.querySelectorAll('[data-interest-axis]').forEach(input => input.addEventListener('input', () => {
    renderRanking();
    if (previewDebounce) window.clearTimeout(previewDebounce);
    previewDebounce = window.setTimeout(buildLivePreviewRoute, 260);
  }));
  renderRanking();
}

function routeButtonLabel(route) {
  if (route.customer_control_group === 'trip_days') return `${route.trip_days || '?'} days`;
  if (route.customer_control_group === 'interest') {
    return String(route.interest_profile || route.profile || route.label || 'Interest').replaceAll('_', ' ');
  }
  return route.label || 'Saved route';
}

function customerVisibleRoutes(index, controlGroup) {
  const routes = Array.isArray(index?.routes) ? index.routes : [];
  return routes
    .filter(route => route.customer_visible !== false)
    .filter(route => !controlGroup || route.customer_control_group === controlGroup)
    .filter(route => route.playable !== false && route.marker_only !== true);
}

function activateCustomerRoute(route, statusText) {
  if (!route || !window.showOnlyRoute) return;
  if (window.clearLivePreviewRoute) window.clearLivePreviewRoute(false);
  window.showOnlyRoute(route);
  const status = document.getElementById('dashboard-status');
  if (status) {
    status.textContent = statusText || `Loaded saved route artifact: ${route.label || route.id}`;
    status.style.color = '#0a6b53';
  }
  document.querySelectorAll('[data-customer-route-id]').forEach(button => {
    button.classList.toggle('active', button.dataset.customerRouteId === route.id);
  });
}

function renderCustomerControls(index) {
  const target = document.getElementById('customer-trip-controls');
  if (!target) return;
  const defaultRoute = customerVisibleRoutes(index).find(route => route.default);
  const dayRoutes = customerVisibleRoutes(index, 'trip_days')
    .sort((a, b) => Number(a.trip_days || 0) - Number(b.trip_days || 0));
  const interestRoutes = customerVisibleRoutes(index, 'interest')
    .sort((a, b) => String(a.interest_profile || a.profile || a.label).localeCompare(String(b.interest_profile || b.profile || b.label)));
  const dayButtons = dayRoutes.length
    ? dayRoutes.map(route => `<button type="button" data-customer-route-id="${escapeHtml(route.id)}" data-customer-route-kind="days">${escapeHtml(routeButtonLabel(route))}</button>`).join('')
    : '<span class="interest-note">No alternate day-count artifacts were exported. Requires pipeline rerun.</span>';
  const interestButtons = interestRoutes.length
    ? interestRoutes.map(route => `<button type="button" data-customer-route-id="${escapeHtml(route.id)}" data-customer-route-kind="interest">${escapeHtml(routeButtonLabel(route))}</button>`).join('')
    : '<span class="interest-note">No saved interest-profile artifacts were exported. Slider changes are preview-only.</span>';
  target.innerHTML = `
    <div class="customer-control-grid">
      <div class="customer-control-block">
        <strong>Trip length</strong>
        <div class="customer-option-grid">${dayButtons}</div>
        <div class="interest-note">Only exported trip-day artifacts can be selected here; other day counts require a Python pipeline rerun.</div>
      </div>
      <div class="customer-control-block">
        <strong>Interest preset</strong>
        <div class="customer-option-grid">${interestButtons}</div>
        <div class="interest-note">Saved profile routes are optimizer artifacts. Custom slider mixes remain preview-only.</div>
      </div>
      <div class="customer-control-block">
        <strong>Hotels</strong>
        <div class="interest-note">Hotel preferences are saved locally in this browser and can be exported as a soft bonus for the next optimizer rerun.</div>
      </div>
    </div>
  `;
  const byId = new Map(customerVisibleRoutes(index).map(route => [route.id, route]));
  target.querySelectorAll('[data-customer-route-id]').forEach(button => {
    button.addEventListener('click', () => {
      const route = byId.get(button.dataset.customerRouteId);
      const kind = button.dataset.customerRouteKind === 'days' ? 'trip length' : 'interest profile';
      activateCustomerRoute(route, `Loaded saved ${kind} route artifact: ${route?.label || route?.id || ''}`);
    });
  });
  if (defaultRoute) {
    target.querySelector(`[data-customer-route-id="${CSS.escape(defaultRoute.id)}"]`)?.classList.add('active');
  }
}

window.renderDashboardMetrics = renderDashboardMetrics;
window.bindDashboardShellControls = bindDashboardShellControls;
window.renderActiveStopDetail = renderActiveStopDetail;
window.renderCityDetails = renderCityDetails;
window.renderHotelChoices = renderHotelChoices;
window.renderNatureExplore = renderNatureExplore;
window.renderDebugSummary = renderDebugSummary;
window.renderInterestPreview = renderInterestPreview;
window.renderCustomerControls = renderCustomerControls;
