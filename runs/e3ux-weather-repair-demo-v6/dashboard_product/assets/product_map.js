(() => {
  'use strict';

  const data = window.PRODUCT_DASHBOARD_DATA;
  const mapElement = document.getElementById('product-map');
  const status = document.getElementById('map-status');
  if (!data || !mapElement || !status) return;
  if (!window.L) {
    status.textContent = 'Interactive map unavailable. Use the text alternative below.';
    return;
  }

  const map = L.map(mapElement, { zoomControl: true, scrollWheelZoom: false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  const parentLayer = L.layerGroup().addTo(map);
  const childLayer = L.layerGroup().addTo(map);
  const markerLayer = L.layerGroup().addTo(map);
  const routeLines = [];
  const affectedDays = new Set(data.map.affected_days || []);
  const routeStopDays = new Map();
  [data.map.parent, data.map.child].filter(Boolean).forEach(route => {
    route.stops.forEach(stop => routeStopDays.set(stop.id, Number(stop.day)));
  });

  function popupNode(stop, routeLabel) {
    const wrapper = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = stop.name;
    const detail = document.createElement('div');
    detail.textContent = `${routeLabel} · Day ${stop.day} · ${stop.city || 'City unavailable'}`;
    wrapper.append(title, detail);
    return wrapper;
  }

  function routeStyle(kind, day, selectedDay) {
    const affected = affectedDays.has(day);
    const selected = selectedDay === undefined || day === Number(selectedDay);
    const style = kind === 'parent'
      ? { color: affected ? '#854306' : '#59645c', weight: affected ? 5 : 3, opacity: affected ? 0.9 : 0.5, dashArray: '8 7' }
      : { color: affected ? '#854306' : '#0b6f68', weight: affected ? 6 : 4, opacity: affected ? 1 : 0.82 };
    if (!selected) return { ...style, weight: 2, opacity: affected ? 0.28 : 0.16 };
    return { ...style, weight: style.weight + 2, opacity: 1 };
  }

  function drawRoute(route, kind, layer) {
    if (!route) return;
    route.segments.forEach(segment => {
      const day = routeStopDays.get(segment.destination_id) || 0;
      const line = L.polyline(segment.coordinates, routeStyle(kind, day));
      line.productDay = day;
      line.productKind = kind;
      line.bindTooltip(`${kind === 'parent' ? 'Original' : 'Repaired'} route · ${segment.origin_id} to ${segment.destination_id}`);
      line.addTo(layer);
      routeLines.push(line);
    });
    route.stops.forEach(stop => {
      if (!Number.isFinite(Number(stop.latitude)) || !Number.isFinite(Number(stop.longitude))) return;
      const affected = affectedDays.has(Number(stop.day));
      const marker = L.circleMarker([stop.latitude, stop.longitude], {
        radius: affected ? 8 : 6,
        color: affected ? '#854306' : kind === 'parent' ? '#59645c' : '#0b6f68',
        fillColor: '#fffdf7',
        fillOpacity: 1,
        weight: affected ? 4 : 2,
        opacity: kind === 'parent' ? 0.65 : 1,
      });
      marker.productDay = Number(stop.day);
      marker.productKind = kind;
      marker.productStrength = stop.ownership_strength || '';
      marker.bindPopup(popupNode(stop, kind === 'parent' ? 'Original plan' : 'Repaired plan'));
      marker.addTo(markerLayer);
    });
  }

  drawRoute(data.map.parent, 'parent', parentLayer);
  drawRoute(data.map.child, 'child', childLayer);
  L.control.layers(
    {},
    { 'Original route': parentLayer, 'Repaired route': childLayer, 'Stops': markerLayer },
    { collapsed: true, position: 'topright' }
  ).addTo(map);

  const allCoordinates = [];
  [data.map.parent, data.map.child].filter(Boolean).forEach(route => {
    route.stops.forEach(stop => {
      if (Number.isFinite(Number(stop.latitude)) && Number.isFinite(Number(stop.longitude))) {
        allCoordinates.push([stop.latitude, stop.longitude]);
      }
    });
  });
  if (allCoordinates.length) map.fitBounds(allCoordinates, { padding: [24, 24] });
  else map.setView([36.4, -119.7], 5);

  function selectDay(day) {
    const selectedCoordinates = [];
    routeLines.forEach(line => {
      line.setStyle(routeStyle(line.productKind, line.productDay, day));
    });
    markerLayer.eachLayer(marker => {
      const selected = marker.productDay === Number(day);
      const protectedStop = marker.productStrength === 'booked' || marker.productStrength === 'locked';
      marker.setStyle({
        radius: selected ? (protectedStop ? 11 : 9) : (protectedStop ? 7 : 5),
        weight: protectedStop ? 5 : selected ? 3 : 2,
        opacity: selected ? 1 : 0.42,
        fillOpacity: selected ? 1 : 0.55,
      });
      if (selected) selectedCoordinates.push(marker.getLatLng());
    });
    if (selectedCoordinates.length) {
      map.fitBounds(selectedCoordinates, { padding: [36, 36], maxZoom: 11 });
    }
    const evidence = affectedDays.has(Number(day))
      ? ` Affected-day evidence: ${data.map.evidence.label} (${data.map.evidence.source_status}).`
      : '';
    status.textContent = selectedCoordinates.length
      ? `Showing ${selectedCoordinates.length} route stop markers for day ${day}.${evidence}`
      : `No mappable stop coordinates are recorded for day ${day}.`;
  }

  window.addEventListener('product-day-selected', event => selectDay(event.detail.day));
  window.addEventListener('product-dashboard-ready', () => {
    map.invalidateSize();
    selectDay(data.trip.selected_day);
  });
  selectDay(data.trip.selected_day);
  window.productDashboardMap = map;
})();
