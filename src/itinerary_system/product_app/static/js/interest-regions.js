const LEVELS = [0.5, 0.8, 0.95];
export const REQUIRED_INTEREST_COLOR = "#082f49";
const validNumber = value => typeof value === "number" && Number.isFinite(value);
const position = value => Array.isArray(value) && value.length === 2
  && value.every(validNumber) && Math.abs(value[0]) <= 180 && Math.abs(value[1]) <= 90;

export function interestPlacesInRegion(payload, level) {
  const region = payload.surface.regions_geojson.features.find(feature => feature.properties.mass_level === level);
  if (!region) return payload.poi_predictions;
  const inRing = ([x, y], ring) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [ax, ay] = ring[j], [bx, by] = ring[i];
      if (Math.abs((x - ax) * (by - ay) - (y - ay) * (bx - ax)) < 1e-12
          && x >= Math.min(ax, bx) && x <= Math.max(ax, bx)
          && y >= Math.min(ay, by) && y <= Math.max(ay, by)) return true;
      if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) inside = !inside;
    }
    return inside;
  };
  return payload.poi_predictions.filter(point => {
    const coordinate = [point.display_coordinate.longitude, point.display_coordinate.latitude];
    return region.geometry.coordinates.some(([outer, ...holes]) =>
      inRing(coordinate, outer) && !holes.some(hole => inRing(coordinate, hole)));
  });
}

function validateCollection(value, maximum) {
  if (value?.type !== "FeatureCollection" || !Array.isArray(value.features) || value.features.length > maximum) {
    throw new Error("interest_geometry_invalid");
  }
  let count = 0;
  for (const feature of value.features) {
    if (feature.type !== "Feature" || feature.geometry?.type !== "MultiPolygon"
        || !Array.isArray(feature.geometry.coordinates)) throw new Error("interest_geometry_invalid");
    for (const polygon of feature.geometry.coordinates) {
      if (!Array.isArray(polygon) || !polygon.length) throw new Error("interest_geometry_invalid");
      for (const ring of polygon) {
        if (!Array.isArray(ring) || ring.length < 4 || !ring.every(position)
            || ring[0][0] !== ring.at(-1)[0] || ring[0][1] !== ring.at(-1)[1]) throw new Error("interest_geometry_invalid");
        count += ring.length;
        if (count > 100000) throw new Error("interest_geometry_too_large");
      }
    }
  }
}

export function normalizeInterestResponse(payload, expected) {
  if (payload?.schema_version !== "product-interest-regions-v1"
      || payload.session_id !== expected.sessionId || payload.session_revision !== expected.revision
      || payload.accepted_plan_id !== expected.planId || payload.accepted_plan_content_hash !== expected.planHash
      || payload.exploration_plan_id !== expected.explorationPlanId
      || payload.exploration_plan_content_hash !== expected.explorationPlanHash
      || payload.plan_scope !== (expected.explorationPlanId === expected.planId ? "accepted" : "evaluated_preview")
      || payload.context?.day !== expected.day || payload.context?.route_leg_id !== expected.routeLegId
      || payload.nature_preference !== expected.naturePreference
      || payload.context?.replacement_target_id !== expected.replacementTargetId) throw new Error("interest_context_stale");
  const search = payload.search_context;
  if (!search || typeof search.stop_id !== "string" || !search.stop_id || typeof search.name !== "string" || !search.name
      || !position([search.coordinate?.longitude, search.coordinate?.latitude])
      || !validNumber(search.radius_km) || search.radius_km < 1 || search.radius_km > 25
      || search.distance_kind !== "straight_line"
      || expected.focusStopId && search.stop_id !== expected.focusStopId
      || expected.searchArea && search.area_mode !== expected.searchArea
      || expected.radiusKm != null && search.radius_km !== expected.radiusKm) throw new Error("interest_context_stale");
  if (payload.candidate_universe?.truncated !== false || !/^[a-f0-9]{64}$/.test(payload.candidate_universe.sha256 || "")
      || !Array.isArray(payload.poi_predictions) || payload.poi_predictions.length > 500
      || payload.candidate_universe.count !== payload.poi_predictions.length
      || payload.model?.calibration_status !== "not_calibrated"
      || !["prior_only", "uncalibrated_posterior"].includes(payload.model.evidence_mode)) throw new Error("interest_prediction_invalid");
  const ids = new Set();
  for (const point of payload.poi_predictions) {
    const interval = point.interest_probability_interval;
    if (typeof point.place_id !== "string" || !point.place_id || ids.has(point.place_id)
        || typeof point.name !== "string" || !point.name
        || !position([point.display_coordinate?.longitude, point.display_coordinate?.latitude])
        || !validNumber(point.interest_probability_mean) || point.interest_probability_mean < 0 || point.interest_probability_mean > 1
        || !Array.isArray(interval) || interval.length !== 2 || !interval.every(validNumber)
        || interval[0] < 0 || interval[1] > 1 || interval[0] > interval[1]
        || point.interval_level !== 0.9) throw new Error("interest_prediction_invalid");
    if (!validNumber(point.distance_from_focus_km) || point.distance_from_focus_km < 0
        || point.distance_from_focus_km > search.radius_km) throw new Error("interest_distance_invalid");
    const radians = Math.PI / 180;
    const latitude = point.display_coordinate.latitude * radians, originLat = search.coordinate.latitude * radians;
    const haversine = Math.sin((latitude - originLat) / 2) ** 2 + Math.cos(latitude) * Math.cos(originLat)
      * Math.sin((point.display_coordinate.longitude - search.coordinate.longitude) * radians / 2) ** 2;
    const distance = 2 * 6371.0088 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, haversine))));
    if (distance > search.radius_km + 1e-6 || Math.abs(distance - point.distance_from_focus_km) > 0.01) {
      throw new Error("interest_distance_invalid");
    }
    ids.add(point.place_id);
  }
  if (!Array.isArray(payload.required_points) || payload.required_points.length > 500) throw new Error("interest_required_invalid");
  const requiredIds = new Set();
  for (const point of payload.required_points) {
    if (typeof point.place_id !== "string" || !point.place_id || requiredIds.has(point.place_id)
        || point.coordinate !== null && !position(point.coordinate)
        || !Array.isArray(point.constraint_ids) || !point.constraint_ids.length) throw new Error("interest_required_invalid");
    requiredIds.add(point.place_id);
  }
  if (!["ready", "unavailable"].includes(payload.surface?.status)) throw new Error("interest_surface_invalid");
  validateCollection(payload.surface.density_geojson, 12);
  validateCollection(payload.surface.regions_geojson, 3);
  if (payload.surface.status === "ready") {
    if (!validNumber(payload.surface.integrated_mass) || Math.abs(payload.surface.integrated_mass - 1) > 1e-3
        || payload.surface.regions_geojson.features.length !== 3) throw new Error("interest_surface_invalid");
    const levels = new Set();
    for (const feature of payload.surface.regions_geojson.features) {
      const properties = feature.properties;
      if (!LEVELS.includes(properties?.mass_level) || levels.has(properties.mass_level)
          || !validNumber(properties.achieved_mass) || properties.achieved_mass + 1e-12 < properties.mass_level
          || properties.achieved_mass > 1 + 1e-12) throw new Error("interest_mass_invalid");
      levels.add(properties.mass_level);
    }
    for (const feature of payload.surface.density_geojson.features) {
      if (!validNumber(feature.properties?.density_fraction) || feature.properties.density_fraction < 0
          || feature.properties.density_fraction > 1) throw new Error("interest_density_invalid");
    }
  } else if (payload.surface.density_geojson.features.length || payload.surface.regions_geojson.features.length) {
    throw new Error("interest_unavailable_geometry_invalid");
  }
  return payload;
}

export function addInterestAreaLayers(map, payload, level) {
  if (!payload || payload.surface.status !== "ready") return;
  map.addSource("interest-density", {type: "geojson", data: payload.surface.density_geojson});
  map.addLayer({id: "interest-density", type: "fill", source: "interest-density", paint: {
    "fill-color": ["interpolate", ["linear"], ["get", "density_fraction"], 0, "#e0f2fe", 0.35, "#7dd3fc", 0.7, "#0284c7", 1, "#075985"],
    "fill-opacity": 0.48, "fill-antialias": false,
  }});
  map.addSource("interest-regions", {type: "geojson", data: payload.surface.regions_geojson});
  map.addLayer({id: "interest-region-boundary", type: "line", source: "interest-regions",
    filter: ["==", ["get", "mass_level"], LEVELS.includes(level) ? level : 0.8],
    paint: {"line-color": "#0369a1", "line-width": 1.8, "line-dasharray": [3, 2]},
  });
}

export function addInterestPointLayers(map, payload, selectedId, onSelect) {
  if (!payload) return;
  const points = {type: "FeatureCollection", features: payload.poi_predictions.map(point => ({
    type: "Feature", geometry: {type: "Point", coordinates: [point.display_coordinate.longitude, point.display_coordinate.latitude]},
    properties: {place_id: point.place_id, name: point.name, probability: point.interest_probability_mean},
  }))};
  map.addSource("interest-pois", {type: "geojson", data: points});
  map.addLayer({id: "interest-pois", type: "circle", source: "interest-pois", paint: {
    "circle-radius": ["case", ["==", ["get", "place_id"], selectedId || ""], 10, 6],
    "circle-color": ["interpolate", ["linear"], ["get", "probability"], 0, "#bae6fd", 1, "#075985"],
    "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5,
  }});
  const font = map.getStyle?.().layers?.find(layer => layer.type === "symbol" && layer.layout?.["text-font"])?.layout["text-font"];
  if (font) map.addLayer({id: "interest-poi-labels", type: "symbol", source: "interest-pois",
    filter: ["!=", ["get", "place_id"], selectedId || ""],
    layout: {"text-field": ["get", "name"], "text-font": font, "text-size": 12,
      "text-anchor": "top", "text-offset": [0, 0.8], "text-max-width": 16},
    paint: {"text-color": "#075985", "text-halo-color": "#ffffff", "text-halo-width": 2},
  });
  if (font && selectedId) map.addLayer({id: "interest-selected-label", type: "symbol", source: "interest-pois",
    filter: ["==", ["get", "place_id"], selectedId],
    layout: {"text-field": ["get", "name"], "text-font": font, "text-size": 14,
      "text-anchor": "top", "text-offset": [0, 1], "text-max-width": 16, "text-allow-overlap": true},
    paint: {"text-color": "#075985", "text-halo-color": "#ffffff", "text-halo-width": 2},
  });
  const required = {type: "FeatureCollection", features: payload.required_points.filter(point => point.coordinate).map(point => ({
    type: "Feature", geometry: {type: "Point", coordinates: point.coordinate},
    properties: {place_id: point.place_id, conflict_status: point.conflict_status},
  }))};
  map.addSource("interest-required", {type: "geojson", data: required});
  map.addLayer({id: "interest-required", type: "circle", source: "interest-required", paint: {
    "circle-radius": 11, "circle-color": REQUIRED_INTEREST_COLOR,
    "circle-stroke-color": "#ffffff", "circle-stroke-width": 3,
  }});
  if (font) map.addLayer({id: "interest-required-label", type: "symbol", source: "interest-required",
    layout: {"text-field": ["case", ["==", ["get", "conflict_status"], "satisfied"], "Must-go", "Must-go: conflict"],
      "text-font": font, "text-size": 12, "text-offset": [0, 1.5], "text-allow-overlap": true},
    paint: {"text-color": REQUIRED_INTEREST_COLOR, "text-halo-color": "#ffffff", "text-halo-width": 2},
  });
  ["interest-pois", "interest-required", ...(font ? ["interest-poi-labels"] : []),
    ...(font && selectedId ? ["interest-selected-label"] : [])].forEach(id => map.on("click", id, event => {
    if (id !== "interest-required" && map.queryRenderedFeatures?.(event.point, {layers: ["interest-required"]})?.length) return;
    const placeId = event.features?.[0]?.properties?.place_id;
    // Selection rebuilds the map; finish MapLibre's delegated event dispatch first.
    if (placeId) queueMicrotask(() => onSelect(placeId));
  }));
}
