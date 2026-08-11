const EXPECTED_GEOGRAPHY_SCHEMA = "product-geography-v2";
const EXPECTED_MAP_SCHEMA = "product-map-configuration-v2";
const READY_GEOGRAPHY_STATES = new Set(["ready", "ready_with_gaps"]);
const READY_PLAN_STATES = new Set(["ready", "ready_with_gaps"]);
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost"]);

let pmtilesProtocolRegistered = false;

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isPosition(value) {
  return Array.isArray(value)
    && value.length >= 2
    && isFiniteNumber(value[0])
    && isFiniteNumber(value[1])
    && value[0] >= -180
    && value[0] <= 180
    && value[1] >= -90
    && value[1] <= 90;
}

function isPointFeatureCollection(value, { nullable = false } = {}) {
  return value?.type === "FeatureCollection"
    && Array.isArray(value.features)
    && value.features.every((feature) => {
      if (feature?.type !== "Feature" || !feature.properties
          || typeof feature.properties !== "object") return false;
      if (nullable && feature.geometry === null) return true;
      return feature.geometry?.type === "Point" && isPosition(feature.geometry.coordinates);
    });
}

function isLineFeatureCollection(value) {
  return value?.type === "FeatureCollection"
    && Array.isArray(value.features)
    && value.features.every((feature) => feature?.type === "Feature"
      && feature.geometry?.type === "LineString"
      && Array.isArray(feature.geometry.coordinates)
      && feature.geometry.coordinates.length >= 2
      && feature.geometry.coordinates.every(isPosition)
      && feature.properties
      && typeof feature.properties === "object");
}

function validateBounds(bounds) {
  return Array.isArray(bounds)
    && bounds.length === 4
    && bounds.every(isFiniteNumber)
    && bounds[0] >= -180
    && bounds[2] <= 180
    && bounds[1] >= -90
    && bounds[3] <= 90
    && bounds[0] < bounds[2]
    && bounds[1] < bounds[3];
}

function validateCoverage(coverage) {
  return coverage && typeof coverage === "object"
    && ["complete", "gaps_present"].includes(coverage.status)
    && Number.isInteger(coverage.route_path_node_count)
    && coverage.route_path_node_count >= 0
    && Number.isInteger(coverage.required_leg_count)
    && coverage.required_leg_count >= 0
    && Number.isInteger(coverage.road_validated_leg_count)
    && coverage.road_validated_leg_count >= 0
    && Number.isInteger(coverage.gap_count)
    && coverage.gap_count >= 0;
}

function validatePlan(plan) {
  if (!plan || typeof plan.plan_id !== "string" || !plan.plan_id
      || typeof plan.content_hash !== "string" || !plan.content_hash
      || !["original", "registered_repair", "alternative", "draft_preview"].includes(plan.role)
      || !READY_PLAN_STATES.has(plan.status)
      || !validateCoverage(plan.coverage)
      || !isPointFeatureCollection(plan.stops)
      || !isPointFeatureCollection(plan.route_path, { nullable: true })
      || !isLineFeatureCollection(plan.validated_legs)
      || !isPointFeatureCollection(plan.gaps, { nullable: true })) {
    throw new Error("geography_plan_invalid");
  }
  plan.stops.features.forEach((feature) => {
    if (feature.properties.plan_id !== plan.plan_id
        || typeof feature.properties.stop_id !== "string"
        || !feature.properties.stop_id) {
      throw new Error("geography_stop_lineage_invalid");
    }
  });
  plan.route_path.features.forEach((feature) => {
    if (feature.properties.plan_id !== plan.plan_id
        || typeof feature.properties.node_id !== "string"
        || !feature.properties.node_id
        || typeof feature.properties.route_anchor !== "boolean"
        || !Number.isInteger(feature.properties.occurrence_index)) {
      throw new Error("geography_route_path_invalid");
    }
  });
  plan.validated_legs.features.forEach((feature) => {
    if (feature.properties.plan_id !== plan.plan_id
        || typeof feature.properties.route_leg_id !== "string"
        || !feature.properties.route_leg_id
        || feature.properties.validation_status !== "road_validated"
        || feature.properties.road_validated !== true
        || feature.properties.fallback_used !== false
        || typeof feature.properties.origin_id !== "string"
        || typeof feature.properties.destination_id !== "string") {
      throw new Error("geography_route_lineage_invalid");
    }
  });
  plan.gaps.features.forEach((feature) => {
    const properties = feature.properties;
    if (feature.geometry !== null
        || properties.plan_id !== plan.plan_id
        || properties.validation_status !== "unvalidated_gap"
        || typeof properties.failure_code !== "string"
        || !properties.failure_code) {
      throw new Error("geography_route_gap_invalid");
    }
  });
  if (plan.coverage.route_path_node_count !== plan.route_path.features.length
      || plan.coverage.road_validated_leg_count !== plan.validated_legs.features.length
      || plan.coverage.gap_count !== plan.gaps.features.length
      || plan.coverage.required_leg_count
        !== plan.validated_legs.features.length + plan.gaps.features.length) {
    throw new Error("geography_coverage_mismatch");
  }
  return plan;
}

function selectVisiblePlans(geography, activePlanId) {
  if (geography?.schema_version !== EXPECTED_GEOGRAPHY_SCHEMA
      || !READY_GEOGRAPHY_STATES.has(geography.status)) {
    throw new Error("geography_unavailable");
  }
  if (!validateBounds(geography.bounds) || !Array.isArray(geography.plans)) {
    throw new Error("geography_invalid");
  }
  if (typeof activePlanId !== "string" || !activePlanId) {
    throw new Error("active_plan_required");
  }
  const originalCandidates = geography.plans.filter((plan) => plan?.role === "original");
  if (originalCandidates.length !== 1) throw new Error("geography_original_plan_invalid");
  const activeCandidate = geography.plans.find((plan) => plan?.plan_id === activePlanId);
  if (!activeCandidate) throw new Error("active_plan_not_found");

  // Validate only the plans this pane can display. A malformed unrelated
  // alternative must not take down the baseline or another option pane.
  const original = validatePlan(originalCandidates[0]);
  const active = activeCandidate === originalCandidates[0]
    ? original
    : validatePlan(activeCandidate);
  return {
    fallbackBounds: geography.bounds,
    plans: active.plan_id === original.plan_id ? [original] : [original, active],
    original,
    active,
  };
}

function localAssetUrl(value, baseUrl) {
  const base = new URL(baseUrl);
  const candidate = new URL(value, base);
  if (base.protocol !== "http:" || candidate.protocol !== "http:"
      || !LOOPBACK_HOSTS.has(base.hostname) || !LOOPBACK_HOSTS.has(candidate.hostname)
      || candidate.origin !== base.origin || candidate.username || candidate.password) {
    throw new Error("map_asset_url_not_loopback");
  }
  return candidate.href;
}

function loadStylesheet(url) {
  const id = "local-maplibre-v2-stylesheet";
  let existing = document.getElementById(id);
  if (existing?.dataset.loadState === "failed") {
    existing.remove();
    existing = null;
  }
  if (existing) {
    if (existing.href !== url) return Promise.reject(new Error("map_stylesheet_conflict"));
    if (existing.dataset.loadState === "loaded") return Promise.resolve();
    return new Promise((resolve, reject) => {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", () => {
        existing.dataset.loadState = "failed";
        existing.remove();
        reject(new Error("map_stylesheet_failed"));
      }, { once: true });
    });
  }
  return new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = url;
    link.dataset.loadState = "loading";
    link.addEventListener("load", () => {
      link.dataset.loadState = "loaded";
      resolve();
    }, { once: true });
    link.addEventListener("error", () => {
      link.dataset.loadState = "failed";
      link.remove();
      reject(new Error("map_stylesheet_failed"));
    }, { once: true });
    document.head.append(link);
  });
}

function loadScript(url, id, ready) {
  if (ready()) return Promise.resolve();
  let existing = document.getElementById(id);
  if (existing?.dataset.loadState === "failed") {
    existing.remove();
    existing = null;
  }
  if (existing) {
    if (existing.src !== url) return Promise.reject(new Error("map_script_conflict"));
    if (existing.dataset.loadState === "loaded") {
      existing.remove();
      return Promise.reject(new Error("map_runtime_invalid"));
    }
    return new Promise((resolve, reject) => {
      existing.addEventListener("load", () => {
        existing.dataset.loadState = "loaded";
        if (ready()) resolve();
        else {
          existing.remove();
          reject(new Error("map_runtime_invalid"));
        }
      }, { once: true });
      existing.addEventListener("error", () => {
        existing.dataset.loadState = "failed";
        existing.remove();
        reject(new Error("map_script_failed"));
      }, { once: true });
    });
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.id = id;
    script.src = url;
    script.dataset.loadState = "loading";
    script.addEventListener("load", () => {
      script.dataset.loadState = "loaded";
      if (ready()) resolve();
      else {
        script.remove();
        reject(new Error("map_runtime_invalid"));
      }
    }, { once: true });
    script.addEventListener("error", () => {
      script.dataset.loadState = "failed";
      script.remove();
      reject(new Error("map_script_failed"));
    }, { once: true });
    document.head.append(script);
  });
}

async function loadMapRuntime(config) {
  if (config?.schema_version !== EXPECTED_MAP_SCHEMA
      || config.status !== "ready" || config.provider !== "maplibre_pmtiles") {
    throw new Error("map_runtime_unavailable");
  }
  const base = localAssetUrl(config.base_url, config.base_url);
  const stylesheet = localAssetUrl(config.stylesheet_url, base);
  const protocolScript = localAssetUrl(config.protocol_script_url, base);
  const mapScript = localAssetUrl(config.script_url, base);
  const style = localAssetUrl(config.style_url, base);
  await Promise.all([
    loadStylesheet(stylesheet),
    loadScript(protocolScript, "local-pmtiles-v2-runtime", () => Boolean(window.pmtiles?.Protocol)),
  ]);
  await loadScript(mapScript, "local-maplibre-v2-runtime", () => Boolean(window.maplibregl?.Map));
  if (!window.pmtiles?.Protocol || !window.maplibregl?.Map) throw new Error("map_runtime_invalid");
  if (!pmtilesProtocolRegistered) {
    const protocol = new window.pmtiles.Protocol();
    window.maplibregl.addProtocol("pmtiles", protocol.tile);
    pmtilesProtocolRegistered = true;
  }
  return style;
}

function withDisplayRole(feature, displayRole) {
  return {
    ...feature,
    properties: { ...feature.properties, display_role: displayRole },
  };
}

function collectFeatureCollection(plans, field, originalPlanId, activePlanId) {
  return {
    type: "FeatureCollection",
    features: plans.flatMap((plan) => plan[field].features.map((feature) => withDisplayRole(
      feature,
      plan.plan_id === activePlanId && plan.plan_id !== originalPlanId ? "active" : "original",
    ))),
  };
}

function deriveRouteAnchors(plans, originalPlanId, activePlanId) {
  return {
    type: "FeatureCollection",
    features: plans.flatMap((plan) => plan.route_path.features
      .filter((feature) => feature.properties.route_anchor === true && feature.geometry !== null)
      .map((feature) => withDisplayRole(
        feature,
        plan.plan_id === activePlanId && plan.plan_id !== originalPlanId ? "active" : "original",
      ))),
  };
}

function midpoint(left, right) {
  return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2];
}

function resolveGapPoint(plan, gap) {
  const properties = gap.properties;
  if (typeof properties.origin_id !== "string" || typeof properties.destination_id !== "string") {
    return null;
  }
  const path = plan.route_path.features;
  if (Number.isInteger(properties.leg_index) && properties.leg_index > 0) {
    const origin = path[properties.leg_index - 1];
    const destination = path[properties.leg_index];
    if (origin?.properties.node_id === properties.origin_id
        && destination?.properties.node_id === properties.destination_id) {
      const originPosition = origin.geometry?.coordinates;
      const destinationPosition = destination.geometry?.coordinates;
      if (isPosition(originPosition) && isPosition(destinationPosition)) {
        return midpoint(originPosition, destinationPosition);
      }
      if (isPosition(originPosition)) return originPosition;
      if (isPosition(destinationPosition)) return destinationPosition;
    }
  }
  for (let index = 0; index < path.length - 1; index += 1) {
    const origin = path[index];
    const destination = path[index + 1];
    if (origin.properties.node_id !== properties.origin_id
        || destination.properties.node_id !== properties.destination_id) continue;
    const originPosition = origin.geometry?.coordinates;
    const destinationPosition = destination.geometry?.coordinates;
    if (isPosition(originPosition) && isPosition(destinationPosition)) {
      return midpoint(originPosition, destinationPosition);
    }
    if (isPosition(originPosition)) return originPosition;
    if (isPosition(destinationPosition)) return destinationPosition;
  }
  return null;
}

function gapDescription(plan, gap, coordinate) {
  const properties = gap.properties;
  const origin = properties.origin_id || "unknown origin";
  const destination = properties.destination_id || "unknown destination";
  return Object.freeze({
    plan_id: plan.plan_id,
    requirement_id: properties.requirement_id || null,
    route_leg_id: properties.route_leg_id || null,
    origin_id: properties.origin_id || null,
    destination_id: properties.destination_id || null,
    failure_code: properties.failure_code,
    marker_available: coordinate !== null,
    text: `Unvalidated route gap: ${origin} to ${destination} (${properties.failure_code}).`,
  });
}

function resolveGapMarkers(plans, originalPlanId, activePlanId) {
  const features = [];
  const descriptions = [];
  plans.forEach((plan) => {
    plan.gaps.features.forEach((gap) => {
      const coordinate = resolveGapPoint(plan, gap);
      const description = gapDescription(plan, gap, coordinate);
      descriptions.push(description);
      if (coordinate !== null) {
        features.push({
          ...withDisplayRole(gap, plan.plan_id === activePlanId
            && plan.plan_id !== originalPlanId ? "active" : "original"),
          geometry: { type: "Point", coordinates: coordinate },
          properties: { ...gap.properties, display_text: description.text },
        });
      }
    });
  });
  return {
    markers: { type: "FeatureCollection", features },
    descriptions: Object.freeze(descriptions),
  };
}

function roleFilter(role) {
  return ["==", ["get", "display_role"], role];
}

function hideVisualCanvas(map) {
  const canvas = map.getCanvas();
  canvas.setAttribute("aria-hidden", "true");
  canvas.setAttribute("tabindex", "-1");
}

function renderedCanvasCoverage(container, map) {
  const containerRect = container.getBoundingClientRect();
  const canvasRect = map.getCanvas().getBoundingClientRect();
  if (containerRect.width <= 0 || containerRect.height <= 0) return 0;
  const width = Math.max(0, Math.min(containerRect.right, canvasRect.right)
    - Math.max(containerRect.left, canvasRect.left));
  const height = Math.max(0, Math.min(containerRect.bottom, canvasRect.bottom)
    - Math.max(containerRect.top, canvasRect.top));
  return (width * height) / (containerRect.width * containerRect.height);
}

function validateDraftWaypoint(waypoint) {
  if (!waypoint || typeof waypoint.waypoint_id !== "string" || !waypoint.waypoint_id
      || typeof waypoint.name !== "string" || !waypoint.name
      || !isPosition(waypoint.coordinate)
      || !["custom_stop", "route_waypoint"].includes(waypoint.kind)) {
    throw new Error("draft_waypoint_invalid");
  }
  return waypoint;
}

function draftRouteFeatureCollection(routeLegs) {
  if (!Array.isArray(routeLegs)) throw new Error("draft_route_legs_invalid");
  return {
    type: "FeatureCollection",
    features: routeLegs.map((leg) => {
      if (!leg || typeof leg.route_leg_id !== "string" || !leg.route_leg_id
          || leg.road_validated !== true
          || leg.validation_status !== "road_validated"
          || leg.geometry?.type !== "LineString"
          || !Array.isArray(leg.geometry.coordinates)
          || leg.geometry.coordinates.length < 2
          || !leg.geometry.coordinates.every(isPosition)) {
        throw new Error("draft_route_leg_invalid");
      }
      return {
        type: "Feature",
        geometry: leg.geometry,
        properties: {
          route_leg_id: leg.route_leg_id,
          waypoint_id: leg.waypoint_id || null,
          validation_status: "road_validated",
          road_validated: true,
          display_role: "route_checked_draft",
        },
      };
    }),
  };
}

function addDraftRouteLayers(map, routeLegs) {
  map.addSource("v2-draft-routes", { type: "geojson", data: routeLegs });
  map.addLayer({
    id: "v2-draft-route-casing",
    type: "line",
    source: "v2-draft-routes",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ffffff", "line-width": 8, "line-opacity": 0.9 },
  });
  map.addLayer({
    id: "v2-draft-route",
    type: "line",
    source: "v2-draft-routes",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-color": "#6d5bd0",
      "line-width": 5,
      "line-opacity": 0.98,
      "line-dasharray": [1.2, 1.5],
    },
  });
}

function addDraftWaypointMarkers({
  map,
  waypoints,
  interactionMode,
  selectedWaypointId,
  onCustomWaypointSelect,
  onCustomWaypointDrag,
}) {
  return waypoints.map((rawWaypoint) => {
    const waypoint = validateDraftWaypoint(rawWaypoint);
    const element = document.createElement("button");
    element.type = "button";
    element.className = `draft-waypoint-marker ${waypoint.kind === "route_waypoint" ? "is-route-waypoint" : "is-custom-stop"}`;
    if (waypoint.waypoint_id === selectedWaypointId) element.classList.add("is-selected");
    element.setAttribute("aria-label", `${waypoint.name}. ${waypoint.kind === "route_waypoint" ? "Route-only waypoint" : "Custom draft stop"}.`);
    element.textContent = waypoint.kind === "route_waypoint" ? "VIA" : "+";
    let wasDragged = false;
    element.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (wasDragged) {
        wasDragged = false;
        return;
      }
      onCustomWaypointSelect({ ...waypoint });
    });
    const marker = new window.maplibregl.Marker({
      element,
      draggable: interactionMode === "edit" && waypoint.waypoint_id === selectedWaypointId,
      anchor: "center",
    }).setLngLat(waypoint.coordinate).addTo(map);
    if (interactionMode === "edit" && waypoint.waypoint_id === selectedWaypointId) {
      marker.on("dragstart", () => {
        wasDragged = true;
      });
      marker.on("dragend", () => {
        const raw = marker.getLngLat();
        onCustomWaypointDrag({
          waypoint_id: waypoint.waypoint_id,
          longitude: raw.lng,
          latitude: raw.lat,
        });
      });
    }
    return marker;
  });
}

function addRouteLayers(map, routes, selectedRouteLegId, selectedDay) {
  map.addSource("v2-routes", { type: "geojson", data: routes });
  map.addLayer({
    id: "v2-route-original",
    type: "line",
    source: "v2-routes",
    filter: roleFilter("original"),
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-color": "#294a59",
      "line-width": 4,
      "line-opacity": 0.76,
      "line-dasharray": [1.5, 2.2],
      "line-offset": -2,
    },
  });
  map.addLayer({
    id: "v2-route-active-casing",
    type: "line",
    source: "v2-routes",
    filter: roleFilter("active"),
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ffffff", "line-width": 9, "line-opacity": 0.86 },
  });
  map.addLayer({
    id: "v2-route-active",
    type: "line",
    source: "v2-routes",
    filter: roleFilter("active"),
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#007f78", "line-width": 6, "line-opacity": 0.98 },
  });
  map.addLayer({
    id: "v2-route-selected-day",
    type: "line",
    source: "v2-routes",
    filter: Number.isInteger(selectedDay)
      ? ["==", ["get", "day"], selectedDay]
      : ["==", ["get", "day"], -1],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#eb6a27", "line-width": 7, "line-opacity": 0.72 },
  });
  map.addLayer({
    id: "v2-route-selected",
    type: "line",
    source: "v2-routes",
    filter: selectedRouteLegId
      ? ["==", ["get", "route_leg_id"], selectedRouteLegId]
      : ["==", ["get", "route_leg_id"], ""],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#f06423", "line-width": 8, "line-opacity": 1 },
  });
}

function exploratoryPinFeature(exploratoryPin) {
  if (exploratoryPin === null || exploratoryPin === undefined) {
    return { type: "FeatureCollection", features: [] };
  }
  const coordinate = [exploratoryPin.longitude, exploratoryPin.latitude];
  if (!isPosition(coordinate)) throw new Error("exploratory_pin_invalid");
  return {
    type: "FeatureCollection",
    features: [{
      type: "Feature",
      geometry: { type: "Point", coordinates: coordinate },
      properties: {
        interaction_state: "exploratory",
        validation_status: "raw_coordinate_only",
      },
    }],
  };
}

function poiCandidateFeatureCollections(candidates, visible) {
  if (!Array.isArray(candidates) || candidates.length > 10) {
    throw new Error("poi_candidate_layer_invalid");
  }
  if (!visible) {
    return {
      display: { type: "FeatureCollection", features: [] },
      access: { type: "FeatureCollection", features: [] },
    };
  }
  const identifiers = new Set();
  const display = [];
  const access = [];
  candidates.forEach((candidate) => {
    const displayCoordinate = [
      candidate?.place?.display_coordinate?.longitude,
      candidate?.place?.display_coordinate?.latitude,
    ];
    const accessCoordinate = [
      candidate?.selected_access_point?.coordinate?.longitude,
      candidate?.selected_access_point?.coordinate?.latitude,
    ];
    if (typeof candidate?.candidate_id !== "string" || !candidate.candidate_id
        || identifiers.has(candidate.candidate_id)
        || typeof candidate.place?.name !== "string" || !candidate.place.name
        || !isPosition(displayCoordinate) || !isPosition(accessCoordinate)) {
      throw new Error("poi_candidate_layer_invalid");
    }
    identifiers.add(candidate.candidate_id);
    const properties = {
      candidate_id: candidate.candidate_id,
      name: candidate.place.name,
      candidate_state: candidate.burden?.candidate_state || "unavailable",
      recommended: candidate.burden?.recommended === true,
    };
    display.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: displayCoordinate },
      properties,
    });
    access.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: accessCoordinate },
      properties: {
        ...properties,
        road_validated: candidate.selected_access_point.road_validated === true,
        access_confidence: candidate.selected_access_point.access_confidence,
      },
    });
  });
  return {
    display: { type: "FeatureCollection", features: display },
    access: { type: "FeatureCollection", features: access },
  };
}

function addPoiCandidateLayers(map, candidates, selectedCandidateId) {
  map.addSource("v2-poi-candidates", { type: "geojson", data: candidates.display });
  map.addLayer({
    id: "v2-poi-candidates",
    type: "circle",
    source: "v2-poi-candidates",
    paint: {
      "circle-radius": ["case", ["==", ["get", "candidate_id"], selectedCandidateId || ""], 12, 9],
      "circle-color": ["case", ["==", ["get", "recommended"], true], "#0b7f79", "#7655c5"],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": 3,
      "circle-opacity": 0.96,
    },
  });
  map.addLayer({
    id: "v2-poi-candidate-labels",
    type: "symbol",
    source: "v2-poi-candidates",
    layout: {
      "text-field": ["get", "name"],
      "text-size": 11,
      "text-offset": [0, 1.45],
      "text-anchor": "top",
      "text-optional": true,
    },
    paint: {
      "text-color": "#2f285f",
      "text-halo-color": "#ffffff",
      "text-halo-width": 1.5,
    },
  });
  map.addSource("v2-poi-access-points", { type: "geojson", data: candidates.access });
  map.addLayer({
    id: "v2-poi-access-points",
    type: "circle",
    source: "v2-poi-access-points",
    paint: {
      "circle-radius": 4,
      "circle-color": "#ffffff",
      "circle-stroke-color": "#0b7f79",
      "circle-stroke-width": 2,
    },
  });
}

function exactCandidateSelection(event, onCandidateSelect) {
  const properties = event.features?.[0]?.properties;
  if (!properties || typeof properties.candidate_id !== "string" || !properties.candidate_id) return;
  onCandidateSelect({ candidate_id: properties.candidate_id });
}

function addPointLayers(map, stops, anchors, gaps, exploratoryPin, selectedDay) {
  map.addSource("v2-stops", { type: "geojson", data: stops });
  map.addLayer({
    id: "v2-stops-original",
    type: "circle",
    source: "v2-stops",
    filter: roleFilter("original"),
    paint: {
      "circle-radius": 5,
      "circle-color": "#ffffff",
      "circle-stroke-color": "#294a59",
      "circle-stroke-width": 2,
    },
  });
  map.addLayer({
    id: "v2-stops-active",
    type: "circle",
    source: "v2-stops",
    filter: roleFilter("active"),
    paint: {
      "circle-radius": 7,
      "circle-color": "#ffffff",
      "circle-stroke-color": "#007f78",
      "circle-stroke-width": 3,
    },
  });
  map.addLayer({
    id: "v2-stops-selected-day",
    type: "circle",
    source: "v2-stops",
    filter: Number.isInteger(selectedDay)
      ? ["==", ["get", "day"], selectedDay]
      : ["==", ["get", "day"], -1],
    paint: {
      "circle-radius": 9,
      "circle-color": "#ffffff",
      "circle-stroke-color": "#eb6a27",
      "circle-stroke-width": 4,
    },
  });
  map.addSource("v2-route-anchors", { type: "geojson", data: anchors });
  map.addLayer({
    id: "v2-route-anchors",
    type: "circle",
    source: "v2-route-anchors",
    paint: {
      "circle-radius": 4,
      "circle-color": "#294a59",
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": 1.5,
    },
  });
  map.addSource("v2-route-gaps", { type: "geojson", data: gaps });
  map.addLayer({
    id: "v2-route-gap-markers",
    type: "circle",
    source: "v2-route-gaps",
    paint: {
      "circle-radius": 9,
      "circle-color": "#fff3e8",
      "circle-stroke-color": "#d84a05",
      "circle-stroke-width": 3,
    },
  });
  map.addLayer({
    id: "v2-route-gap-labels",
    type: "symbol",
    source: "v2-route-gaps",
    layout: { "text-field": "!", "text-size": 13 },
    paint: { "text-color": "#9f2f00" },
  });
  map.addSource("v2-exploratory-pin", { type: "geojson", data: exploratoryPin });
  map.addLayer({
    id: "v2-exploratory-pin",
    type: "circle",
    source: "v2-exploratory-pin",
    paint: {
      "circle-radius": 10,
      "circle-color": "#ffffff",
      "circle-opacity": 0.82,
      "circle-stroke-color": "#7a52c7",
      "circle-stroke-width": 3,
      "circle-stroke-opacity": 0.95,
    },
  });
}

function exactRouteLegSelection(event, onRouteLegSelect) {
  const properties = event.features?.[0]?.properties;
  if (!properties || typeof properties.route_leg_id !== "string" || !properties.route_leg_id) return;
  onRouteLegSelect({
    plan_id: properties.plan_id,
    route_leg_id: properties.route_leg_id,
    origin_id: properties.origin_id,
    destination_id: properties.destination_id,
    day: properties.day ?? null,
  });
}

function exactStopSelection(event, onStopSelect) {
  const properties = event.features?.[0]?.properties;
  if (!properties || typeof properties.stop_id !== "string" || !properties.stop_id) return;
  onStopSelect({
    plan_id: properties.plan_id,
    stop_id: properties.stop_id,
    day: properties.day ?? null,
  });
}

function exactGapSelection(event, onGapSelect) {
  const properties = event.features?.[0]?.properties;
  if (!properties || properties.validation_status !== "unvalidated_gap") return;
  onGapSelect({
    plan_id: properties.plan_id,
    requirement_id: properties.requirement_id || null,
    route_leg_id: properties.route_leg_id || null,
    origin_id: properties.origin_id || null,
    destination_id: properties.destination_id || null,
    failure_code: properties.failure_code,
  });
}

function textFallbackFor(plans, gaps) {
  const routeLegs = plans.flatMap((plan) => plan.validated_legs.features.map((feature) => ({
    plan_id: plan.plan_id,
    route_leg_id: feature.properties.route_leg_id,
    origin_id: feature.properties.origin_id,
    destination_id: feature.properties.destination_id,
    text: `${feature.properties.origin_id} to ${feature.properties.destination_id}: road-validated route.`,
  })));
  return Object.freeze({
    summary: gaps.length === 0
      ? `${routeLegs.length} road-validated route legs are displayed.`
      : `${routeLegs.length} road-validated route legs and ${gaps.length} unvalidated route gaps.`,
    route_legs: Object.freeze(routeLegs),
    gaps,
  });
}

function diagnosticsFor(plans, gaps) {
  const routes = plans.flatMap((plan) => plan.validated_legs.features);
  const routePath = plans.flatMap((plan) => plan.route_path.features);
  return Object.freeze({
    visiblePlanCount: plans.length,
    stopCount: plans.reduce((count, plan) => count + plan.stops.features.length, 0),
    routeAnchorCount: routePath.filter(
      (feature) => feature.properties.route_anchor === true && feature.geometry !== null,
    ).length,
    routeLegCount: routes.length,
    roadValidatedLegCount: routes.length,
    gapCount: gaps.length,
    unresolvedGapCount: gaps.filter((gap) => !gap.marker_available).length,
  });
}

function boundsForPlans(plans, fallbackBounds, draftWaypoints = [], draftRoutes = null) {
  const positions = [];
  plans.forEach((plan) => {
    [plan.route_path, plan.stops, plan.validated_legs].forEach((collection) => {
      collection.features.forEach((feature) => {
        const coordinates = feature.geometry?.coordinates;
        if (isPosition(coordinates)) positions.push(coordinates);
        else if (Array.isArray(coordinates)) coordinates.forEach((point) => {
          if (isPosition(point)) positions.push(point);
        });
      });
    });
  });
  draftWaypoints.forEach((waypoint) => {
    if (isPosition(waypoint.coordinate)) positions.push(waypoint.coordinate);
  });
  draftRoutes?.features?.forEach((feature) => {
    feature.geometry?.coordinates?.forEach((point) => {
      if (isPosition(point)) positions.push(point);
    });
  });
  if (positions.length === 0) return fallbackBounds;
  const longitudes = positions.map((point) => point[0]);
  const latitudes = positions.map((point) => point[1]);
  const result = [
    Math.min(...longitudes),
    Math.min(...latitudes),
    Math.max(...longitudes),
    Math.max(...latitudes),
  ];
  return validateBounds(result) ? result : fallbackBounds;
}

function boundsForDay(plans, selectedDay) {
  if (!Number.isInteger(selectedDay)) return null;
  const positions = [];
  plans.forEach((plan) => {
    [plan.route_path, plan.stops, plan.validated_legs].forEach((collection) => {
      collection.features
        .filter((feature) => Number(feature.properties?.day) === selectedDay)
        .forEach((feature) => {
          const coordinates = feature.geometry?.coordinates;
          if (isPosition(coordinates)) positions.push(coordinates);
          else if (Array.isArray(coordinates)) coordinates.forEach((point) => {
            if (isPosition(point)) positions.push(point);
          });
        });
    });
  });
  if (positions.length < 2) return null;
  const result = [
    Math.min(...positions.map((point) => point[0])),
    Math.min(...positions.map((point) => point[1])),
    Math.max(...positions.map((point) => point[0])),
    Math.max(...positions.map((point) => point[1])),
  ];
  return validateBounds(result) ? result : null;
}

/**
 * Create one instance-owned visual map controller.
 *
 * The WebGL canvas is hidden from assistive technology. The caller owns the
 * equivalent DOM stop, route-leg, gap, and action controls and receives a
 * textual fallback before the visual runtime is loaded.
 */
export function createGeographicMapController({
  container,
  mapConfig,
  paneId = "map",
  onRouteLegSelect = () => {},
  onStopSelect = () => {},
  onCandidateSelect = () => {},
  onGapSelect = () => {},
  onEmptyMapClick = () => {},
  onCustomWaypointSelect = () => {},
  onCustomWaypointDrag = () => {},
  onTextFallback = () => {},
  onStatus = () => {},
}) {
  let map = null;
  let generation = 0;
  let lastDiagnostics = null;
  let suppressNextCameraEvent = false;
  let customMarkers = [];
  let resizeObserver = null;
  let visibilityCleanup = null;
  let resizeFrame = null;
  const cameraListeners = new Set();

  function emitStatus(payload) {
    onStatus({ pane_id: paneId, ...payload });
  }

  function waitForVisibleContainer(renderGeneration) {
    const visible = () => {
      const rect = container.getBoundingClientRect();
      return container.isConnected && rect.width >= 1 && rect.height >= 1;
    };
    if (visible()) return Promise.resolve(true);
    return new Promise((resolve) => {
      let settled = false;
      const finish = (result) => {
        if (settled) return;
        settled = true;
        observer.disconnect();
        if (visibilityCleanup === cancel) visibilityCleanup = null;
        resolve(result);
      };
      const observer = new ResizeObserver(() => {
        if (renderGeneration !== generation) finish(false);
        else if (visible()) finish(true);
      });
      const cancel = () => finish(false);
      visibilityCleanup = cancel;
      observer.observe(container);
    });
  }

  function observeMapSize(nextMap, renderGeneration) {
    const resize = () => {
      if (renderGeneration !== generation || map !== nextMap || !container.isConnected) return;
      if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = null;
        if (renderGeneration === generation && map === nextMap && container.isConnected) {
          nextMap.resize();
        }
      });
    };
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }

  async function render({
    geography,
    activePlanId,
    selectedRouteLegId = null,
    selectedDay = null,
    interactionMode = "select",
    exploratoryPin = null,
    draftWaypoints = [],
    draftRouteLegs = [],
    selectedWaypointId = null,
    poiCandidates = [],
    selectedCandidateId = null,
    showPoiCandidates = false,
  }) {
    disposeMap();
    const renderGeneration = generation;
    try {
      if (!["select", "edit"].includes(interactionMode)) {
        throw new Error("map_interaction_mode_invalid");
      }
      const selected = selectVisiblePlans(geography, activePlanId);
      const originalPlanId = selected.original.plan_id;
      const routes = collectFeatureCollection(
        selected.plans, "validated_legs", originalPlanId, activePlanId,
      );
      const stops = collectFeatureCollection(selected.plans, "stops", originalPlanId, activePlanId);
      const anchors = deriveRouteAnchors(selected.plans, originalPlanId, activePlanId);
      const resolvedGaps = resolveGapMarkers(selected.plans, originalPlanId, activePlanId);
      const provisionalPin = exploratoryPinFeature(exploratoryPin);
      const checkedDraftRoutes = draftRouteFeatureCollection(draftRouteLegs);
      const candidatePoints = poiCandidateFeatureCollections(poiCandidates, showPoiCandidates);
      draftWaypoints.forEach(validateDraftWaypoint);
      const textualFallback = textFallbackFor(selected.plans, resolvedGaps.descriptions);
      lastDiagnostics = diagnosticsFor(selected.plans, resolvedGaps.descriptions);
      onTextFallback({ pane_id: paneId, ...textualFallback });

      const style = await loadMapRuntime(mapConfig);
      if (renderGeneration !== generation || !container.isConnected
          || !(await waitForVisibleContainer(renderGeneration))) return null;
      const fittedBounds = boundsForDay(selected.plans, selectedDay) || boundsForPlans(
        selected.plans, selected.fallbackBounds, draftWaypoints, checkedDraftRoutes,
      );
      container.dataset.mapReady = "false";
      const nextMap = new window.maplibregl.Map({
        container,
        style,
        bounds: fittedBounds,
        fitBoundsOptions: { padding: 54, maxZoom: 13, duration: 0 },
        attributionControl: false,
        cooperativeGestures: true,
      });
      map = nextMap;
      const removeWindowResize = observeMapSize(nextMap, renderGeneration);
      hideVisualCanvas(nextMap);
      nextMap.once("load", () => {
        if (renderGeneration !== generation || map !== nextMap) return;
        addRouteLayers(nextMap, routes, selectedRouteLegId, selectedDay);
        addDraftRouteLayers(nextMap, checkedDraftRoutes);
        addPointLayers(
          nextMap, stops, anchors, resolvedGaps.markers, provisionalPin, selectedDay,
        );
        addPoiCandidateLayers(nextMap, candidatePoints, selectedCandidateId);
        customMarkers = addDraftWaypointMarkers({
          map: nextMap,
          waypoints: draftWaypoints,
          interactionMode,
          selectedWaypointId,
          onCustomWaypointSelect,
          onCustomWaypointDrag,
        });
        ["v2-route-active", "v2-route-original"].forEach((layerId) => {
          nextMap.on("click", layerId, (event) => exactRouteLegSelection(event, onRouteLegSelect));
        });
        ["v2-stops-active", "v2-stops-original"].forEach((layerId) => {
          nextMap.on("click", layerId, (event) => exactStopSelection(event, onStopSelect));
        });
        nextMap.on("click", "v2-route-gap-markers", (event) => exactGapSelection(event, onGapSelect));
        ["v2-poi-candidates", "v2-poi-candidate-labels", "v2-poi-access-points"]
          .forEach((layerId) => nextMap.on(
            "click", layerId, (event) => exactCandidateSelection(event, onCandidateSelect),
          ));
        nextMap.on("click", (event) => {
          if (interactionMode !== "edit") return;
          const occupied = nextMap.queryRenderedFeatures(event.point, {
            layers: [
              "v2-route-original",
              "v2-route-active",
              "v2-route-selected",
              "v2-route-selected-day",
              "v2-draft-route",
              "v2-stops-original",
              "v2-stops-active",
              "v2-stops-selected-day",
              "v2-route-anchors",
              "v2-route-gap-markers",
              "v2-exploratory-pin",
              "v2-poi-candidates",
              "v2-poi-candidate-labels",
              "v2-poi-access-points",
            ],
          });
          if (occupied.length > 0) return;
          onEmptyMapClick({
            longitude: event.lngLat.lng,
            latitude: event.lngLat.lat,
            validation_status: "raw_coordinate_only",
          });
        });
        nextMap.on("moveend", () => {
          if (suppressNextCameraEvent) {
            suppressNextCameraEvent = false;
            return;
          }
          const center = nextMap.getCenter();
          const camera = {
            center: [center.lng, center.lat],
            zoom: nextMap.getZoom(),
            bearing: nextMap.getBearing(),
            pitch: nextMap.getPitch(),
          };
          cameraListeners.forEach((listener) => listener(camera));
        });
        hideVisualCanvas(nextMap);
        nextMap.once("idle", () => {
          if (renderGeneration !== generation || map !== nextMap) return;
          nextMap.resize();
          const tileCoverageRatio = renderedCanvasCoverage(container, nextMap);
          const tilesLoaded = typeof nextMap.areTilesLoaded !== "function"
            || nextMap.areTilesLoaded();
          if (!tilesLoaded || tileCoverageRatio < 0.95) {
            emitStatus({
              state: "failed",
              code: "map_tile_coverage_incomplete",
              textual_fallback: true,
              tileCoverageRatio,
              tilesLoaded,
            });
            return;
          }
          container.dataset.mapReady = "true";
          emitStatus({
            state: lastDiagnostics.gapCount > 0 ? "ready_with_gaps" : "ready",
            code: lastDiagnostics.gapCount > 0
              ? "geographic_map_v2_has_route_gaps"
              : "geographic_map_v2_ready",
            gaps: resolvedGaps.descriptions,
            tileCoverageRatio,
            tilesLoaded,
            ...lastDiagnostics,
          });
        });
        nextMap.resize();
        nextMap.fitBounds(fittedBounds, { padding: 54, maxZoom: 13, duration: 0 });
      });
      nextMap.on("error", () => {
        emitStatus({ state: "failed", code: "map_render_failed", textual_fallback: true });
      });
      nextMap.once("remove", removeWindowResize);
      return nextMap;
    } catch (error) {
      if (renderGeneration === generation) {
        emitStatus({
          state: "failed",
          code: error?.message || "map_render_failed",
          textual_fallback: lastDiagnostics !== null,
        });
      }
      return null;
    }
  }

  function diagnostics() {
    return lastDiagnostics;
  }

  function subscribeCamera(listener) {
    if (typeof listener !== "function") throw new Error("camera_listener_invalid");
    cameraListeners.add(listener);
    return () => cameraListeners.delete(listener);
  }

  function applySynchronizedCamera(camera) {
    if (!map || !isPosition(camera?.center) || !isFiniteNumber(camera.zoom)
        || !isFiniteNumber(camera.bearing) || !isFiniteNumber(camera.pitch)) return false;
    const currentCenter = map.getCenter();
    const unchanged = Math.abs(currentCenter.lng - camera.center[0]) < 1e-9
      && Math.abs(currentCenter.lat - camera.center[1]) < 1e-9
      && Math.abs(map.getZoom() - camera.zoom) < 1e-9
      && Math.abs(map.getBearing() - camera.bearing) < 1e-9
      && Math.abs(map.getPitch() - camera.pitch) < 1e-9;
    if (unchanged) return true;
    suppressNextCameraEvent = true;
    map.jumpTo(camera);
    setTimeout(() => {
      suppressNextCameraEvent = false;
    }, 0);
    return true;
  }

  function disposeMap() {
    generation += 1;
    visibilityCleanup?.();
    visibilityCleanup = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
    resizeFrame = null;
    delete container.dataset.mapReady;
    customMarkers.forEach((marker) => marker.remove());
    customMarkers = [];
    if (map) {
      map.remove();
      map = null;
    }
  }

  function dispose() {
    disposeMap();
    cameraListeners.clear();
    lastDiagnostics = null;
  }

  return Object.freeze({
    render,
    diagnostics,
    subscribeCamera,
    applySynchronizedCamera,
    dispose,
  });
}

/** Link two pane-owned controllers without allowing move events to recurse. */
export function synchronizeMapCameras(leftController, rightController) {
  let synchronizationInProgress = false;
  const synchronize = (target, camera) => {
    if (synchronizationInProgress) return;
    synchronizationInProgress = true;
    try {
      target.applySynchronizedCamera(camera);
    } finally {
      synchronizationInProgress = false;
    }
  };
  const unsubscribeLeft = leftController.subscribeCamera(
    (camera) => synchronize(rightController, camera),
  );
  const unsubscribeRight = rightController.subscribeCamera(
    (camera) => synchronize(leftController, camera),
  );
  return () => {
    unsubscribeLeft();
    unsubscribeRight();
  };
}
