import { addInterestAreaLayers, addInterestPointLayers } from "./interest-regions.js";

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
      || !["original", "registered_repair", "alternative", "draft_preview", "accepted"].includes(plan.role)
      || !READY_PLAN_STATES.has(plan.status)
      || !validateCoverage(plan.coverage)
      || !isPointFeatureCollection(plan.stops)
      || !isPointFeatureCollection(plan.route_path, { nullable: true })
      || !isLineFeatureCollection(plan.validated_legs)
      || !isPointFeatureCollection(plan.gaps, { nullable: true })) {
    throw new Error("geography_plan_invalid");
  }
  const marine = plan.marine_routes;
  const islands = plan.offshore_stops;
  if (marine || islands) {
    const port = plan.stops.features.find(f => f.properties.stop_id === "channel_islands");
    if (!isLineFeatureCollection(marine) || !isPointFeatureCollection(islands)
        || marine.features.length !== 1 || islands.features.length !== 1 || !port
        || ![marine.features[0], islands.features[0]].every(f =>
          f.properties.plan_id === plan.plan_id && f.properties.content_hash === plan.content_hash
          && f.properties.source_stop_id === "channel_islands" && f.properties.day === port.properties.day
          && f.properties.travel_mode === "passenger_ferry" && f.properties.road_validated === false
          && f.properties.geometry_source === "schematic_port_to_island" && f.properties.round_trip === true
          && f.properties.schedule_verified === false && f.properties.reservation_verified === false)
        || JSON.stringify(marine.features[0].geometry.coordinates)
          !== JSON.stringify([port.geometry.coordinates, [-119.5562112, 34.0493266]])
        || JSON.stringify(islands.features[0].geometry.coordinates) !== JSON.stringify([-119.5562112, 34.0493266])) {
      throw new Error("geography_island_excursion_invalid");
    }
  } else if (plan.stops.features.some(f => f.properties.stop_id === "channel_islands")) {
    throw new Error("geography_island_excursion_missing");
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

// An extract retains complete tiles at its boundary. Outside detailed coverage,
// use only a coarse zoom whose retained tile rectangles contain the whole view.
// Road/stop layers keep their original geometry and the readiness check is unchanged.
export function retainedOverview(bounds, header) {
  if (!validateBounds(bounds)) throw new Error("basemap_bounds_invalid");
  const native = [header.minLon, header.minLat, header.maxLon, header.maxLat];
  if (!native.every(Number.isFinite) || !Number.isInteger(header.maxZoom)) throw new Error("basemap_header_invalid");
  const dx = Math.max(0.1, bounds[2] - bounds[0]);
  const dy = Math.max(0.1, bounds[3] - bounds[1]);
  const padded = [bounds[0]-dx, bounds[1]-dy, bounds[2]+dx, bounds[3]+dy];
  // A fitted viewport extends beyond the route itself, including at archive edges.
  if (padded[0] >= native[0] && padded[1] >= native[1]
      && padded[2] <= native[2] && padded[3] <= native[3]) return null;
  const tileY = (lat, n) => (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n;
  const latitude = (y, n) => Math.atan(Math.sinh(Math.PI * (1 - 2*y/n))) * 180 / Math.PI;
  for (let zoom = header.maxZoom; zoom >= 0; zoom -= 1) {
    const n = 2 ** zoom;
    const west = Math.floor((native[0]+180)/360*n)/n*360-180;
    const east = (Math.floor((native[2]+180)/360*n)+1)/n*360-180;
    const north = latitude(Math.floor(tileY(native[3],n)),n);
    const south = latitude(Math.floor(tileY(native[1],n))+1,n);
    if (padded[0] >= west && padded[1] >= south && padded[2] <= east && padded[3] <= north) {
      return {maxzoom:zoom,bounds:[west,south,east,north]};
    }
  }
  throw new Error("basemap_overview_unavailable");
}

async function styleForRetainedCoverage(styleUrl, baseUrl, bounds) {
  const response = await fetch(styleUrl, {credentials:"omit"});
  if (!response.ok) throw new Error("basemap_style_unavailable");
  const style = await response.json();
  let overview = false;
  for (const source of Object.values(style.sources || {})) {
    if (source.type !== "vector" || !source.url?.startsWith("pmtiles://")) continue;
    const archiveUrl = localAssetUrl(source.url.slice("pmtiles://".length), baseUrl);
    const header = await new window.pmtiles.PMTiles(archiveUrl).getHeader();
    const retained = retainedOverview(bounds, header);
    if (retained) {
      source.maxzoom = retained.maxzoom;
      source.bounds = retained.bounds;
      overview = true;
    }
  }
  return {style, overview};
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

function canvasViewportCoverage(container, map) {
  const containerRect = container.getBoundingClientRect();
  const canvasRect = map.getCanvas().getBoundingClientRect();
  if (containerRect.width <= 0 || containerRect.height <= 0) return 0;
  const width = Math.max(0, Math.min(containerRect.right, canvasRect.right)
    - Math.max(containerRect.left, canvasRect.left));
  const height = Math.max(0, Math.min(containerRect.bottom, canvasRect.bottom)
    - Math.max(containerRect.top, canvasRect.top));
  return (width * height) / (containerRect.width * containerRect.height);
}

export function invalidateMapReadiness(container) {
  container.dataset.mapReady = "false";
  delete container.dataset.mapReadyAt;
  delete container.dataset.mapTileCoverage;
  delete container.dataset.mapCoverageMethod;
  delete container.dataset.renderedMapPlanId;
  delete container.dataset.renderedPlanContentHash;
  delete container.dataset.renderedGeometryHash;
  delete container.dataset.renderedMarkerIds;
}

function clippedPolygonArea(points, width, height) {
  let polygon = points;
  for (const [axis, limit, direction] of [["x", 0, 1], ["x", width, -1], ["y", 0, 1], ["y", height, -1]]) {
    const output = [];
    for (let index = 0; index < polygon.length; index += 1) {
      const from = polygon[index];
      const to = polygon[(index + 1) % polygon.length];
      const fromInside = direction * (from[axis] - limit) >= 0;
      const toInside = direction * (to[axis] - limit) >= 0;
      if (fromInside) output.push(from);
      if (fromInside !== toInside) {
        const fraction = (limit - from[axis]) / (to[axis] - from[axis]);
        output.push({ x: from.x + fraction * (to.x - from.x), y: from.y + fraction * (to.y - from.y) });
      }
    }
    polygon = output;
  }
  return Math.abs(polygon.reduce((area, point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    return area + point.x * next.y - next.x * point.y;
  }, 0)) / 2;
}

function loadedTileViewportCoverage(map, sourceId) {
  // This adapter intentionally targets the locally pinned MapLibre tile manager.
  // Aggregate isSourceLoaded/areTilesLoaded may also be true for failed tiles.
  // Unknown renderer internals fail closed; they are not a coverage fallback.
  const manager = map.style?.tileManagers?.[sourceId];
  if (typeof manager?.getVisibleCoordinates !== "function" || typeof manager?.getTile !== "function"
      || typeof map.getContainer !== "function" || typeof map.project !== "function"
      || (map.getProjection?.()?.type || "mercator") !== "mercator") return 0;
  const { width, height } = map.getContainer().getBoundingClientRect();
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return 0;
  const boxes = [];
  for (const id of manager.getVisibleCoordinates()) {
    const tile = manager.getTile(id);
    if (tile?.state !== "loaded" || tile.isRenderable?.() !== true) continue;
    const { x, y, z } = id.canonical || {};
    const wrap = id.wrap;
    const size = 2 ** z;
    if (![x, y, z, wrap].every(Number.isInteger) || z < 0 || z > 24
        || x < 0 || y < 0 || x >= size || y >= size) return 0;
    boxes.push({ left: x / size + wrap, right: (x + 1) / size + wrap, top: y / size, bottom: (y + 1) / size });
  }
  // Union the quadtree rectangles in Mercator space first, avoiding double-counted
  // parent/child fallback tiles. Project disjoint strips, clip, then sum screen area.
  const edges = [...new Set(boxes.flatMap(box => [box.left, box.right]))].sort((a, b) => a - b);
  let area = 0;
  const project = (x, y) => {
    const point = map.project([x * 360 - 180, Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI]);
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) throw new Error("map_tile_projection_invalid");
    return point;
  };
  for (let index = 0; index < edges.length - 1; index += 1) {
    const left = edges[index];
    const right = edges[index + 1];
    const intervals = boxes.filter(box => box.left <= left && box.right >= right)
      .map(box => [box.top, box.bottom]).sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const interval of intervals) {
      const previous = merged.at(-1);
      if (previous && interval[0] <= previous[1]) previous[1] = Math.max(previous[1], interval[1]);
      else merged.push([...interval]);
    }
    for (const [top, bottom] of merged) {
      area += clippedPolygonArea([project(left, top), project(right, top), project(right, bottom), project(left, bottom)], width, height);
    }
  }
  return Math.max(0, Math.min(1, area / (width * height)));
}

export function mapTileReadiness(map) {
  if (typeof map?.areTilesLoaded !== "function"
      || typeof map?.getStyle !== "function"
      || typeof map?.isSourceLoaded !== "function") {
    return Object.freeze({ tilesLoaded: false, tileCoverageRatio: 0, tileSourceCount: 0 });
  }
  try {
    const sources = Object.entries(map.getStyle()?.sources || {})
      .filter(([, source]) => ["vector", "raster", "raster-dem"].includes(source?.type));
    if (sources.length === 0) {
      return Object.freeze({ tilesLoaded: false, tileCoverageRatio: 0, tileSourceCount: 0 });
    }
    const loadedSourceCount = sources.filter(([sourceId]) => map.isSourceLoaded(sourceId) === true).length;
    const tileCoverageRatio = Math.min(...sources.map(([sourceId]) => loadedTileViewportCoverage(map, sourceId)));
    return Object.freeze({
      tilesLoaded: map.areTilesLoaded() === true && loadedSourceCount === sources.length && tileCoverageRatio >= 0.95,
      tileCoverageRatio,
      tileSourceCount: sources.length,
    });
  } catch {
    return Object.freeze({ tilesLoaded: false, tileCoverageRatio: 0, tileSourceCount: 0 });
  }
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

function addRouteLayers(map, routes, selectedRouteLegId, selectedDay, activePlanId) {
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
      ? ["all", ["==", ["get", "day"], selectedDay], ["==", ["get", "plan_id"], activePlanId]]
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
    layout: { "text-field": "!", "text-font": ["Noto Sans Regular"], "text-size": 13 },
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

function canonicalGeometryJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalGeometryJson).join(",")}]`;
  return `{${Object.keys(value).sort().map(
    (key) => `${JSON.stringify(key)}:${canonicalGeometryJson(value[key])}`,
  ).join(",")}}`;
}

export async function routeGeometrySha256(plan) {
  if (!plan) return null;
  const semanticLegs = {
    ...plan.validated_legs,
    features: plan.validated_legs.features.map((feature) => ({
      ...feature,
      properties: Object.fromEntries(
        Object.entries(feature.properties).filter(([key]) => key !== "role"),
      ),
    })),
  };
  const geometryPayload = plan.marine_routes?.features.length
    ? {road: semanticLegs, marine: plan.marine_routes, offshore: plan.offshore_stops} : semanticLegs;
  const bytes = new TextEncoder().encode(canonicalGeometryJson(geometryPayload));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function diagnosticsFor(plans, gaps, activePlanId) {
  const routes = plans.flatMap((plan) => plan.validated_legs.features);
  const routePath = plans.flatMap((plan) => plan.route_path.features);
  const activePlan = plans.find((plan) => plan.plan_id === activePlanId);
  return Object.freeze({
    activePlanId: activePlan?.plan_id || null,
    activePlanContentHash: activePlan?.content_hash || null,
    activeGeometryHash: await routeGeometrySha256(activePlan),
    activeMarkerIds: Object.freeze((activePlan?.stops?.features || [])
      .map((feature) => feature.properties?.stop_id)
      .filter(Boolean)
      .sort()),
    visiblePlanCount: plans.length,
    stopCount: activePlan?.stops?.features.length || 0,
    routeAnchorCount: routePath.filter(
      (feature) => feature.properties.route_anchor === true && feature.geometry !== null,
    ).length,
    routeLegCount: routes.length,
    roadValidatedLegCount: routes.length,
    marineSchematicLegCount: activePlan?.marine_routes?.features.length || 0,
    gapCount: gaps.length,
    unresolvedGapCount: gaps.filter((gap) => !gap.marker_available).length,
  });
}

function boundsForPlans(plans, fallbackBounds, draftWaypoints = [], draftRoutes = null) {
  const positions = [];
  plans.forEach((plan) => {
    [plan.route_path, plan.stops, plan.validated_legs, plan.marine_routes, plan.offshore_stops].filter(Boolean).forEach((collection) => {
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
    [plan.route_path, plan.stops, plan.validated_legs, plan.marine_routes, plan.offshore_stops].filter(Boolean).forEach((collection) => {
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
  onInterestSelect = () => {},
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

  function observeMapSize(nextMap, renderGeneration, validateReady) {
    let revalidationPending = false;
    let resizeSequence = 0;
    let observedWidth = null;
    let observedHeight = null;
    const resize = () => {
      if (renderGeneration !== generation || map !== nextMap || !container.isConnected) return;
      const rect = container.getBoundingClientRect();
      const sizeChanged = observedWidth === null || observedHeight === null
        || Math.abs(rect.width - observedWidth) >= 0.5
        || Math.abs(rect.height - observedHeight) >= 0.5;
      observedWidth = rect.width;
      observedHeight = rect.height;
      if (!sizeChanged) return;
      const wasReady = container.dataset.mapReady === "true";
      revalidationPending = revalidationPending || wasReady;
      if (wasReady) {
        invalidateMapReadiness(container);
        emitStatus({ state: "loading", code: "map_resize_revalidation_pending" });
      }
      resizeSequence += 1;
      const scheduledSequence = resizeSequence;
      if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = null;
        if (renderGeneration !== generation || map !== nextMap || !container.isConnected) return;
        if (revalidationPending) {
          nextMap.once("idle", () => {
            if (renderGeneration !== generation || map !== nextMap
                || scheduledSequence !== resizeSequence) return;
            revalidationPending = false;
            validateReady();
          });
        }
        nextMap.resize();
        nextMap.triggerRepaint?.();
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
    interestRegions = null,
    interestMassLevel = 0.8,
    selectedInterestId = null,
    initialCamera = null,
  }) {
    disposeMap();
    const renderGeneration = generation;
    try {
      emitStatus({ state: "loading", code: "map_render_revalidation_pending" });
      if (!["select", "edit"].includes(interactionMode)) {
        throw new Error("map_interaction_mode_invalid");
      }
      if (initialCamera !== null && (!isPosition(initialCamera.center)
          || ![initialCamera.zoom, initialCamera.bearing, initialCamera.pitch].every(isFiniteNumber))) {
        throw new Error("map_camera_invalid");
      }
      const selected = selectVisiblePlans(geography, activePlanId);
      const originalPlanId = selected.original.plan_id;
      const routes = collectFeatureCollection(
        selected.plans, "validated_legs", originalPlanId, activePlanId,
      );
      const stops = collectFeatureCollection(
        selected.plans.filter(plan => plan.plan_id === activePlanId), "stops", originalPlanId, activePlanId,
      );
      const activePlan = selected.plans.find(p => p.plan_id === activePlanId);
      const marineRoutes = activePlan?.marine_routes || {type:"FeatureCollection",features:[]};
      const offshoreStops = activePlan?.offshore_stops || {type:"FeatureCollection",features:[]};
      const anchors = deriveRouteAnchors(selected.plans, originalPlanId, activePlanId);
      const resolvedGaps = resolveGapMarkers(selected.plans, originalPlanId, activePlanId);
      const provisionalPin = exploratoryPinFeature(exploratoryPin);
      const checkedDraftRoutes = draftRouteFeatureCollection(draftRouteLegs);
      const candidatePoints = poiCandidateFeatureCollections(poiCandidates, showPoiCandidates);
      draftWaypoints.forEach(validateDraftWaypoint);
      const textualFallback = textFallbackFor(selected.plans, resolvedGaps.descriptions);
      const renderDiagnostics = await diagnosticsFor(selected.plans, resolvedGaps.descriptions, activePlanId);
      if (renderGeneration !== generation || !container.isConnected) return null;
      lastDiagnostics = renderDiagnostics;
      onTextFallback({ pane_id: paneId, ...textualFallback });

      const style = await loadMapRuntime(mapConfig);
      if (renderGeneration !== generation || !container.isConnected
          || !(await waitForVisibleContainer(renderGeneration))) return null;
      const currentInterest = interestRegions?.exploration_plan_id === activePlanId
        && interestRegions.exploration_plan_content_hash === activePlan?.content_hash
        && interestRegions.context?.day === selectedDay ? interestRegions : null;
      const interestPositions = currentInterest ? [
        ...currentInterest.poi_predictions.map(point => [point.display_coordinate.longitude, point.display_coordinate.latitude]),
        ...currentInterest.required_points.filter(point => point.coordinate).map(point => point.coordinate),
        ...currentInterest.surface.regions_geojson.features.flatMap(feature => feature.geometry.coordinates.flat(2)),
      ] : [];
      const interestBounds = interestPositions.length ? [
        Math.min(...interestPositions.map(point => point[0])), Math.min(...interestPositions.map(point => point[1])),
        Math.max(...interestPositions.map(point => point[0])), Math.max(...interestPositions.map(point => point[1])),
      ] : null;
      const focusedInterest = currentInterest?.poi_predictions.find(point => point.place_id === selectedInterestId);
      const focusCoordinate = focusedInterest
        ? [focusedInterest.display_coordinate.longitude, focusedInterest.display_coordinate.latitude]
        : currentInterest?.required_points.find(point => point.place_id === selectedInterestId)?.coordinate;
      const focusBounds = focusCoordinate ? [focusCoordinate[0] - 0.015, focusCoordinate[1] - 0.01,
        focusCoordinate[0] + 0.015, focusCoordinate[1] + 0.01] : null;
      const fittedBounds = focusBounds || (validateBounds(interestBounds) ? interestBounds : null)
        || boundsForDay(selected.plans, selectedDay) || boundsForPlans(
        selected.plans, selected.fallbackBounds, draftWaypoints, checkedDraftRoutes,
      );
      const basemap = await styleForRetainedCoverage(style, mapConfig.base_url, fittedBounds);
      if (renderGeneration !== generation || !container.isConnected) return null;
      container.dataset.basemapDetail = basemap.overview ? "overview" : "detailed";
      // A fixed 54px margin consumes over half a narrow Compare pane and can
      // zoom beyond the local archive. Keep useful margins proportional to it.
      const fitBoundsOptions = {
        padding: Math.min(54, Math.min(container.clientWidth, container.clientHeight) / 8),
        maxZoom: 13,
        duration: 0,
      };
      invalidateMapReadiness(container);
      const nextMap = new window.maplibregl.Map({
        container,
        style: basemap.style,
        ...(initialCamera || { bounds: fittedBounds, fitBoundsOptions }),
        attributionControl: false,
        cooperativeGestures: true,
        maxPitch: 0,
      });
      map = nextMap;
      nextMap.getCanvasContainer().addEventListener("pointerdown", (event) => {
        if (event.pointerType === "mouse" && event.button === 0
            && event.target instanceof Element
            && event.target.closest(".maplibregl-canvas, .draft-waypoint-marker")) {
          event.target.setPointerCapture(event.pointerId);
        }
      });
      let webglContextLost = false;
      const validateReady = () => {
        if (renderGeneration !== generation || map !== nextMap || webglContextLost) return false;
        const canvasViewportRatio = canvasViewportCoverage(container, nextMap);
        const readiness = mapTileReadiness(nextMap);
        if (!readiness.tilesLoaded || readiness.tileCoverageRatio < 0.95
            || canvasViewportRatio < 0.95) {
          invalidateMapReadiness(container);
          emitStatus({
            state: "failed",
            code: "map_tile_coverage_incomplete",
            textual_fallback: true,
            canvasViewportRatio,
            ...readiness,
          });
          return false;
        }
        if (container.dataset.mapReady === "true"
            && container.dataset.renderedMapPlanId === renderDiagnostics.activePlanId
            && container.dataset.renderedGeometryHash === renderDiagnostics.activeGeometryHash
            && Number(container.dataset.mapTileCoverage) === readiness.tileCoverageRatio) return true;
        const mapReadyAt = new Date().toISOString();
        container.dataset.mapReady = "true";
        container.dataset.mapReadyAt = mapReadyAt;
        container.dataset.mapTileCoverage = String(readiness.tileCoverageRatio);
        container.dataset.mapCoverageMethod = "loaded_tile_projected_union_area_v1";
        container.dataset.renderedMapPlanId = renderDiagnostics.activePlanId;
        container.dataset.renderedPlanContentHash = renderDiagnostics.activePlanContentHash;
        container.dataset.renderedGeometryHash = renderDiagnostics.activeGeometryHash;
        container.dataset.renderedMarkerIds = JSON.stringify(renderDiagnostics.activeMarkerIds);
        emitStatus({
          state: renderDiagnostics.gapCount > 0 ? "ready_with_gaps" : "ready",
          code: renderDiagnostics.gapCount > 0
            ? "geographic_map_v2_has_route_gaps"
            : "geographic_map_v2_ready",
          gaps: resolvedGaps.descriptions,
          basemapOverviewOnly: basemap.overview,
          canvasViewportRatio,
          mapReadyAt,
          coverageMethod: "loaded_tile_projected_union_area_v1",
          ...readiness,
          ...renderDiagnostics,
        });
        return true;
      };
      const removeWindowResize = observeMapSize(nextMap, renderGeneration, validateReady);
      hideVisualCanvas(nextMap);
      nextMap.once("load", () => {
        if (renderGeneration !== generation || map !== nextMap) return;
        addInterestAreaLayers(nextMap, currentInterest, interestMassLevel);
        addRouteLayers(nextMap, routes, selectedRouteLegId, selectedDay, activePlanId);
        addDraftRouteLayers(nextMap, checkedDraftRoutes);
        addPointLayers(
          nextMap, stops, anchors, resolvedGaps.markers, provisionalPin, selectedDay,
        );
        nextMap.addSource("v2-marine-routes", {type:"geojson",data:marineRoutes});
        nextMap.addLayer({id:"v2-marine-routes",type:"line",source:"v2-marine-routes",
          paint:{"line-color":"#7445b0","line-width":4,"line-dasharray":[2,2]}});
        nextMap.addSource("v2-offshore-stops", {type:"geojson",data:offshoreStops});
        nextMap.addLayer({id:"v2-offshore-stops",type:"circle",source:"v2-offshore-stops",
          paint:{"circle-radius":8,"circle-color":"#ffffff","circle-stroke-color":"#7445b0","circle-stroke-width":3}});
        nextMap.addLayer({id:"v2-offshore-labels",type:"symbol",source:"v2-offshore-stops",
          layout:{"text-field":"Scorpion Anchorage · Channel Islands","text-font":["Noto Sans Regular"],"text-size":12,"text-anchor":"top","text-offset":[0,1],"text-max-width":18},
          paint:{"text-color":"#553078","text-halo-color":"#ffffff","text-halo-width":2}});
        addPoiCandidateLayers(nextMap, candidatePoints, selectedCandidateId);
        addInterestPointLayers(nextMap, currentInterest, selectedInterestId, onInterestSelect);
        customMarkers = addDraftWaypointMarkers({
          map: nextMap,
          waypoints: draftWaypoints,
          interactionMode,
          selectedWaypointId,
          onCustomWaypointSelect,
          onCustomWaypointDrag,
        });
        const interestHitLayers = currentInterest ? ["interest-pois", "interest-required", "interest-poi-labels", "interest-selected-label"]
          .filter(id => nextMap.getLayer(id)) : [];
        const hitsInterestPoint = event => currentInterest && nextMap.queryRenderedFeatures(event.point, {
          layers: interestHitLayers,
        }).length > 0;
        ["v2-route-active", "v2-route-original"].forEach((layerId) => {
          nextMap.on("click", layerId, (event) => { if (!hitsInterestPoint(event)) exactRouteLegSelection(event, onRouteLegSelect); });
        });
        ["v2-stops-active", "v2-stops-original"].forEach((layerId) => {
          nextMap.on("click", layerId, (event) => { if (!hitsInterestPoint(event)) exactStopSelection(event, onStopSelect); });
        });
        nextMap.on("click", "v2-route-gap-markers", (event) => { if (!hitsInterestPoint(event)) exactGapSelection(event, onGapSelect); });
        ["v2-poi-candidates", "v2-poi-access-points"]
          .forEach((layerId) => nextMap.on(
            "click", layerId, (event) => { if (!hitsInterestPoint(event)) exactCandidateSelection(event, onCandidateSelect); },
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
              "v2-poi-access-points",
              ...interestHitLayers,
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
          const view = camera();
          cameraListeners.forEach((listener) => listener(view));
        });
        hideVisualCanvas(nextMap);
        nextMap.on("movestart", () => {
          if (renderGeneration !== generation || map !== nextMap) return;
          invalidateMapReadiness(container);
          emitStatus({ state: "loading", code: "map_camera_revalidation_pending" });
        });
        nextMap.on("idle", () => {
          if (renderGeneration !== generation || map !== nextMap) return;
          validateReady();
        });
        nextMap.resize();
        if (!initialCamera) nextMap.fitBounds(fittedBounds, fitBoundsOptions);
      });
      nextMap.on("webglcontextlost", () => {
        if (renderGeneration !== generation || map !== nextMap) return;
        webglContextLost = true;
        invalidateMapReadiness(container);
        emitStatus({ state: "failed", code: "map_webgl_context_lost", textual_fallback: true });
      });
      nextMap.on("webglcontextrestored", () => {
        if (renderGeneration !== generation || map !== nextMap) return;
        webglContextLost = false;
        invalidateMapReadiness(container);
        emitStatus({ state: "loading", code: "map_webgl_context_restoring" });
      });
      nextMap.on("error", () => {
        if (renderGeneration !== generation || map !== nextMap) return;
        invalidateMapReadiness(container);
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

  function camera() {
    if (!map) return null;
    const center = map.getCenter();
    return { center: [center.lng, center.lat], zoom: map.getZoom(),
      bearing: map.getBearing(), pitch: map.getPitch() };
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
    invalidateMapReadiness(container);
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
    camera,
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
