import { createGeographicMapController } from "./map-controller-v2.js?v=20260810-stability5";

let activeController = null;

const CONFIRMED_PARAMETER_KEYS = new Set([
  "schema_version", "snap_preview_id", "waypoint_id", "name", "day", "role", "duration",
  "raw_coordinate", "snapped_coordinate", "selected_access_point", "insertion",
  "affected_route_legs",
]);
const ACCESS_KEYS = new Set([
  "access_point_id", "access_type", "coordinate", "source", "road_validated",
  "access_confidence", "evidence_refs",
]);
const INSERTION_KEYS = new Set([
  "route_leg_id", "predecessor_id", "successor_id", "travel_mode",
]);
const ROUTE_LEG_KEYS = new Set([
  "route_leg_id", "origin_id", "destination_id", "travel_mode", "validation_status",
  "geometry", "distance_m", "duration_s", "provider", "routing_status", "geometry_source",
  "distance_source", "duration_source", "road_validated", "fallback_used", "query_hash",
  "evidence_refs", "retrieved_at", "snap_distance_origin_m", "snap_distance_destination_m",
]);
const CUSTOM_ROLES = new Set(["attraction", "activity", "meal", "rest_stop", "scenic_stop"]);
const WAYPOINT_ID_PATTERN = /^waypoint_[0-9a-f]{32}$/;
const OPERATION_ID_PATTERN = /^operation_[0-9a-f]{32}$/;
const SNAP_ID_PATTERN = /^snap_[0-9a-f]{32}$/;
const ACCESS_ID_PATTERN = /^access_[0-9a-f]{32}$/;
const QUERY_HASH_PATTERN = /^[0-9a-f]{64}$/;
const FROZEN_ROUTE_QUERY_HASH_PATTERN = /^[0-9a-f]{16}$/;
const POI_RESPONSE_KEYS = new Set([
  "schema_version", "session_id", "session_revision", "context", "catalog", "routing",
  "candidates",
]);
const POI_CONTEXT_KEYS = new Set([
  "kind", "day", "route_leg_id", "replacement_target_id", "predecessor_id", "successor_id",
  "baseline_route_leg_ids", "baseline_travel_minutes", "baseline_travel_distance_m",
]);
const POI_CANDIDATE_KEYS = new Set([
  "candidate_id", "place", "selected_access_point", "sources", "burden", "precheck",
  "route_evidence_refs", "registered_replacement",
]);
const POI_PLACE_KEYS = new Set([
  "place_id", "name", "place_categories", "display_coordinate", "description", "official_url",
  "informational_urls", "source_refs", "source_freshness", "opening_hours_evidence_ref",
  "recommended_visit_minutes", "weather_suitability",
]);
const POI_ACCESS_KEYS = new Set([
  "access_point_id", "access_type", "coordinate", "source_ref", "road_validated",
  "access_confidence", "evidence_refs",
]);
const POI_BURDEN_KEYS = new Set([
  "schema_version", "context_kind", "candidate_id", "place_id", "predecessor_id", "successor_id",
  "replacement_target_id", "baseline_route_leg_ids", "baseline_travel_minutes",
  "baseline_travel_distance_m",
  "geographic_distance_m", "predecessor_to_candidate_minutes", "candidate_to_successor_minutes",
  "predecessor_to_successor_minutes", "marginal_travel_minutes", "marginal_travel_distance_m",
  "visit_minutes", "parking_minutes", "walking_minutes", "waiting_minutes",
  "total_insertion_minutes", "nearby", "route_near", "likely_feasible",
  "evaluated_feasible", "ranking_eligible", "recommended", "candidate_state",
  "evaluator_rank", "evaluator_evidence_refs", "blocking_codes", "evidence_refs",
]);
const POI_PRECHECK_KEYS = new Set([
  "predicted_arrival", "open_at_arrival", "status", "evidence_refs",
]);
const POI_STATES = new Set([
  "unavailable", "nearby", "route_near", "likely_feasible", "evaluated_feasible",
  "recommended",
]);
const POI_CATALOG_KEYS = new Set(["catalog_id", "catalog_sha256", "generated_at"]);
const POI_ROUTING_KEYS = new Set([
  "matrix_id", "context_snapshot_id", "source_bundle_id", "source_content_sha256",
  "road_validated_only", "fallback_allowed",
]);

function exactKeys(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value);
  return keys.length === expected.size && keys.every((key) => expected.has(key));
}

function finiteCoordinate(value) {
  if (!exactKeys(value, new Set(["latitude", "longitude"]))) return null;
  const latitude = value.latitude;
  const longitude = value.longitude;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)
      || latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
  return { latitude, longitude };
}

function nullableFinite(value) {
  return value === null || (typeof value === "number" && Number.isFinite(value) && value >= 0);
}

function nullableSignedFinite(value) {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function stringArray(value, maximum = 64) {
  return Array.isArray(value) && value.length <= maximum
    && value.every((item) => typeof item === "string" && item.length > 0 && item.length <= 256);
}

function sameStringArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function nullableHttpUrl(value) {
  if (value === null) return true;
  if (typeof value !== "string" || value.length > 2048) return false;
  try {
    const parsed = new URL(value);
    return ["http:", "https:"].includes(parsed.protocol)
      && !parsed.username && !parsed.password && Boolean(parsed.hostname);
  } catch {
    return false;
  }
}

function validPoiAccessPoint(value, routing, routeEvidenceRefs, place, sources) {
  const candidateRouteHashes = new Set(
    Array.isArray(routeEvidenceRefs)
      ? routeEvidenceRefs
        .filter((reference) => ["predecessor_candidate", "candidate_successor"]
          .includes(reference?.role))
        .map((reference) => reference.query_hash)
      : [],
  );
  const accessEvidence = new Set(Array.isArray(value?.evidence_refs) ? value.evidence_refs : []);
  const sourceIds = new Set(Array.isArray(sources) ? sources.map((source) => source?.source_id) : []);
  return exactKeys(value, POI_ACCESS_KEYS)
    && typeof value.access_point_id === "string" && value.access_point_id.length > 0
    && typeof value.access_type === "string" && value.access_type.length > 0
    && finiteCoordinate(value.coordinate) !== null
    && value.source_ref === routing.matrix_id
    && place.source_refs.includes(value.source_ref)
    && sourceIds.has(value.source_ref)
    && value.road_validated === true
    && typeof value.access_confidence === "string" && value.access_confidence.length > 0
    && stringArray(value.evidence_refs)
    && value.evidence_refs.length === 2
    && accessEvidence.size === 2
    && candidateRouteHashes.size === 2
    && [...accessEvidence].every((reference) => candidateRouteHashes.has(reference));
}

function validPoiBurden(value, candidate, context) {
  if (!exactKeys(value, POI_BURDEN_KEYS)
      || value.schema_version !== "product-candidate-burden-v1"
      || value.candidate_id !== candidate.candidate_id
      || value.place_id !== candidate.place.place_id
      || value.context_kind !== context.kind
      || value.replacement_target_id !== context.replacement_target_id
      || value.predecessor_id !== context.predecessor_id
      || value.successor_id !== context.successor_id
      || !sameStringArray(value.baseline_route_leg_ids, context.baseline_route_leg_ids)
      || value.baseline_travel_minutes !== context.baseline_travel_minutes
      || value.baseline_travel_distance_m !== context.baseline_travel_distance_m
      || !POI_STATES.has(value.candidate_state)
      || !["nearby", "route_near", "likely_feasible", "evaluated_feasible", "ranking_eligible", "recommended"]
        .every((key) => typeof value[key] === "boolean")
      || ![
        "geographic_distance_m", "predecessor_to_candidate_minutes",
        "candidate_to_successor_minutes", "predecessor_to_successor_minutes",
        "visit_minutes", "parking_minutes", "walking_minutes", "waiting_minutes",
      ].every((key) => nullableFinite(value[key]))
      || !["marginal_travel_minutes", "marginal_travel_distance_m", "total_insertion_minutes"]
        .every((key) => nullableSignedFinite(value[key]))
      || !(value.evaluator_rank === null
        || (Number.isInteger(value.evaluator_rank) && value.evaluator_rank > 0))
      || !stringArray(value.evaluator_evidence_refs)
      || !stringArray(value.blocking_codes) || !stringArray(value.evidence_refs)
      || new Set(value.evaluator_evidence_refs).size !== value.evaluator_evidence_refs.length
      || !value.evaluator_evidence_refs.every((reference) => value.evidence_refs.includes(reference))
      || (value.evaluated_feasible && value.evaluator_evidence_refs.length === 0)) return false;
  const expectedState = value.recommended ? "recommended"
    : value.evaluated_feasible ? "evaluated_feasible"
      : value.likely_feasible ? "likely_feasible"
        : value.route_near ? "route_near"
          : value.nearby ? "nearby" : "unavailable";
  return value.candidate_state === expectedState
    && (!value.likely_feasible || value.route_near)
    && (!value.ranking_eligible || value.evaluated_feasible)
    && (!value.recommended || value.ranking_eligible)
    && (value.ranking_eligible
      ? Number.isInteger(value.evaluator_rank) && value.evaluator_rank > 0
      : value.evaluator_rank === null);
}

function validPoiPrecheck(value) {
  if (!exactKeys(value, POI_PRECHECK_KEYS)
      || !(value.predicted_arrival === null || typeof value.predicted_arrival === "string")
      || !(value.open_at_arrival === null || typeof value.open_at_arrival === "boolean")
      || !["passed", "failed", "unavailable"].includes(value.status)
      || !stringArray(value.evidence_refs)) return false;
  return value.status === "unavailable"
    ? value.predicted_arrival === null && value.open_at_arrival === null
      && value.evidence_refs.length === 0
    : value.evidence_refs.length > 0;
}

function validRegisteredReplacement(value, candidate, context) {
  if (value === null) return true;
  return exactKeys(value, new Set(["draft_type", "target_stop_id", "candidate_id"]))
    && context.kind === "replacement"
    && value.draft_type === "replace_nearby"
    && value.target_stop_id === context.replacement_target_id
    && value.candidate_id === candidate.candidate_id
    && value.candidate_id === candidate.place.place_id;
}

function validRouteEvidenceRefs(values, context, burden) {
  const expectedRoles = context.kind === "replacement"
    ? new Set([
      "predecessor_candidate", "candidate_successor", "predecessor_target", "target_successor",
    ])
    : new Set(["predecessor_candidate", "candidate_successor", "predecessor_successor"]);
  const baselineByRole = context.kind === "replacement"
    ? new Map([
      ["predecessor_target", context.baseline_route_leg_ids[0]],
      ["target_successor", context.baseline_route_leg_ids[1]],
    ])
    : new Map([["predecessor_successor", context.baseline_route_leg_ids[0]]]);
  const burdenEvidence = new Set(Array.isArray(burden?.evidence_refs) ? burden.evidence_refs : []);
  if (!Array.isArray(values) || values.length !== expectedRoles.size) return false;
  const roles = new Set();
  const routeLegs = new Set();
  const queryHashes = new Set();
  return values.every((reference) => {
    if (!exactKeys(reference, new Set(["role", "route_leg_id", "query_hash"]))
        || !expectedRoles.has(reference.role) || roles.has(reference.role)
        || typeof reference.route_leg_id !== "string" || !reference.route_leg_id
        || routeLegs.has(reference.route_leg_id)
        || !FROZEN_ROUTE_QUERY_HASH_PATTERN.test(reference.query_hash || "")
        || queryHashes.has(reference.query_hash)
        || (baselineByRole.has(reference.role)
          && reference.route_leg_id !== baselineByRole.get(reference.role))
        || !burdenEvidence.has(reference.route_leg_id)
        || !burdenEvidence.has(reference.query_hash)) return false;
    roles.add(reference.role);
    routeLegs.add(reference.route_leg_id);
    queryHashes.add(reference.query_hash);
    return true;
  }) && roles.size === expectedRoles.size;
}

function validPoiContextBaseline(context) {
  if (!stringArray(context.baseline_route_leg_ids, 2)
      || new Set(context.baseline_route_leg_ids).size !== context.baseline_route_leg_ids.length) {
    return false;
  }
  return context.kind === "insertion"
    ? context.baseline_route_leg_ids.length === 1
      && context.baseline_route_leg_ids[0] === context.route_leg_id
    : context.baseline_route_leg_ids.length === 2
      && context.baseline_route_leg_ids.includes(context.route_leg_id);
}

/**
 * Validate the exact server-owned route-aware POI response. Any mismatch fails
 * closed so the browser never turns discovery coordinates into feasibility.
 */
export function normalizePoiCandidateResponse(payload, expected) {
  const context = payload?.context;
  if (!exactKeys(payload, POI_RESPONSE_KEYS)
      || payload.schema_version !== "product-poi-candidates-v1"
      || payload.session_id !== expected.sessionId
      || payload.session_revision !== expected.revision
      || !exactKeys(context, POI_CONTEXT_KEYS)
      || !["insertion", "replacement"].includes(context.kind)
      || context.day !== expected.day
      || context.route_leg_id !== expected.routeLegId
      || context.replacement_target_id !== (expected.replacementTargetId || null)
      || typeof context.predecessor_id !== "string" || !context.predecessor_id
      || typeof context.successor_id !== "string" || !context.successor_id
      || !validPoiContextBaseline(context)
      || !nullableFinite(context.baseline_travel_minutes)
      || !nullableFinite(context.baseline_travel_distance_m)
      || !exactKeys(payload.catalog, POI_CATALOG_KEYS)
      || typeof payload.catalog.catalog_id !== "string" || !payload.catalog.catalog_id
      || !QUERY_HASH_PATTERN.test(payload.catalog.catalog_sha256 || "")
      || typeof payload.catalog.generated_at !== "string"
      || !exactKeys(payload.routing, POI_ROUTING_KEYS)
      || ["matrix_id", "context_snapshot_id", "source_bundle_id"]
        .some((key) => typeof payload.routing[key] !== "string" || !payload.routing[key])
      || !QUERY_HASH_PATTERN.test(payload.routing.source_content_sha256 || "")
      || payload.routing.road_validated_only !== true
      || payload.routing.fallback_allowed !== false
      || !Array.isArray(payload.candidates) || payload.candidates.length > 10) {
    throw new Error("poi_candidate_response_invalid");
  }
  const candidateIds = new Set();
  const candidates = payload.candidates.map((candidate) => {
    const place = candidate?.place;
    if (!exactKeys(candidate, POI_CANDIDATE_KEYS)
        || typeof candidate.candidate_id !== "string" || !candidate.candidate_id
        || candidateIds.has(candidate.candidate_id)
        || !exactKeys(place, POI_PLACE_KEYS)
        || typeof place.place_id !== "string" || !place.place_id
        || candidate.candidate_id !== place.place_id
        || typeof place.name !== "string" || !place.name.trim()
        || !stringArray(place.place_categories, 32)
        || finiteCoordinate(place.display_coordinate) === null
        || !(place.description === null || typeof place.description === "string")
        || !nullableHttpUrl(place.official_url)
        || !Array.isArray(place.informational_urls)
        || !place.informational_urls.every(nullableHttpUrl)
        || !stringArray(place.source_refs, 32)
        || !(place.source_freshness === null || typeof place.source_freshness === "string")
        || !(place.opening_hours_evidence_ref === null
          || typeof place.opening_hours_evidence_ref === "string")
        || !(place.recommended_visit_minutes === null
          || (Number.isInteger(place.recommended_visit_minutes)
            && place.recommended_visit_minutes > 0))
        || !(place.weather_suitability === null
          || (typeof place.weather_suitability === "number"
            && Number.isFinite(place.weather_suitability)))
        || !Array.isArray(candidate.sources)
        || !candidate.sources.every((source) => exactKeys(source, new Set([
          "source_id", "source_type", "source_url", "retrieved_at",
        ])) && typeof source.source_id === "string" && typeof source.source_type === "string"
          && nullableHttpUrl(source.source_url)
          && (source.retrieved_at === null || typeof source.retrieved_at === "string"))
        || !validPoiBurden(candidate.burden, candidate, context)
        || !validRouteEvidenceRefs(candidate.route_evidence_refs, context, candidate.burden)
        || !validPoiAccessPoint(candidate.selected_access_point, payload.routing,
          candidate.route_evidence_refs, place, candidate.sources)
        || !validPoiPrecheck(candidate.precheck)
        || candidate.burden.likely_feasible
          !== (candidate.burden.route_near && candidate.precheck.status === "passed")
        || !candidate.precheck.evidence_refs.every(
          (reference) => candidate.burden.evidence_refs.includes(reference),
        )
        || !candidate.burden.evidence_refs.includes(payload.routing.matrix_id)
        || !candidate.burden.evidence_refs.includes(payload.routing.context_snapshot_id)
        || !validRegisteredReplacement(candidate.registered_replacement, candidate, context)) {
      throw new Error("poi_candidate_response_invalid");
    }
    candidateIds.add(candidate.candidate_id);
    return Object.freeze({ ...candidate });
  });
  return Object.freeze({
    ...payload,
    context: Object.freeze({ ...context }),
    candidates: Object.freeze(candidates),
  });
}

function sameCoordinate(left, right) {
  return left.latitude === right.latitude && left.longitude === right.longitude;
}

function validTimestamp(value) {
  return typeof value === "string"
    && /(Z|[+-]\d{2}:\d{2})$/.test(value)
    && Number.isFinite(Date.parse(value));
}

function validDuration(role, duration) {
  if (role === "route_waypoint") return duration === null;
  if (!CUSTOM_ROLES.has(role)
      || !exactKeys(duration, new Set([
        "mode", "preferred_minutes", "minimum_minutes", "maximum_minutes",
      ]))) return false;
  const { mode, preferred_minutes: preferred, minimum_minutes: minimum, maximum_minutes: maximum } = duration;
  const minutes = (value) => Number.isInteger(value) && value >= 15 && value <= 480;
  if (mode === "exact") return minutes(preferred) && minimum === preferred && maximum === preferred;
  if (mode === "preferred") return minutes(preferred) && minimum === null && maximum === null;
  if (mode === "minimum") return preferred === null && minutes(minimum) && maximum === null;
  if (mode === "maximum") return preferred === null && minimum === null && minutes(maximum);
  return mode === "range" && preferred === null && minutes(minimum) && minutes(maximum)
    && minimum <= maximum;
}

function sameDuration(left, right) {
  if (left === null || right === null) return left === right;
  return left.mode === right.mode
    && left.preferred_minutes === right.preferred_minutes
    && left.minimum_minutes === right.minimum_minutes
    && left.maximum_minutes === right.maximum_minutes;
}

function validLineString(geometry) {
  return exactKeys(geometry, new Set(["type", "coordinates"]))
    && geometry.type === "LineString"
    && Array.isArray(geometry.coordinates)
    && geometry.coordinates.length >= 2
    && geometry.coordinates.every((point) => Array.isArray(point)
      && point.length === 2
      && Number.isFinite(point[0]) && point[0] >= -180 && point[0] <= 180
      && Number.isFinite(point[1]) && point[1] >= -90 && point[1] <= 90);
}

function validRouteLeg(leg) {
  if (!exactKeys(leg, ROUTE_LEG_KEYS)
      || !QUERY_HASH_PATTERN.test(leg.query_hash || "")
      || leg.route_leg_id !== `preview_leg_${leg.query_hash.slice(0, 16)}`
      || typeof leg.origin_id !== "string" || !leg.origin_id || leg.origin_id.length > 128
      || typeof leg.destination_id !== "string" || !leg.destination_id || leg.destination_id.length > 128
      || leg.travel_mode !== "driving"
      || leg.validation_status !== "road_validated"
      || leg.provider !== "runtime_osrm"
      || leg.routing_status !== "osrm_route_validated"
      || leg.geometry_source !== "runtime_osrm_geojson"
      || leg.distance_source !== "runtime_osrm_route"
      || leg.duration_source !== "runtime_osrm_route"
      || leg.road_validated !== true || leg.fallback_used !== false
      || !Number.isFinite(leg.distance_m) || leg.distance_m <= 0
      || !Number.isFinite(leg.duration_s) || leg.duration_s <= 0
      || !Number.isFinite(leg.snap_distance_origin_m) || leg.snap_distance_origin_m < 0
      || !Number.isFinite(leg.snap_distance_destination_m) || leg.snap_distance_destination_m < 0
      || !validLineString(leg.geometry)
      || !validTimestamp(leg.retrieved_at)) return false;
  const evidence = `route_query:${leg.query_hash}`;
  return Array.isArray(leg.evidence_refs)
    && leg.evidence_refs.length === 1
    && leg.evidence_refs[0] === evidence;
}

function validAccessPoint(access, snapped, expectedEvidence) {
  if (!exactKeys(access, ACCESS_KEYS)
      || !ACCESS_ID_PATTERN.test(access.access_point_id || "")
      || access.access_type !== "road_snap"
      || access.source !== "runtime_osrm"
      || access.road_validated !== true
      || access.access_confidence !== "road_snap_only"
      || !Array.isArray(access.evidence_refs)
      || access.evidence_refs.length !== expectedEvidence.length
      || access.evidence_refs.some((value, index) => value !== expectedEvidence[index])) return false;
  const coordinate = finiteCoordinate(access.coordinate);
  return coordinate !== null && sameCoordinate(coordinate, snapped);
}

function sameInsertion(left, right) {
  return left.route_leg_id === right.route_leg_id
    && left.predecessor_id === right.predecessor_id
    && left.successor_id === right.successor_id
    && left.travel_mode === right.travel_mode;
}

function validateConfirmedOperation(operation, dayCount) {
  if (!operation || operation.source !== "confirmed_map_interaction"
      || !OPERATION_ID_PATTERN.test(operation.operation_id || "")
      || !validTimestamp(operation.created_at)
      || !["add_custom_waypoint", "add_route_waypoint", "relocate_custom_waypoint"].includes(operation.type)
      || !WAYPOINT_ID_PATTERN.test(operation.target || "")
      || !Array.isArray(operation.evidence_refs)) return null;
  const parameters = operation.parameters;
  if (!exactKeys(parameters, CONFIRMED_PARAMETER_KEYS)
      || parameters.schema_version !== "confirmed-map-operation-v1"
      || parameters.waypoint_id !== operation.target
      || !WAYPOINT_ID_PATTERN.test(parameters.waypoint_id)
      || !SNAP_ID_PATTERN.test(parameters.snap_preview_id || "")
      || typeof parameters.name !== "string"
      || parameters.name !== parameters.name.trim()
      || parameters.name.length < 1 || parameters.name.length > 120
      || !Number.isInteger(parameters.day) || parameters.day < 1 || parameters.day > dayCount
      || !validDuration(parameters.role, parameters.duration)) return null;
  const raw = finiteCoordinate(parameters.raw_coordinate);
  const snapped = finiteCoordinate(parameters.snapped_coordinate);
  const insertion = parameters.insertion;
  if (raw === null || snapped === null
      || !exactKeys(insertion, INSERTION_KEYS)
      || insertion.travel_mode !== "driving"
      || [insertion.route_leg_id, insertion.predecessor_id, insertion.successor_id]
        .some((value) => typeof value !== "string" || !value || value.length > 128)
      || insertion.predecessor_id === parameters.waypoint_id
      || insertion.successor_id === parameters.waypoint_id
      || insertion.predecessor_id === insertion.successor_id) return null;
  const legs = parameters.affected_route_legs;
  if (!Array.isArray(legs) || legs.length !== 2 || !legs.every(validRouteLeg)) return null;
  const [incoming, outgoing] = legs;
  if (incoming.query_hash === outgoing.query_hash
      || incoming.route_leg_id === outgoing.route_leg_id
      || incoming.origin_id !== insertion.predecessor_id
      || incoming.destination_id !== parameters.waypoint_id
      || outgoing.origin_id !== parameters.waypoint_id
      || outgoing.destination_id !== insertion.successor_id) return null;
  const accessPoint = [snapped.longitude, snapped.latitude];
  const incomingEnd = incoming.geometry.coordinates.at(-1);
  const outgoingStart = outgoing.geometry.coordinates[0];
  if (incomingEnd[0] !== accessPoint[0] || incomingEnd[1] !== accessPoint[1]
      || outgoingStart[0] !== accessPoint[0] || outgoingStart[1] !== accessPoint[1]) return null;
  const expectedEvidence = legs.map((leg) => `route_query:${leg.query_hash}`);
  if (operation.evidence_refs.length !== 2
      || operation.evidence_refs.some((value, index) => value !== expectedEvidence[index])
      || !validAccessPoint(parameters.selected_access_point, snapped, expectedEvidence)) return null;
  return { parameters, routeLegs: legs };
}

/**
 * Reduce only exact, server-confirmed map operations into a visual draft view.
 * Invalid records are ignored whole; no partial route or marker is displayed.
 */
export function reduceConfirmedMapDraftOperations(operations, dayCount) {
  if (!Array.isArray(operations) || !Number.isInteger(dayCount) || dayCount < 1) {
    return Object.freeze({ waypoints: Object.freeze([]), routeLegs: Object.freeze([]) });
  }
  const byWaypoint = new Map();
  const claimedAdds = new Set();
  operations.forEach((operation) => {
    const target = typeof operation?.target === "string" ? operation.target : null;
    const isAdd = ["add_custom_waypoint", "add_route_waypoint"].includes(operation?.type);
    if (isAdd && target && WAYPOINT_ID_PATTERN.test(target)) {
      if (claimedAdds.has(target)) return;
      claimedAdds.add(target);
    }
    const validated = validateConfirmedOperation(operation, dayCount);
    if (validated === null) return;
    const { parameters, routeLegs } = validated;
    const previous = byWaypoint.get(parameters.waypoint_id);
    if (operation.type === "relocate_custom_waypoint") {
      if (!previous
          || parameters.name !== previous.name
          || parameters.role !== previous.role
          || !sameDuration(parameters.duration, previous.duration)
          || parameters.day !== previous.day
          || !sameInsertion(parameters.insertion, previous.insertion)) return;
    } else if (previous || (operation.type === "add_route_waypoint") !== (parameters.role === "route_waypoint")) {
      return;
    }
    const kind = parameters.role === "route_waypoint" ? "route_waypoint" : "custom_stop";
    if (previous && previous.kind !== kind) return;
    byWaypoint.set(parameters.waypoint_id, Object.freeze({
      waypoint_id: parameters.waypoint_id,
      name: parameters.name,
      role: parameters.role,
      day: parameters.day,
      duration: parameters.duration,
      insertion: parameters.insertion,
      kind,
      coordinate: Object.freeze([
        parameters.snapped_coordinate.longitude,
        parameters.snapped_coordinate.latitude,
      ]),
      route_legs: Object.freeze(routeLegs.map((leg) => Object.freeze({
        ...leg,
        waypoint_id: parameters.waypoint_id,
      }))),
    }));
  });
  const waypoints = Object.freeze([...byWaypoint.values()]);
  return Object.freeze({
    waypoints,
    routeLegs: Object.freeze(waypoints.flatMap((waypoint) => waypoint.route_legs)),
  });
}

function activePlanId(geography, requestedPlanId) {
  const plans = Array.isArray(geography?.plans) ? geography.plans : [];
  if (requestedPlanId && plans.some((plan) => plan.plan_id === requestedPlanId)) {
    return requestedPlanId;
  }
  return plans.find((plan) => plan.role === "registered_repair")?.plan_id
    || plans.find((plan) => plan.role === "original")?.plan_id
    || null;
}

export function disposeGeographicMap() {
  activeController?.dispose();
  activeController = null;
}

/**
 * Compatibility adapter for the product workspace.
 *
 * The instance-owned V2 controller is the only owner of MapLibre lifecycle,
 * visual layers, and raw map events. This module only translates application
 * selection/edit state into that controller's frozen contract.
 */
export async function renderGeographicMap({
  container,
  geography,
  mapConfig,
  selectedAlternativeId,
  selectedDay = null,
  selectedRouteLegId = null,
  interactionMode = "select",
  exploratoryPin = null,
  draftWaypoints = [],
  draftRouteLegs = [],
  selectedWaypointId = null,
  poiCandidates = [],
  selectedCandidateId = null,
  showPoiCandidates = false,
  onSelectStop = () => {},
  onSelectRouteLeg = () => {},
  onSelectCandidate = () => {},
  onSelectCustomWaypoint = () => {},
  onCustomWaypointDrag = () => {},
  onEmptyMapClick = () => {},
  onStatus = () => {},
}) {
  disposeGeographicMap();
  const planId = activePlanId(geography, selectedAlternativeId);
  if (!planId) {
    onStatus({ state: "failed", code: "active_plan_required" });
    return null;
  }

  const controller = createGeographicMapController({
    container,
    mapConfig,
    paneId: "workspace-map",
    onStopSelect: onSelectStop,
    onRouteLegSelect: onSelectRouteLeg,
    onCandidateSelect: onSelectCandidate,
    onCustomWaypointSelect: onSelectCustomWaypoint,
    onCustomWaypointDrag,
    onGapSelect: (gap) => onStatus({
      state: "ready_with_gaps",
      code: "route_gap_selected",
      gapCount: 1,
      selectedGap: gap,
    }),
    onEmptyMapClick,
    onTextFallback: () => {},
    onStatus,
  });
  activeController = controller;
  return controller.render({
    geography,
    activePlanId: planId,
    selectedRouteLegId,
    selectedDay,
    interactionMode,
    exploratoryPin,
    draftWaypoints,
    draftRouteLegs,
    selectedWaypointId,
    poiCandidates,
    selectedCandidateId,
    showPoiCandidates,
  });
}
