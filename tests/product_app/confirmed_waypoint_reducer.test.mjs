import assert from "node:assert/strict";
import { reduceConfirmedMapDraftOperations } from "../../src/itinerary_system/product_app/static/js/map.js";

const WAYPOINT_ID = `waypoint_${"a".repeat(32)}`;
const INSERTION = Object.freeze({
  route_leg_id: "route_leg_parent_day3_2_3",
  predecessor_id: "parent_stop_2",
  successor_id: "parent_stop_3",
  travel_mode: "driving",
});
const DURATION = Object.freeze({
  mode: "exact",
  preferred_minutes: 60,
  minimum_minutes: 60,
  maximum_minutes: 60,
});

function routeLeg({ hashCharacter, originId, destinationId, coordinates }) {
  const queryHash = hashCharacter.repeat(64);
  return {
    route_leg_id: `preview_leg_${queryHash.slice(0, 16)}`,
    origin_id: originId,
    destination_id: destinationId,
    travel_mode: "driving",
    validation_status: "road_validated",
    geometry: { type: "LineString", coordinates },
    distance_m: 1250,
    duration_s: 180,
    provider: "runtime_osrm",
    routing_status: "osrm_route_validated",
    geometry_source: "runtime_osrm_geojson",
    distance_source: "runtime_osrm_route",
    duration_source: "runtime_osrm_route",
    road_validated: true,
    fallback_used: false,
    query_hash: queryHash,
    evidence_refs: [`route_query:${queryHash}`],
    retrieved_at: "2026-08-08T12:00:00Z",
    snap_distance_origin_m: 0,
    snap_distance_destination_m: 8,
  };
}

function operation({
  type = "add_custom_waypoint",
  operationCharacter = "c",
  snapCharacter = "b",
  accessCharacter = "f",
  access = [-118.25, 34.05],
  hashCharacters = ["1", "2"],
} = {}) {
  const legs = [
    routeLeg({
      hashCharacter: hashCharacters[0],
      originId: INSERTION.predecessor_id,
      destinationId: WAYPOINT_ID,
      coordinates: [[-118.3, 34.0], access],
    }),
    routeLeg({
      hashCharacter: hashCharacters[1],
      originId: WAYPOINT_ID,
      destinationId: INSERTION.successor_id,
      coordinates: [access, [-118.2, 34.1]],
    }),
  ];
  return {
    operation_id: `operation_${operationCharacter.repeat(32)}`,
    type,
    target: WAYPOINT_ID,
    parameters: {
      schema_version: "confirmed-map-operation-v1",
      snap_preview_id: `snap_${snapCharacter.repeat(32)}`,
      waypoint_id: WAYPOINT_ID,
      name: "Harbor rest stop",
      day: 3,
      role: "rest_stop",
      duration: { ...DURATION },
      raw_coordinate: { latitude: access[1] + 0.0001, longitude: access[0] - 0.0001 },
      snapped_coordinate: { latitude: access[1], longitude: access[0] },
      selected_access_point: {
        access_point_id: `access_${accessCharacter.repeat(32)}`,
        access_type: "road_snap",
        coordinate: { latitude: access[1], longitude: access[0] },
        source: "runtime_osrm",
        road_validated: true,
        access_confidence: "road_snap_only",
        evidence_refs: legs.map((leg) => `route_query:${leg.query_hash}`),
      },
      insertion: { ...INSERTION },
      affected_route_legs: legs,
    },
    source: "confirmed_map_interaction",
    evidence_refs: legs.map((leg) => `route_query:${leg.query_hash}`),
    created_at: "2026-08-08T12:00:01Z",
  };
}

function clone(value) {
  return structuredClone(value);
}

function rejectMutation(label, mutate) {
  const candidate = clone(operation());
  mutate(candidate);
  const result = reduceConfirmedMapDraftOperations([candidate], 7);
  assert.equal(result.waypoints.length, 0, label);
  assert.equal(result.routeLegs.length, 0, `${label}: route evidence must fail closed too`);
}

const added = operation();
const relocated = operation({
  type: "relocate_custom_waypoint",
  operationCharacter: "d",
  snapCharacter: "e",
  accessCharacter: "9",
  access: [-118.24, 34.06],
  hashCharacters: ["3", "4"],
});
const valid = reduceConfirmedMapDraftOperations([added, relocated], 7);
assert.equal(valid.waypoints.length, 1);
assert.deepEqual(valid.waypoints[0].coordinate, [-118.24, 34.06]);
assert.equal(valid.routeLegs.length, 2);
assert.ok(valid.routeLegs.every((leg) => leg.road_validated === true));

const routeWaypoint = operation({ operationCharacter: "6", snapCharacter: "7", accessCharacter: "8" });
routeWaypoint.type = "add_route_waypoint";
routeWaypoint.parameters.role = "route_waypoint";
routeWaypoint.parameters.duration = null;
const validRouteWaypoint = reduceConfirmedMapDraftOperations([routeWaypoint], 7);
assert.equal(validRouteWaypoint.waypoints[0].kind, "route_waypoint");

rejectMutation("untrusted source", (candidate) => { candidate.source = "map"; });
rejectMutation("invalid operation id", (candidate) => { candidate.operation_id = "operation_bad"; });
rejectMutation("target mismatch", (candidate) => { candidate.target = `waypoint_${"b".repeat(32)}`; });
rejectMutation("unknown parameter", (candidate) => { candidate.parameters.browser_claim = true; });
rejectMutation("untrimmed name", (candidate) => { candidate.parameters.name = " forged "; });
rejectMutation("unknown role", (candidate) => { candidate.parameters.role = "hotel"; });
rejectMutation("invalid preferred duration", (candidate) => {
  candidate.parameters.duration = {
    mode: "preferred", preferred_minutes: 60, minimum_minutes: 15, maximum_minutes: null,
  };
});
rejectMutation("day outside trip", (candidate) => { candidate.parameters.day = 8; });
rejectMutation("one route leg", (candidate) => {
  candidate.parameters.affected_route_legs.pop();
  candidate.evidence_refs.pop();
});
rejectMutation("unlinked route identity", (candidate) => {
  candidate.parameters.affected_route_legs[0].destination_id = "other_waypoint";
});
rejectMutation("route junction does not equal access", (candidate) => {
  candidate.parameters.affected_route_legs[1].geometry.coordinates[0] = [-118.23, 34.07];
});
rejectMutation("access point does not equal snapped coordinate", (candidate) => {
  candidate.parameters.selected_access_point.coordinate.longitude = -118.23;
});
rejectMutation("forged route provenance", (candidate) => {
  candidate.parameters.affected_route_legs[0].provider = "browser";
});
rejectMutation("operation evidence does not bind both legs", (candidate) => {
  candidate.evidence_refs[1] = `route_query:${"5".repeat(64)}`;
});

const changedMetadata = clone(relocated);
changedMetadata.parameters.name = "Changed in relocation";
const preserved = reduceConfirmedMapDraftOperations([added, changedMetadata], 7);
assert.deepEqual(preserved.waypoints[0].coordinate, [-118.25, 34.05]);
assert.equal(preserved.waypoints[0].name, "Harbor rest stop");

const malformedRelocation = clone(relocated);
malformedRelocation.parameters.affected_route_legs.pop();
malformedRelocation.evidence_refs.pop();
const preservedAfterMalformedRelocation = reduceConfirmedMapDraftOperations(
  [added, malformedRelocation], 7,
);
assert.deepEqual(preservedAfterMalformedRelocation.waypoints[0].coordinate, [-118.25, 34.05]);

const relocationWithoutAdd = reduceConfirmedMapDraftOperations([relocated], 7);
assert.equal(relocationWithoutAdd.waypoints.length, 0);

console.log("confirmed waypoint reducer adversarial cases passed");
