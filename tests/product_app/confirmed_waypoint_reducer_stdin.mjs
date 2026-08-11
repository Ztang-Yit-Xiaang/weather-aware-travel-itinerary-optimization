import { readFileSync } from "node:fs";
import { reduceConfirmedMapDraftOperations } from "../../src/itinerary_system/product_app/static/js/map.js";

const payload = JSON.parse(readFileSync(0, "utf8"));
const reduced = reduceConfirmedMapDraftOperations(payload.operations, payload.day_count);
process.stdout.write(JSON.stringify({
  waypoint_count: reduced.waypoints.length,
  route_leg_count: reduced.routeLegs.length,
  waypoint_id: reduced.waypoints[0]?.waypoint_id || null,
  coordinate: reduced.waypoints[0]?.coordinate || null,
}));
