"""A port-based day excursion; marine connectors are explicitly schematic."""

from copy import deepcopy

CHANNEL_ISLANDS = "channel_islands"
PORT = (-119.265667411237, 34.242102483393)
SCORPION = (-119.5562112, 34.0493266)
CHANNEL_URLS = {
    "channel_islands_nps": "https://www.nps.gov/chis/planyourvisit/directions.htm",
    "channel_islands_operator": "https://www.islandpackers.com/santa-cruz-island/",
}
EXCURSION = {
    "schema_version": "island-excursion-v1",
    "excursion_id": "ventura_scorpion_round_trip",
    "departure_name": "Ventura Harbor · Island Packers, 1691 Spinnaker Dr",
    "departure_coordinates": list(PORT),
    "destination_name": "Channel Islands · Santa Cruz Island / Scorpion Anchorage",
    "destination_coordinates": list(SCORPION),
    "travel_mode": "passenger_ferry",
    "geometry_kind": "schematic_not_navigation",
    "round_trip": True,
    "reservation_status": "not_verified",
    "schedule_status": "not_verified",
    "planning_minutes": 480,
    "time_allowance": {
        "check_in_minutes": 45,
        "boat_each_way_minutes": 90,
        "island_minutes": 210,
        "contingency_minutes": 45,
    },
    "source_urls": list(CHANNEL_URLS.values()),
    "coordinate_sources": [
        "US Census current address geocoder: 1691 Spinnaker Drive, Ventura, CA 93001",
        "OpenStreetMap pier way 1110130352 center; NPS Scorpion Anchorage Pier place page",
    ],
}
NOTICE = (
    "Channel Islands day trip: drive to Ventura Harbor, then return by passenger ferry from "
    "Santa Cruz Island / Scorpion Anchorage. The purple dashed sea line is schematic, not "
    "navigation or an OSRM road. A 480-minute planning allowance includes 45-minute check-in, "
    "90 minutes each way by boat, 210 minutes on island and 45 minutes contingency. These are "
    "planning allowances, not a verified sailing schedule. Ferry tickets, landing conditions "
    "and date-specific availability are not verified; confirm them with Island Packers."
)


def island_geography(plan):
    """Derive both marine collections from the immutable port excursion."""
    routes, destinations = [], []
    for stop in plan.get("selected_stops", ()):
        if (stop.get("poi_id") or stop.get("stop_id")) != CHANNEL_ISLANDS:
            continue
        if stop.get("excursion") != EXCURSION:
            raise ValueError("island_excursion_invalid")
        shared = {
            "plan_id": plan["plan_id"], "content_hash": plan["content_hash"],
            "day": stop["day"], "source_stop_id": CHANNEL_ISLANDS,
            "travel_mode": "passenger_ferry", "road_validated": False,
            "geometry_source": "schematic_port_to_island", "round_trip": True,
            "schedule_verified": False, "reservation_verified": False,
            "name": EXCURSION["destination_name"],
        }
        routes.append({
            "type": "Feature", "id": plan["plan_id"] + ":marine:ventura_scorpion",
            "geometry": {"type": "LineString", "coordinates": [list(PORT), list(SCORPION)]},
            "properties": {**shared, "label": "Ferry round trip · schematic / timetable unverified"},
        })
        destinations.append({
            "type": "Feature", "id": plan["plan_id"] + ":island:scorpion",
            "geometry": {"type": "Point", "coordinates": list(SCORPION)},
            "properties": deepcopy(shared),
        })
    return {
        "marine_routes": {"type": "FeatureCollection", "features": routes},
        "offshore_stops": {"type": "FeatureCollection", "features": destinations},
    }
