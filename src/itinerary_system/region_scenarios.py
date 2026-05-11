"""Scenario definitions for city-plus-nature itinerary planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    overnight_base_cities: list[str]
    gateway_routes: dict[tuple[str, str], list[str]]
    nature_regions: list[str] = field(default_factory=list)
    pass_through_scenic_stops: list[str] = field(default_factory=list)
    curated_seed_pois: list[dict[str, Any]] = field(default_factory=list)
    coordinate_fallbacks: dict[str, tuple[float, float]] = field(default_factory=dict)


CALIFORNIA_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}


SCENARIOS: dict[str, ScenarioDefinition] = {
    "california_coast": ScenarioDefinition(
        scenario_id="california_coast",
        label="California Coast",
        overnight_base_cities=[
            "San Francisco",
            "Santa Cruz",
            "Monterey",
            "San Luis Obispo",
            "Santa Barbara",
            "Los Angeles",
        ],
        gateway_routes={
            ("Los Angeles", "Los Angeles"): ["Los Angeles", "Santa Barbara", "San Luis Obispo", "Monterey", "Los Angeles"],
            ("Los Angeles", "San Francisco"): ["Los Angeles", "Santa Barbara", "San Luis Obispo", "Monterey", "Santa Cruz", "San Francisco"],
            ("San Francisco", "Los Angeles"): ["San Francisco", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "Los Angeles"],
            ("San Francisco", "San Francisco"): ["San Francisco", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "San Francisco"],
        },
        nature_regions=["Big Sur", "Central Coast Beaches", "Santa Monica Mountains"],
        pass_through_scenic_stops=["Bixby Creek Bridge Viewpoint", "Pfeiffer Big Sur State Park", "Malibu Beaches"],
        curated_seed_pois=[
            {
                "name": "Pfeiffer Big Sur State Park",
                "city": "Monterey",
                "latitude": 36.2524,
                "longitude": -121.7879,
                "category": "state_park",
                "park_type": "state_park",
                "nature_region": "Big Sur",
                "source_list": "curated_nature_seed",
                "source_score": 8.8,
            },
            {
                "name": "Point Lobos State Natural Reserve",
                "city": "Monterey",
                "latitude": 36.5159,
                "longitude": -121.9425,
                "category": "nature_reserve",
                "park_type": "state_reserve",
                "nature_region": "Big Sur",
                "source_list": "curated_nature_seed",
                "source_score": 8.9,
            },
        ],
        coordinate_fallbacks=CALIFORNIA_COORDS,
    ),
    "las_vegas_national_parks": ScenarioDefinition(
        scenario_id="las_vegas_national_parks",
        label="Las Vegas + National Parks",
        overnight_base_cities=["Las Vegas", "Springdale", "Page", "Flagstaff"],
        gateway_routes={
            ("Las Vegas", "Las Vegas"): ["Las Vegas", "Springdale", "Page", "Flagstaff", "Las Vegas"],
            ("Las Vegas", "Flagstaff"): ["Las Vegas", "Springdale", "Page", "Flagstaff"],
            ("Flagstaff", "Las Vegas"): ["Flagstaff", "Page", "Springdale", "Las Vegas"],
        },
        nature_regions=["Red Rock Canyon", "Valley of Fire", "Zion", "Bryce Canyon", "Grand Canyon", "Death Valley"],
        pass_through_scenic_stops=["Red Rock Canyon", "Valley of Fire", "Zion Canyon Scenic Drive", "Grand Canyon South Rim"],
        curated_seed_pois=[
            {
                "name": "Red Rock Canyon National Conservation Area",
                "city": "Las Vegas",
                "latitude": 36.1355,
                "longitude": -115.4270,
                "category": "protected_area",
                "park_type": "national_conservation_area",
                "nature_region": "Red Rock Canyon",
                "source_list": "curated_nature_seed",
                "source_score": 9.0,
            },
            {
                "name": "Zion National Park",
                "city": "Springdale",
                "latitude": 37.2982,
                "longitude": -113.0263,
                "category": "national_park",
                "park_type": "national_park",
                "nature_region": "Zion",
                "source_list": "curated_nature_seed",
                "source_score": 9.7,
            },
            {
                "name": "Bryce Canyon National Park",
                "city": "Springdale",
                "latitude": 37.5930,
                "longitude": -112.1871,
                "category": "national_park",
                "park_type": "national_park",
                "nature_region": "Bryce Canyon",
                "source_list": "curated_nature_seed",
                "source_score": 9.4,
            },
            {
                "name": "Grand Canyon South Rim",
                "city": "Flagstaff",
                "latitude": 36.0579,
                "longitude": -112.1431,
                "category": "national_park",
                "park_type": "national_park",
                "nature_region": "Grand Canyon",
                "source_list": "curated_nature_seed",
                "source_score": 9.8,
            },
        ],
        coordinate_fallbacks={
            "Las Vegas": (36.1716, -115.1391),
            "Springdale": (37.1889, -112.9986),
            "Page": (36.9147, -111.4558),
            "Flagstaff": (35.1983, -111.6513),
        },
    ),
    "new_york_city_plus_nature": ScenarioDefinition(
        scenario_id="new_york_city_plus_nature",
        label="New York City + Nature",
        overnight_base_cities=["New York City", "Kingston", "Lake Placid"],
        gateway_routes={
            ("New York City", "New York City"): ["New York City", "Kingston", "Lake Placid", "New York City"],
            ("New York City", "Lake Placid"): ["New York City", "Kingston", "Lake Placid"],
            ("Lake Placid", "New York City"): ["Lake Placid", "Kingston", "New York City"],
        },
        nature_regions=["Central Park", "Hudson Valley", "Catskills", "Adirondacks"],
        pass_through_scenic_stops=["Central Park", "Mohonk Preserve", "Adirondack High Peaks"],
        curated_seed_pois=[
            {
                "name": "Central Park",
                "city": "New York City",
                "latitude": 40.7829,
                "longitude": -73.9654,
                "category": "urban_park",
                "park_type": "urban_park",
                "nature_region": "Central Park",
                "source_list": "curated_nature_seed",
                "source_score": 9.1,
            },
            {
                "name": "Mohonk Preserve",
                "city": "Kingston",
                "latitude": 41.7714,
                "longitude": -74.1562,
                "category": "protected_area",
                "park_type": "nature_preserve",
                "nature_region": "Hudson Valley",
                "source_list": "curated_nature_seed",
                "source_score": 8.7,
            },
        ],
        coordinate_fallbacks={
            "New York City": (40.7128, -74.0060),
            "Kingston": (41.9270, -73.9974),
            "Lake Placid": (44.2795, -73.9799),
        },
    ),
}


def get_scenario_definition(scenario_id: str | None = None) -> ScenarioDefinition:
    return SCENARIOS.get(str(scenario_id or "california_coast"), SCENARIOS["california_coast"])


def all_scenario_coordinates() -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for scenario in SCENARIOS.values():
        coords.update(scenario.coordinate_fallbacks)
    return coords


def scenario_summary_rows() -> list[dict[str, Any]]:
    rows = []
    for scenario in SCENARIOS.values():
        rows.append(
            {
                "scenario": scenario.scenario_id,
                "label": scenario.label,
                "overnight_base_cities": " | ".join(scenario.overnight_base_cities),
                "nature_regions": " | ".join(scenario.nature_regions),
                "pass_through_scenic_stops": " | ".join(scenario.pass_through_scenic_stops),
                "curated_seed_poi_count": len(scenario.curated_seed_pois),
                "gateway_route_count": len(scenario.gateway_routes),
            }
        )
    return rows

