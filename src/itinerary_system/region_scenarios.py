"""Scenario definitions for city-plus-nature itinerary planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NatureRegionDefinition:
    region_id: str
    name: str
    latitude: float
    longitude: float
    gateway_bases: list[str]
    park_type: str = "nature_region"
    min_days: int = 1
    recommended_days: int = 1
    nature_score: float = 0.75
    scenic_score: float = 0.70
    weather_sensitivity: float = 0.60
    seasonality_risk: float = 0.20
    national_park: bool = False
    state_park: bool = False
    protected_area: bool = True
    scenic_region: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    overnight_base_cities: list[str]
    gateway_routes: dict[tuple[str, str], list[str]]
    nature_regions: list[str] = field(default_factory=list)
    nature_region_definitions: list[NatureRegionDefinition] = field(default_factory=list)
    route_options: list[dict[str, Any]] = field(default_factory=list)
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


def _route_option(label: str, start: str, end: str, sequence: list[str], nature_regions: list[str]) -> dict[str, Any]:
    return {
        "label": label,
        "gateway_start": start,
        "gateway_end": end,
        "sequence": sequence,
        "nature_regions": nature_regions,
    }


CALIFORNIA_COAST_NATURE_REGIONS = [
    NatureRegionDefinition(
        region_id="big_sur",
        name="Big Sur",
        latitude=36.2704,
        longitude=-121.8081,
        gateway_bases=["Monterey", "San Luis Obispo"],
        park_type="coastal_scenic_region",
        nature_score=0.80,
        scenic_score=0.92,
        state_park=True,
        notes="Coastal nature region compatible with the existing CA-1 workflow.",
    ),
    NatureRegionDefinition(
        region_id="central_coast_beaches",
        name="Central Coast Beaches",
        latitude=34.9500,
        longitude=-120.6000,
        gateway_bases=["Santa Barbara", "San Luis Obispo", "Monterey"],
        park_type="coastal_beach_region",
        nature_score=0.68,
        scenic_score=0.82,
        notes="Beach and coastal scenic stops along the existing corridor.",
    ),
    NatureRegionDefinition(
        region_id="santa_monica_mountains",
        name="Santa Monica Mountains",
        latitude=34.1200,
        longitude=-118.7500,
        gateway_bases=["Los Angeles", "Santa Barbara"],
        park_type="national_recreation_area",
        nature_score=0.70,
        scenic_score=0.76,
        protected_area=True,
        notes="Nature-heavy option near Los Angeles without changing the corridor.",
    ),
]


CALIFORNIA_STATEWIDE_NATURE_REGIONS = [
    NatureRegionDefinition(
        region_id="yosemite",
        name="Yosemite National Park",
        latitude=37.8651,
        longitude=-119.5383,
        gateway_bases=["Mariposa", "Oakhurst", "Fresno", "Yosemite Valley"],
        park_type="national_park",
        min_days=1,
        recommended_days=2,
        nature_score=0.98,
        scenic_score=0.98,
        weather_sensitivity=0.72,
        seasonality_risk=0.35,
        national_park=True,
        notes="Sierra Nevada anchor region; typical gateways are Mariposa, Oakhurst, Fresno, and Yosemite Valley.",
    ),
    NatureRegionDefinition(
        region_id="sequoia",
        name="Sequoia National Park",
        latitude=36.4864,
        longitude=-118.5658,
        gateway_bases=["Three Rivers", "Fresno", "Visalia"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.94,
        scenic_score=0.90,
        weather_sensitivity=0.68,
        seasonality_risk=0.30,
        national_park=True,
    ),
    NatureRegionDefinition(
        region_id="kings_canyon",
        name="Kings Canyon National Park",
        latitude=36.8879,
        longitude=-118.5551,
        gateway_bases=["Three Rivers", "Fresno", "Visalia"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.92,
        scenic_score=0.91,
        weather_sensitivity=0.68,
        seasonality_risk=0.32,
        national_park=True,
    ),
    NatureRegionDefinition(
        region_id="joshua_tree",
        name="Joshua Tree National Park",
        latitude=33.8734,
        longitude=-115.9010,
        gateway_bases=["Palm Springs", "Twentynine Palms", "Los Angeles"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.88,
        scenic_score=0.86,
        weather_sensitivity=0.74,
        seasonality_risk=0.42,
        national_park=True,
    ),
    NatureRegionDefinition(
        region_id="death_valley",
        name="Death Valley National Park",
        latitude=36.5323,
        longitude=-116.9325,
        gateway_bases=["Las Vegas", "Lone Pine", "Furnace Creek"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.90,
        scenic_score=0.91,
        weather_sensitivity=0.85,
        seasonality_risk=0.55,
        national_park=True,
        notes="High weather exposure; should be excluded on short or hot-season trips.",
    ),
    NatureRegionDefinition(
        region_id="lake_tahoe",
        name="Lake Tahoe",
        latitude=39.0968,
        longitude=-120.0324,
        gateway_bases=["South Lake Tahoe", "Truckee", "Sacramento"],
        park_type="lake_mountain_region",
        recommended_days=2,
        nature_score=0.86,
        scenic_score=0.90,
        weather_sensitivity=0.66,
        seasonality_risk=0.34,
        protected_area=True,
    ),
    NatureRegionDefinition(
        region_id="point_reyes",
        name="Point Reyes",
        latitude=38.0691,
        longitude=-122.8069,
        gateway_bases=["Point Reyes Station", "San Rafael", "San Francisco"],
        park_type="national_seashore",
        recommended_days=1,
        nature_score=0.82,
        scenic_score=0.88,
        weather_sensitivity=0.64,
        seasonality_risk=0.20,
        protected_area=True,
    ),
    NatureRegionDefinition(
        region_id="pinnacles",
        name="Pinnacles National Park",
        latitude=36.4906,
        longitude=-121.1825,
        gateway_bases=["Soledad", "Hollister", "Monterey"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.84,
        scenic_score=0.78,
        weather_sensitivity=0.70,
        seasonality_risk=0.32,
        national_park=True,
    ),
    NatureRegionDefinition(
        region_id="lassen_volcanic",
        name="Lassen Volcanic National Park",
        latitude=40.4977,
        longitude=-121.4207,
        gateway_bases=["Redding", "Chester", "Red Bluff"],
        park_type="national_park",
        recommended_days=1,
        nature_score=0.88,
        scenic_score=0.86,
        weather_sensitivity=0.72,
        seasonality_risk=0.46,
        national_park=True,
    ),
    NatureRegionDefinition(
        region_id="redwood",
        name="Redwood National and State Parks",
        latitude=41.2132,
        longitude=-124.0046,
        gateway_bases=["Eureka", "Crescent City", "Arcata"],
        park_type="national_and_state_parks",
        recommended_days=2,
        nature_score=0.95,
        scenic_score=0.92,
        weather_sensitivity=0.68,
        seasonality_risk=0.25,
        national_park=True,
        state_park=True,
    ),
    *CALIFORNIA_COAST_NATURE_REGIONS[:1],
]


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
            ("Los Angeles", "Los Angeles"): [
                "Los Angeles",
                "Santa Barbara",
                "San Luis Obispo",
                "Monterey",
                "Los Angeles",
            ],
            ("Los Angeles", "San Francisco"): [
                "Los Angeles",
                "Santa Barbara",
                "San Luis Obispo",
                "Monterey",
                "Santa Cruz",
                "San Francisco",
            ],
            ("San Francisco", "Los Angeles"): [
                "San Francisco",
                "Santa Cruz",
                "Monterey",
                "San Luis Obispo",
                "Santa Barbara",
                "Los Angeles",
            ],
            ("San Francisco", "San Francisco"): [
                "San Francisco",
                "Santa Cruz",
                "Monterey",
                "San Luis Obispo",
                "Santa Barbara",
                "San Francisco",
            ],
        },
        nature_regions=["Big Sur", "Central Coast Beaches", "Santa Monica Mountains"],
        nature_region_definitions=CALIFORNIA_COAST_NATURE_REGIONS,
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
    "california_statewide_nature": ScenarioDefinition(
        scenario_id="california_statewide_nature",
        label="California Statewide Nature",
        overnight_base_cities=[
            "San Francisco",
            "Mariposa",
            "Oakhurst",
            "Fresno",
            "Yosemite Valley",
            "Three Rivers",
            "Visalia",
            "Los Angeles",
            "Palm Springs",
            "Twentynine Palms",
            "Las Vegas",
            "Lone Pine",
            "Furnace Creek",
            "South Lake Tahoe",
            "Truckee",
            "Redding",
            "Chester",
            "Eureka",
            "Crescent City",
            "Monterey",
            "Soledad",
        ],
        gateway_routes={
            ("San Francisco", "Los Angeles"): ["San Francisco", "Mariposa", "Fresno", "Three Rivers", "Los Angeles"],
            ("Los Angeles", "San Francisco"): [
                "Los Angeles",
                "Palm Springs",
                "Furnace Creek",
                "Yosemite Valley",
                "San Francisco",
            ],
            ("San Francisco", "San Francisco"): [
                "San Francisco",
                "South Lake Tahoe",
                "Redding",
                "Eureka",
                "San Francisco",
            ],
        },
        route_options=[
            _route_option(
                "Sierra national parks",
                "San Francisco",
                "Los Angeles",
                ["San Francisco", "Mariposa", "Fresno", "Three Rivers", "Los Angeles"],
                ["Yosemite National Park", "Sequoia National Park", "Kings Canyon National Park"],
            ),
            _route_option(
                "Desert to Sierra parks",
                "Los Angeles",
                "San Francisco",
                [
                    "Los Angeles",
                    "Palm Springs",
                    "Twentynine Palms",
                    "Furnace Creek",
                    "Yosemite Valley",
                    "San Francisco",
                ],
                ["Joshua Tree National Park", "Death Valley National Park", "Yosemite National Park"],
            ),
            _route_option(
                "Tahoe Lassen Redwood loop",
                "San Francisco",
                "San Francisco",
                ["San Francisco", "South Lake Tahoe", "Truckee", "Redding", "Eureka", "Crescent City", "San Francisco"],
                ["Lake Tahoe", "Lassen Volcanic National Park", "Redwood National and State Parks"],
            ),
            _route_option(
                "Central coast plus Pinnacles",
                "San Francisco",
                "Los Angeles",
                ["San Francisco", "Monterey", "Big Sur", "Soledad", "Los Angeles"],
                ["Big Sur", "Pinnacles National Park"],
            ),
        ],
        nature_regions=[region.name for region in CALIFORNIA_STATEWIDE_NATURE_REGIONS],
        nature_region_definitions=CALIFORNIA_STATEWIDE_NATURE_REGIONS,
        pass_through_scenic_stops=[
            "Yosemite Valley",
            "General Sherman Tree",
            "Keys View",
            "Zabriskie Point",
            "Emerald Bay",
            "Point Reyes Lighthouse",
            "Bear Gulch",
            "Bumpass Hell",
            "Avenue of the Giants",
            "Bixby Creek Bridge Viewpoint",
        ],
        curated_seed_pois=[
            {
                "name": region.name,
                "city": region.gateway_bases[0],
                "latitude": region.latitude,
                "longitude": region.longitude,
                "category": region.park_type,
                "park_type": region.park_type,
                "nature_region": region.name,
                "source_list": "curated_nature_region_seed",
                "source_score": 9.5 if region.national_park else 8.7,
            }
            for region in CALIFORNIA_STATEWIDE_NATURE_REGIONS
        ],
        coordinate_fallbacks={
            **CALIFORNIA_COORDS,
            "Mariposa": (37.4849, -119.9663),
            "Oakhurst": (37.3280, -119.6493),
            "Fresno": (36.7378, -119.7871),
            "Yosemite Valley": (37.7456, -119.5936),
            "Three Rivers": (36.4388, -118.9045),
            "Visalia": (36.3302, -119.2921),
            "Palm Springs": (33.8303, -116.5453),
            "Twentynine Palms": (34.1356, -116.0542),
            "Las Vegas": (36.1716, -115.1391),
            "Lone Pine": (36.6060, -118.0629),
            "Furnace Creek": (36.4570, -116.8675),
            "South Lake Tahoe": (38.9399, -119.9772),
            "Truckee": (39.3279, -120.1833),
            "Sacramento": (38.5816, -121.4944),
            "Redding": (40.5865, -122.3917),
            "Chester": (40.3063, -121.2319),
            "Red Bluff": (40.1785, -122.2358),
            "Eureka": (40.8021, -124.1637),
            "Crescent City": (41.7558, -124.2026),
            "Arcata": (40.8665, -124.0828),
            "Point Reyes Station": (38.0691, -122.8069),
            "San Rafael": (37.9735, -122.5311),
            "Soledad": (36.4247, -121.3263),
            "Hollister": (36.8525, -121.4016),
            "Big Sur": (36.2704, -121.8081),
            **{region.name: (region.latitude, region.longitude) for region in CALIFORNIA_STATEWIDE_NATURE_REGIONS},
        },
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
        pass_through_scenic_stops=[
            "Red Rock Canyon",
            "Valley of Fire",
            "Zion Canyon Scenic Drive",
            "Grand Canyon South Rim",
        ],
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


def get_nature_region_definitions(scenario_id: str | None = None) -> list[NatureRegionDefinition]:
    scenario = get_scenario_definition(scenario_id)
    return list(scenario.nature_region_definitions)


def get_route_options(scenario_id: str | None = None) -> list[dict[str, Any]]:
    scenario = get_scenario_definition(scenario_id)
    if scenario.route_options:
        return [dict(option) for option in scenario.route_options]
    return [
        _route_option(
            f"{start} to {end}",
            start,
            end,
            sequence,
            [],
        )
        for (start, end), sequence in scenario.gateway_routes.items()
    ]


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
                "nature_region_count": len(scenario.nature_regions),
                "pass_through_scenic_stops": " | ".join(scenario.pass_through_scenic_stops),
                "curated_seed_poi_count": len(scenario.curated_seed_pois),
                "gateway_route_count": len(scenario.gateway_routes),
                "route_option_count": len(get_route_options(scenario.scenario_id)),
                "why_not_selected_reason_examples": "coast_only_scenario_restriction | excessive_inland_detour | insufficient_trip_days | daily_drive_limit_exceeded | weather_risk_too_high | budget_or_time_conflict | lower_interest_fit_than_selected_regions",
            }
        )
    return rows
