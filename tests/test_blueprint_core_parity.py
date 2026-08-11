from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from itinerary_system import blueprint_core
from itinerary_system._legacy import import_legacy_module


@pytest.fixture(scope="module")
def legacy_blueprint():
    return import_legacy_module("blueprint_trip_map")


def test_constants_and_sequence_helpers_match_legacy(legacy_blueprint):
    assert blueprint_core.CITY_COORDS == legacy_blueprint.CITY_COORDS
    assert len(blueprint_core.CITY_COORDS) == 48
    assert blueprint_core._coord_for_place(blueprint_core.STANFORD_WAYPOINT_NAME) == [37.4275, -122.1697]
    assert blueprint_core._profile_config("balanced", stops_per_day=3) == {
        "label": "Balanced",
        "stops_per_day": 3,
        "show": True,
        "social_quota": 2,
        "social_weight": 1.15,
        "must_go_bonus": 0.65,
        "anchor_distance_weight": 0.18,
        "hotel_distance_weight": 0.1,
        "hotel_proximity_weight": 0.4,
        "hotel_rating_weight": 0.45,
        "route_mode": "balanced utility/social/compactness route",
    }

    profile_cases = [
        ((), {}),
        (("relaxed",), {}),
        (("unknown-profile",), {}),
        (("balanced",), {"stops_per_day": 5}),
    ]
    for args, kwargs in profile_cases:
        assert blueprint_core._profile_config(*args, **kwargs) == legacy_blueprint._profile_config(*args, **kwargs)

    day_cases = [
        {"Santa Barbara": 2, "Los Angeles": 1},
        "{'Santa Barbara': 2, 'Los Angeles': 1}",
        None,
        float("nan"),
        "not-a-dictionary",
    ]
    for value in day_cases:
        assert blueprint_core._coerce_days_by_city(value) == legacy_blueprint._coerce_days_by_city(value)

    assert blueprint_core._unique_in_order(["A", "B", "A", "C"]) == legacy_blueprint._unique_in_order(
        ["A", "B", "A", "C"]
    )

    trips = [
        {"city_sequence": ["San Francisco", "Monterey", "Los Angeles"]},
        {
            "overnight_bases": ["San Francisco", "Los Angeles"],
            "pass_through_cities": ["Santa Cruz", "Monterey"],
        },
        {"days_by_city": {"Santa Barbara": 2, "Los Angeles": 1}},
        {},
    ]
    for trip in trips:
        assert blueprint_core._trip_sequence_with_pass_through(trip) == legacy_blueprint._trip_sequence_with_pass_through(
            trip
        )

    sequence = ["San Francisco", "Monterey", "Santa Barbara", "Los Angeles"]
    for origin, destination in [
        ("San Francisco", "Los Angeles"),
        ("Los Angeles", "Monterey"),
        ("Unknown", "Los Angeles"),
    ]:
        assert blueprint_core._sequence_between(
            sequence, origin, destination
        ) == legacy_blueprint._sequence_between(sequence, origin, destination)


def test_catalog_normalization_matches_with_missing_and_invalid_fields(legacy_blueprint):
    raw = pd.DataFrame(
        [
            {
                "name": "Coastal Walk",
                "categories": "outdoors",
                "latitude": "34.4208",
                "longitude": "-119.6982",
                "utility": "4.5",
                "social_must_go": True,
                "weather_sensitivity": None,
            },
            {
                "name": None,
                "categories": "invalid",
                "latitude": "not-a-number",
                "longitude": "-119.7",
            },
        ]
    )

    expected = legacy_blueprint._normalize_catalog_columns(raw, "Santa Barbara", "fixture")
    actual = blueprint_core._normalize_catalog_columns(raw, "Santa Barbara", "fixture")

    pd.testing.assert_frame_equal(actual, expected)
    assert actual["name"].tolist() == ["Coastal Walk"]
    assert actual["category"].tolist() == ["outdoors"]
    assert actual["source_score"].tolist() == [4.5]
    assert actual["weather_sensitivity"].tolist() == [0.0]
    assert len(actual.columns) == 39
    pd.testing.assert_frame_equal(
        blueprint_core._normalize_catalog_columns(pd.DataFrame(), "Santa Barbara", "fixture"),
        legacy_blueprint._normalize_catalog_columns(pd.DataFrame(), "Santa Barbara", "fixture"),
    )


def test_city_catalog_and_hotel_fallbacks_match_legacy(tmp_path, legacy_blueprint):
    top100 = pd.DataFrame(
        [
            {
                "city": "Santa Barbara",
                "name": "Mission",
                "latitude": 34.438,
                "longitude": -119.713,
                "utility": 4.2,
            }
        ]
    )
    hotels = pd.DataFrame(
        [
            {
                "name": "Hotel A",
                "latitude": 34.421,
                "longitude": -119.699,
                "stars": 4.5,
                "nightly_price": 220.0,
                "type": "hotel",
            },
            {
                "name": "Hotel B",
                "latitude": 34.425,
                "longitude": -119.705,
                "stars": 4.0,
                "nightly_price": 180.0,
                "type": "motel",
            },
        ]
    )
    context = {
        "OUTPUT_DIR": tmp_path,
        "CITY": "Santa Barbara",
        "top100_with_waiting_time": top100,
        "hotels_df": hotels,
    }

    for city in ["Santa Barbara", "San Diego"]:
        pd.testing.assert_frame_equal(
            blueprint_core._city_poi_catalog(context, city),
            legacy_blueprint._city_poi_catalog(context, city),
        )
        pd.testing.assert_frame_equal(
            blueprint_core._city_hotel_catalog(context, city),
            legacy_blueprint._city_hotel_catalog(context, city),
        )


def test_hotel_and_compact_stop_selection_match_legacy(legacy_blueprint):
    raw_pois = pd.DataFrame(
        [
            {"name": "A", "latitude": 34.421, "longitude": -119.699, "utility": 5.0},
            {"name": "B", "latitude": 34.423, "longitude": -119.701, "utility": 4.0},
            {"name": "C", "latitude": 34.425, "longitude": -119.703, "utility": 3.0},
            {"name": "D", "latitude": 34.450, "longitude": -119.750, "utility": 2.0},
        ]
    )
    poi_catalog_old = legacy_blueprint._normalize_catalog_columns(raw_pois, "Santa Barbara", "fixture")
    poi_catalog_new = blueprint_core._normalize_catalog_columns(raw_pois, "Santa Barbara", "fixture")
    hotel_catalog = pd.DataFrame(
        [
            {
                "name": "Hotel A",
                "latitude": 34.4208,
                "longitude": -119.6982,
                "rating_score": 4.5,
                "nightly_price": 220.0,
            },
            {
                "name": "Hotel B",
                "latitude": 34.4300,
                "longitude": -119.7100,
                "rating_score": 4.5,
                "nightly_price": 180.0,
            },
        ]
    )
    profile_old = legacy_blueprint._profile_config("balanced", stops_per_day=3)
    profile_new = blueprint_core._profile_config("balanced", stops_per_day=3)

    selected_old = legacy_blueprint._select_hotel_for_city_plan(
        hotel_catalog, poi_catalog_old, {"D"}, stops_per_day=3, profile_config=profile_old
    )
    selected_new = blueprint_core._select_hotel_for_city_plan(
        hotel_catalog, poi_catalog_new, {"D"}, stops_per_day=3, profile_config=profile_new
    )
    pd.testing.assert_series_equal(selected_new, selected_old)
    assert selected_new["name"] == "Hotel A"

    stops_old = legacy_blueprint._select_compact_day_stops(
        poi_catalog_old, selected_old, {"D"}, stops_per_day=3, profile_config=profile_old
    )
    stops_new = blueprint_core._select_compact_day_stops(
        poi_catalog_new, selected_new, {"D"}, stops_per_day=3, profile_config=profile_new
    )
    pd.testing.assert_frame_equal(stops_new, stops_old)
    assert stops_new["name"].tolist() == ["A", "B", "C"]


def test_transition_timing_matches_cached_and_geometric_fallback(tmp_path, legacy_blueprint):
    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    pd.DataFrame(
        [
            {
                "route_layer": "Scenic CA-1",
                "from": "Santa Barbara",
                "to": "Los Angeles",
                "estimated_drive_minutes": 123.4,
            }
        ]
    ).to_csv(cached_dir / "production_intercity_legs.csv", index=False)

    cached_context = {"OUTPUT_DIR": cached_dir}
    assert blueprint_core._transition_drive_minutes(
        cached_context, "Santa Barbara", "Los Angeles", []
    ) == legacy_blueprint._transition_drive_minutes(cached_context, "Santa Barbara", "Los Angeles", [])

    fallback_context = {"OUTPUT_DIR": tmp_path / "missing"}
    actual_minutes, actual_source = blueprint_core._transition_drive_minutes(
        fallback_context, "San Francisco", "Los Angeles", ["Monterey"]
    )
    expected_minutes, expected_source = legacy_blueprint._transition_drive_minutes(
        fallback_context, "San Francisco", "Los Angeles", ["Monterey"]
    )
    assert actual_minutes == pytest.approx(expected_minutes)
    assert actual_source == expected_source
    assert actual_minutes == pytest.approx(704.6)
    assert actual_source == "curated_waypoint_speed_limit_proxy"


def test_stanford_transition_stop_matches_output_and_mutation(tmp_path, legacy_blueprint):
    context = {"OUTPUT_DIR": tmp_path, "CITY": "Santa Barbara"}
    legacy_used: dict[str, set[str]] = {}
    core_used: dict[str, set[str]] = {}

    expected = legacy_blueprint._stanford_transition_stop(deepcopy(context), legacy_used)
    actual = blueprint_core._stanford_transition_stop(deepcopy(context), core_used)

    pd.testing.assert_frame_equal(actual, expected)
    assert core_used == legacy_used
    assert actual["name"].tolist() == [blueprint_core.STANFORD_WAYPOINT_NAME]
    assert core_used == {"Palo Alto / Stanford": {blueprint_core.STANFORD_WAYPOINT_NAME}}

    pd.testing.assert_frame_equal(
        blueprint_core._stanford_transition_stop(deepcopy(context), core_used),
        legacy_blueprint._stanford_transition_stop(deepcopy(context), legacy_used),
    )
    assert core_used == legacy_used
