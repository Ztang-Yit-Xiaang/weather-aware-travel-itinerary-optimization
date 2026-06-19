import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.bandit_candidate_selector import (
    build_bandit_arms,
    route_search_strategies_from_config,
    run_bandit_gurobi_stress_benchmark,
)
from itinerary_system._legacy import import_legacy_module
from itinerary_system.artifact_metadata import artifact_metadata_matches, write_artifact_metadata
from itinerary_system.budget import estimate_budget_range
from itinerary_system.config import load_trip_config
from itinerary_system.data_enrichment import canonical_poi_columns
from itinerary_system.diversity import mmr_select_candidates, submodular_diversity
from itinerary_system.experiment_runner import _ensure_required_anchor_stops, _write_route_anchor_audit
from itinerary_system.hierarchical_gurobi import candidate_plans, solve_hierarchical_trip_with_gurobi
from itinerary_system.map_exporter import export_map_artifacts
from itinerary_system.multi_objective_route import solve_multi_objective_route
from itinerary_system.nature_catalog import (
    build_interest_bar_preview,
    compute_interest_adjusted_values,
    ensure_nature_columns,
    load_nps_or_curated_nature_pois,
    write_interest_catalog_artifacts,
)
from itinerary_system.nature_site_routes import (
    apply_nature_site_route_score_bonus,
    build_nature_site_route_artifacts,
    parse_overpass_nature_site_routes,
)
from itinerary_system.region_scenarios import (
    get_nature_region_definitions,
    get_scenario_definition,
    scenario_enrichment_city_universe,
)
from itinerary_system.request_schema import normalize_interest_weights, preset_to_interest_weights
from itinerary_system.route_gurobi_oracle import solve_enriched_route_with_gurobi
from itinerary_system.utility_model import apply_utility_models

CONFIG_PATH = REPO_ROOT / "configs" / "default_trip_config.yaml"


@contextmanager
def temporary_directory():
    base_dir = REPO_ROOT / "results" / "outputs" / "pytest_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def city_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "San Francisco", "city_value_score": 0.95, "data_uncertainty": 0.10},
            {"city": "Santa Cruz", "city_value_score": 0.70, "data_uncertainty": 0.25},
            {"city": "Monterey", "city_value_score": 0.76, "data_uncertainty": 0.20},
            {"city": "San Luis Obispo", "city_value_score": 0.66, "data_uncertainty": 0.30},
            {"city": "Santa Barbara", "city_value_score": 0.88, "data_uncertainty": 0.15},
            {"city": "Los Angeles", "city_value_score": 0.98, "data_uncertainty": 0.12},
        ]
    )


class ConfigurableItinerarySystemTests(unittest.TestCase):
    def test_default_config_exposes_user_editable_parameters(self):
        config = load_trip_config(CONFIG_PATH)

        self.assertEqual(config.trip_days, 7)
        self.assertEqual(config.user_budget, 2000.0)
        self.assertFalse(config.get("enrichment", "use_yelp_live_api"))
        self.assertTrue(config.get("enrichment", "use_local_yelp_dataset"))
        self.assertTrue(config.get("enrichment", "use_open_meteo"))
        self.assertTrue(config.get("enrichment", "use_osrm"))
        self.assertIn("high_must_go", config.bandit_arms)
        self.assertIn("nature_heavy", config.bandit_arms)
        self.assertIn("national_park_priority", config.bandit_arms)
        self.assertTrue((config.to_frame()["parameter"] == "max_detour_minutes").any())
        self.assertEqual(config.get("utility", "method"), "bayesian_ucb")
        self.assertTrue(config.get("multi_objective", "use_epsilon_constraints"))
        self.assertEqual(config.get("diversity", "method"), "submodular_sqrt")
        self.assertIn("data_uncertainty", canonical_poi_columns())
        self.assertTrue(config.get("interest", "enabled"))
        self.assertEqual(config.get("interest", "mode"), "balanced_interest")
        self.assertIn("interest_adjusted_value", canonical_poi_columns())
        self.assertTrue(config.get("nature", "enabled"))
        self.assertEqual(config.get("nature", "nps_api_key_env"), "NPS_API_KEY")
        self.assertEqual(config.get("map_dashboard", "default_mode"), "research")
        self.assertIn("customer", config.get("map_dashboard", "export_modes"))
        self.assertTrue(config.get("map_dashboard", "customer_hide_research_controls"))
        self.assertEqual(config.get("map_export", "mode"), "both")

    def test_interest_mode_nature_heavy_resolves_preset_weights(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={"interest": {"enabled": True, "mode": "nature_heavy"}},
        )

        self.assertTrue(config.get("interest", "enabled"))
        self.assertEqual(config.get("interest", "mode"), "nature_heavy")
        self.assertAlmostEqual(config.get("interest", "weights")["nature"], 0.55)
        self.assertAlmostEqual(sum(config.get("interest", "weights").values()), 1.0)

    def test_trip_interest_profile_alias_enables_and_resolves_preset(self):
        config = load_trip_config(CONFIG_PATH, overrides={"trip": {"interest_profile": "nature_heavy"}})

        self.assertTrue(config.get("interest", "enabled"))
        self.assertEqual(config.get("trip", "interest_profile"), "nature_heavy")
        self.assertAlmostEqual(config.get("interest", "weights")["nature"], 0.55)

    def test_interest_bar_weights_normalize_and_presets_resolve(self):
        weights = normalize_interest_weights({"nature": 30, "city": 40, "culture": 20, "history": 10})

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["nature"], 0.30)
        self.assertAlmostEqual(weights["city"], 0.40)

        preset = preset_to_interest_weights("nature_heavy")
        self.assertAlmostEqual(sum(preset.values()), 1.0)
        self.assertGreater(preset["nature"], preset["city"])

    def test_small_route_solver_uses_open_path_start_and_end_depots(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "optimization": {
                    "max_pois_per_day": 2,
                    "candidate_pool_size": 2,
                    "gurobi_time_limit_seconds": 5,
                },
                "multi_objective": {
                    "use_epsilon_constraints": False,
                    "secondary_travel_penalty": 0.0,
                },
            },
        )
        candidates = pd.DataFrame(
            [
                {
                    "name": "Near start",
                    "city": "Test City",
                    "latitude": 0.0,
                    "longitude": 0.1,
                    "category": "park",
                    "final_poi_value": 10.0,
                    "social_score": 0.0,
                    "weather_risk": 0.0,
                },
                {
                    "name": "Near end",
                    "city": "Test City",
                    "latitude": 0.0,
                    "longitude": 0.2,
                    "category": "park",
                    "final_poi_value": 10.0,
                    "social_score": 0.0,
                    "weather_risk": 0.0,
                },
            ]
        )

        result = solve_multi_objective_route(
            candidates,
            config,
            start_depot=(0.0, 0.0),
            end_depot=(0.0, 0.3),
            candidate_size=2,
        )

        self.assertEqual(result["selected_pois"], ["Near start", "Near end"])
        self.assertLess(result["total_travel_minutes"], 70.0)

    def test_missing_nature_columns_are_filled_safely(self):
        frame = pd.DataFrame(
            [
                {
                    "name": "Old Museum",
                    "city": "Santa Barbara",
                    "latitude": 34.4,
                    "longitude": -119.7,
                    "category": "museum",
                    "final_poi_value": 0.5,
                }
            ]
        )

        output = ensure_nature_columns(frame)

        self.assertIn("interest_adjusted_value", output.columns)
        self.assertFalse(bool(output["is_nature"].iloc[0]))
        self.assertEqual(float(output["nature_score"].iloc[0]), 0.0)

    def test_interest_disabled_keeps_final_value_unchanged(self):
        config = load_trip_config(CONFIG_PATH, overrides={"interest": {"enabled": False}})
        frame = pd.DataFrame(
            [
                {
                    "name": "Coastal Park",
                    "city": "Monterey",
                    "latitude": 36.5,
                    "longitude": -121.9,
                    "category": "state park scenic viewpoint",
                    "source_list": "curated_nature_seed",
                    "final_poi_value": 0.7,
                }
            ]
        )

        output = compute_interest_adjusted_values(frame, config)

        self.assertAlmostEqual(float(output["interest_adjusted_value"].iloc[0]), 0.7)
        self.assertAlmostEqual(float(output["interest_delta"].iloc[0]), 0.0)

    def test_nature_heavy_interest_boosts_nature_poi(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "interest": {
                    "enabled": True,
                    "weights": {"nature": 0.70, "city": 0.10, "culture": 0.10, "history": 0.10},
                }
            },
        )
        frame = pd.DataFrame(
            [
                {
                    "name": "National Park",
                    "city": "Springdale",
                    "latitude": 37.2,
                    "longitude": -113.0,
                    "category": "national park hiking scenic",
                    "source_list": "curated",
                    "final_poi_value": 0.5,
                    "weather_risk": 0.1,
                },
                {
                    "name": "Downtown Market",
                    "city": "Las Vegas",
                    "latitude": 36.1,
                    "longitude": -115.1,
                    "category": "downtown market",
                    "source_list": "curated",
                    "final_poi_value": 0.5,
                    "weather_risk": 0.1,
                },
            ]
        )

        output = compute_interest_adjusted_values(frame, config)
        park_value = output.loc[output["name"].eq("National Park"), "interest_adjusted_value"].iloc[0]
        city_value = output.loc[output["name"].eq("Downtown Market"), "interest_adjusted_value"].iloc[0]

        self.assertGreater(park_value, city_value)

    def test_missing_nps_api_key_uses_curated_fallback(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "las_vegas_national_parks"},
                "nature": {"use_nps_api": True, "use_curated_nature_fallback": True},
            },
        )
        with mock.patch.dict("os.environ", {}, clear=True), temporary_directory() as tmpdir:
            pois, audit = load_nps_or_curated_nature_pois(Path(tmpdir) / "outputs", config)

        self.assertFalse(pois.empty)
        self.assertFalse(audit.empty)
        self.assertTrue(pois["is_nature"].astype(bool).any())
        self.assertEqual(audit["status"].iloc[0], "missing_NPS_API_KEY_curated_fallback")

    def test_nps_disabled_writes_honest_fallback_audit(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "california_statewide_nature"},
                "nature": {"use_nps_api": False, "use_curated_nature_fallback": True},
            },
        )
        with temporary_directory() as tmpdir:
            pois, audit = load_nps_or_curated_nature_pois(Path(tmpdir) / "outputs", config)

        self.assertFalse(pois.empty)
        self.assertEqual(audit["status"].iloc[0], "nps_disabled_curated_fallback")

    def test_overpass_nature_site_routes_parse_line_geometry(self):
        config = load_trip_config(CONFIG_PATH)
        site = pd.Series(
            {
                "_site_id": "yosemite",
                "name": "Yosemite National Park",
                "city": "Mariposa",
                "nature_region": "Yosemite National Park",
                "latitude": 37.8651,
                "longitude": -119.5383,
            }
        )
        elements = [
            {
                "type": "way",
                "tags": {"name": "Valley Loop Trail", "route": "hiking", "website": "https://example.test/trail"},
                "geometry": [
                    {"lat": 37.748, "lon": -119.596},
                    {"lat": 37.752, "lon": -119.592},
                    {"lat": 37.754, "lon": -119.588},
                ],
            }
        ]

        routes, points = parse_overpass_nature_site_routes(elements, site, "overpass_route_cache", config)

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route_type"], "short_hike")
        self.assertGreater(routes[0]["distance_km"], 0)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0]["route_id"], routes[0]["route_id"])

    def test_curated_internal_routes_write_artifacts_when_live_apis_disabled(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "california_statewide_nature"},
                "enrichment": {"run_live_apis": False},
                "nature_detail_routes": {"enabled": True, "use_curated_fallback": True},
            },
        )
        enriched = pd.DataFrame(
            [
                {
                    "name": "Yosemite National Park",
                    "city": "Mariposa",
                    "nature_region": "Yosemite National Park",
                    "latitude": 37.8651,
                    "longitude": -119.5383,
                    "category": "national_park",
                    "is_nature": True,
                    "is_national_park": True,
                    "nature_score": 0.9,
                    "final_poi_value": 0.8,
                }
            ]
        )
        with temporary_directory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            updated, routes, points, audit = build_nature_site_route_artifacts(enriched, output_dir, config)

            self.assertTrue((output_dir / "production_nature_site_routes.csv").exists())
            self.assertTrue((output_dir / "production_nature_site_route_points.csv").exists())
            self.assertTrue((output_dir / "production_nature_site_route_audit.csv").exists())

        self.assertFalse(routes.empty)
        self.assertFalse(points.empty)
        self.assertFalse(audit.empty)
        self.assertGreater(updated["internal_route_count"].iloc[0], 0)
        self.assertIn("curated_internal_route_fallback", set(routes["source"]))

    def test_internal_route_score_bonus_is_bounded(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "interest": {"mode": "nature_heavy", "weights": {"nature": 0.55, "city": 0.2, "culture": 0.15, "history": 0.1}},
                "nature_detail_routes": {"soft_score_weight": 0.06},
            },
        )
        frame = pd.DataFrame(
            [
                {
                    "name": "Sequoia National Park",
                    "final_poi_value": 0.5,
                    "best_internal_route_score": 1.0,
                    "best_internal_route_duration_minutes": 90,
                    "internal_route_source": "curated_internal_route_fallback",
                }
            ]
        )

        adjusted = apply_nature_site_route_score_bonus(frame, config)

        self.assertGreater(adjusted["final_poi_value"].iloc[0], 0.5)
        self.assertLessEqual(adjusted["internal_route_value_bonus"].iloc[0], 0.06 * 0.55 + 1e-9)

    def test_scenarios_have_expected_fallback_structure(self):
        california = get_scenario_definition("california_coast")
        las_vegas = get_scenario_definition("las_vegas_national_parks")
        statewide = get_scenario_definition("california_statewide_nature")

        self.assertIn(("San Francisco", "Los Angeles"), california.gateway_routes)
        self.assertIn("Las Vegas", las_vegas.overnight_base_cities)
        self.assertTrue(las_vegas.curated_seed_pois)
        self.assertIn("Yosemite National Park", statewide.nature_regions)
        self.assertIn("Mariposa", statewide.overnight_base_cities)
        self.assertGreater(len(statewide.overnight_base_cities), len(california.overnight_base_cities))

    def test_statewide_scenario_has_yosemite_region_and_is_not_ca1_only(self):
        regions = get_nature_region_definitions("california_statewide_nature")
        names = {region.name for region in regions}
        yosemite = next(region for region in regions if region.name == "Yosemite National Park")
        statewide = get_scenario_definition("california_statewide_nature")

        self.assertIn("Yosemite National Park", names)
        self.assertIn("Fresno", yosemite.gateway_bases)
        self.assertIn("Lake Tahoe", names)
        self.assertIn("Death Valley National Park", names)
        self.assertTrue(any("Yosemite" in " ".join(option["nature_regions"]) for option in statewide.route_options))

    def test_statewide_scenario_route_options_include_sierra_la_and_joshua_tree(self):
        statewide = get_scenario_definition("california_statewide_nature")
        labels = {option["label"] for option in statewide.route_options}
        route_text = " | ".join(
            " -> ".join(option["sequence"]) + " :: " + " -> ".join(option["nature_regions"])
            for option in statewide.route_options
        )

        self.assertIn("Sierra national parks", labels)
        self.assertIn("Coast and southern desert nature", labels)
        self.assertIn("Southern desert to Sierra national parks", labels)
        self.assertIn("Los Angeles", route_text)
        self.assertIn("Yosemite National Park", route_text)
        self.assertIn("Sequoia National Park", route_text)
        self.assertIn("Joshua Tree National Park", route_text)

    def test_nature_config_candidate_plans_prioritize_sierra_required_anchors(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {
                    "scenario": "california_statewide_nature",
                    "interest_profile": "nature_heavy",
                    "start_city_options": ["San Francisco"],
                    "end_city_options": ["Los Angeles"],
                },
                "interest": {"mode": "nature_heavy"},
                "nature": {
                    "required_selected_anchors_for_nature_heavy": ["Yosemite", "Sequoia"],
                    "preferred_statewide_route_shape": "sierra_national_parks",
                    "anchor_inclusion_policy": "require_if_feasible_else_explain",
                },
            },
        )
        cities = [
            "San Francisco",
            "Mariposa",
            "Fresno",
            "Three Rivers",
            "Los Angeles",
            "Monterey",
            "Big Sur",
            "Santa Barbara",
            "Palm Springs",
            "Twentynine Palms",
            "Soledad",
        ]
        summary = pd.DataFrame(
            {
                "city": cities,
                "city_value_score": [0.80, 0.50, 0.45, 0.55, 0.75, 0.60, 0.65, 0.58, 0.40, 0.30, 0.35],
                "data_uncertainty": [0.2] * len(cities),
                "data_confidence": [0.8] * len(cities),
                "city_must_go_score": [0.0] * len(cities),
                "city_must_go_count": [0.0] * len(cities),
                "city_social_score": [0.0] * len(cities),
            }
        )

        plans = sorted(candidate_plans(config, summary), key=lambda row: row["objective"], reverse=True)
        self.assertTrue(plans)
        best = plans[0]

        self.assertEqual(best["gateway_start"], "San Francisco")
        self.assertEqual(best["gateway_end"], "Los Angeles")
        self.assertEqual(best["route_option_label"], "Sierra national parks")
        self.assertIn("Mariposa", best["days_by_city"])
        self.assertIn("Three Rivers", best["days_by_city"])
        self.assertEqual(set(best["required_nature_anchor_coverage"]), {"Yosemite", "Sequoia"})
        self.assertEqual(best["required_nature_anchor_misses"], [])

    def test_scenario_enrichment_city_universe_expands_hardcoded_coastal_seed(self):
        cities = scenario_enrichment_city_universe(
            "california_statewide_nature",
            seed_city_names=["San Francisco", "Los Angeles", "Santa Barbara"],
            start_city_options=["San Francisco", "Los Angeles"],
            end_city_options=["Los Angeles", "San Francisco"],
        )

        self.assertIn("Mariposa", cities)
        self.assertIn("Three Rivers", cities)
        self.assertIn("Palm Springs", cities)
        self.assertIn("Los Angeles", cities)

    def test_statewide_curated_fallback_contains_real_park_candidates(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "california_statewide_nature"},
                "nature": {"use_nps_api": False, "use_curated_nature_fallback": True},
            },
        )
        with temporary_directory() as tmpdir:
            pois, _audit = load_nps_or_curated_nature_pois(Path(tmpdir) / "outputs", config)

        names = set(pois["name"].astype(str))
        self.assertIn("Yosemite National Park", names)
        self.assertIn("Sequoia National Park", names)
        self.assertIn("Joshua Tree National Park", names)
        yosemite = pois[pois["name"].eq("Yosemite National Park")].iloc[0]
        self.assertTrue(bool(yosemite["is_national_park"]))
        self.assertTrue(bool(yosemite["social_must_go"]))
        self.assertGreater(float(yosemite["data_confidence"]), 0.0)

    def test_blueprint_catalog_normalization_preserves_nature_route_columns(self):
        blueprint_trip_map = import_legacy_module("blueprint_trip_map")
        frame = pd.DataFrame(
            [
                {
                    "name": "Yosemite National Park",
                    "city": "Mariposa",
                    "latitude": 37.8651,
                    "longitude": -119.5383,
                    "category": "national_park",
                    "route_fit": 0.95,
                    "is_national_park": True,
                    "nature_region": "Yosemite National Park",
                    "interest_adjusted_value": 1.2,
                }
            ]
        )

        output = blueprint_trip_map._normalize_catalog_columns(frame, "Mariposa", "test")

        self.assertIn("route_fit", output.columns)
        self.assertIn("is_national_park", output.columns)
        self.assertIn("interest_adjusted_value", output.columns)
        self.assertAlmostEqual(float(output["route_fit"].iloc[0]), 0.95)
        self.assertTrue(bool(output["is_national_park"].iloc[0]))

    def test_artifact_metadata_mismatch_marks_dashboard_stale(self):
        config = load_trip_config(CONFIG_PATH, overrides={"trip": {"scenario": "california_coast"}})
        statewide_config = load_trip_config(
            CONFIG_PATH,
            overrides={"trip": {"scenario": "california_statewide_nature"}},
        )
        with temporary_directory() as tmpdir:
            write_artifact_metadata(tmpdir, config, artifact_files=["production_method_route_stops.csv"])

            self.assertTrue(artifact_metadata_matches(tmpdir, config))
            self.assertFalse(artifact_metadata_matches(tmpdir, statewide_config))

    def test_route_anchor_audit_explains_candidates_selected_and_gateways(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "california_statewide_nature", "interest_profile": "nature_heavy"},
                "interest": {"mode": "nature_heavy"},
            },
        )
        enriched = pd.DataFrame(
            [
                {
                    "name": "Yosemite National Park",
                    "city": "Mariposa",
                    "nature_region": "Yosemite National Park",
                    "category": "national_park",
                },
                {
                    "name": "Sequoia National Park",
                    "city": "Three Rivers",
                    "nature_region": "Sequoia National Park",
                    "category": "national_park",
                },
                {
                    "name": "Griffith Observatory",
                    "city": "Los Angeles",
                    "category": "social_must_go:viewpoint",
                },
            ]
        )
        route_stops = pd.DataFrame(
            [
                {
                    "method": "hierarchical_bandit_gurobi_repair",
                    "day": 2,
                    "attraction_name": "Sequoia National Park",
                    "city": "Three Rivers",
                    "overnight_city": "Three Rivers",
                    "gateway_start": "Los Angeles",
                    "gateway_end": "San Francisco",
                    "route_start_city": "Los Angeles",
                    "route_end_city": "Three Rivers",
                }
            ]
        )
        with temporary_directory() as tmpdir:
            audit = _write_route_anchor_audit(
                output_dir=Path(tmpdir),
                enriched_df=enriched,
                route_stops_df=route_stops,
                config=config,
            )
            self.assertTrue((Path(tmpdir) / "production_dataset_completeness_audit.csv").exists())

        by_anchor = audit.set_index("anchor")
        self.assertEqual(by_anchor.loc["Sequoia", "role"], "selected_stop")
        self.assertEqual(by_anchor.loc["Yosemite", "role"], "candidate_only")
        self.assertEqual(by_anchor.loc["Los Angeles", "role"], "gateway")
        self.assertTrue(bool(by_anchor.loc["Los Angeles", "gateway_or_base"]))

    def test_required_anchor_stops_survive_feasible_base_day_selection(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {
                    "scenario": "california_statewide_nature",
                    "interest_profile": "nature_heavy",
                },
                "interest": {"mode": "nature_heavy"},
                "nature": {
                    "required_selected_anchors_for_nature_heavy": ["Yosemite", "Sequoia"],
                    "anchor_inclusion_policy": "require_if_feasible_else_explain",
                },
            },
        )
        route_stops = pd.DataFrame(
            [
                {
                    "comparison_type": "method",
                    "comparison_label": "Method",
                    "method": "hierarchical_bandit_gurobi_repair",
                    "method_display_name": "Bandit + Repair",
                    "trip_days": 7,
                    "gateway_start": "San Francisco",
                    "gateway_end": "Los Angeles",
                    "day": 2,
                    "stop_order": 1,
                    "city": "Mariposa",
                    "overnight_city": "Mariposa",
                    "hotel_name": "Mariposa Hotel",
                    "hotel_latitude": 37.48,
                    "hotel_longitude": -119.96,
                    "attraction_name": "Local Museum",
                    "latitude": 37.49,
                    "longitude": -119.97,
                    "category": "museum",
                    "source_list": "fixture",
                    "social_must_go": False,
                    "final_poi_value": 0.2,
                    "route_fit": 0.2,
                    "route_start_city": "San Francisco",
                    "route_start_name": "SF Hotel",
                    "route_start_latitude": 37.77,
                    "route_start_longitude": -122.42,
                    "route_end_city": "Mariposa",
                    "route_end_name": "Mariposa Hotel",
                    "route_end_latitude": 37.48,
                    "route_end_longitude": -119.96,
                },
                {
                    "comparison_type": "method",
                    "comparison_label": "Method",
                    "method": "hierarchical_bandit_gurobi_repair",
                    "method_display_name": "Bandit + Repair",
                    "trip_days": 7,
                    "gateway_start": "San Francisco",
                    "gateway_end": "Los Angeles",
                    "day": 4,
                    "stop_order": 1,
                    "city": "Three Rivers",
                    "overnight_city": "Three Rivers",
                    "hotel_name": "Three Rivers Hotel",
                    "hotel_latitude": 36.44,
                    "hotel_longitude": -118.90,
                    "attraction_name": "Foothills Visitor Center",
                    "latitude": 36.45,
                    "longitude": -118.91,
                    "category": "visitor_center",
                    "source_list": "fixture",
                    "social_must_go": False,
                    "final_poi_value": 0.2,
                    "route_fit": 0.2,
                    "route_start_city": "Fresno",
                    "route_start_name": "Fresno Hotel",
                    "route_start_latitude": 36.74,
                    "route_start_longitude": -119.79,
                    "route_end_city": "Three Rivers",
                    "route_end_name": "Three Rivers Hotel",
                    "route_end_latitude": 36.44,
                    "route_end_longitude": -118.90,
                },
            ]
        )
        for column in route_stops.columns:
            route_stops[column] = route_stops[column]
        enriched = pd.DataFrame(
            [
                {
                    "name": "Yosemite National Park",
                    "city": "Mariposa",
                    "latitude": 37.8651,
                    "longitude": -119.5383,
                    "category": "national_park",
                    "source_list": "curated_nature_region_seed",
                    "final_poi_value": 0.95,
                    "interest_adjusted_value": 1.20,
                    "nature_score": 0.98,
                    "scenic_score": 0.98,
                    "is_nature": True,
                    "is_national_park": True,
                    "nature_region": "Yosemite National Park",
                },
                {
                    "name": "Sequoia National Park",
                    "city": "Three Rivers",
                    "latitude": 36.4864,
                    "longitude": -118.5658,
                    "category": "national_park",
                    "source_list": "curated_nature_region_seed",
                    "final_poi_value": 0.94,
                    "interest_adjusted_value": 1.15,
                    "nature_score": 0.97,
                    "scenic_score": 0.96,
                    "is_nature": True,
                    "is_national_park": True,
                    "nature_region": "Sequoia National Park",
                },
            ]
        )

        patched, notes = _ensure_required_anchor_stops(route_stops, enriched_df=enriched, config=config)
        selected_names = " | ".join(patched["attraction_name"].astype(str))

        self.assertIn("Yosemite National Park", selected_names)
        self.assertIn("Sequoia National Park", selected_names)
        self.assertTrue(any("Required anchor Yosemite" in note for note in notes))
        self.assertTrue(any("Required anchor Sequoia" in note for note in notes))

    def test_interest_preview_json_has_browser_fields(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "interest": {"enabled": True, "weights": {"nature": 0.6, "city": 0.2, "culture": 0.1, "history": 0.1}}
            },
        )
        frame = pd.DataFrame(
            [
                {
                    "name": "Scenic Park",
                    "city": "Monterey",
                    "latitude": 36.5,
                    "longitude": -121.9,
                    "category": "state park scenic viewpoint",
                    "source_list": "curated",
                    "final_poi_value": 0.8,
                    "weather_risk": 0.1,
                }
            ]
        )
        preview = build_interest_bar_preview(frame, config)

        self.assertIn("top_boosted_pois", preview)
        self.assertIn("bar", preview)
        self.assertIn("park_bonus", preview["top_boosted_pois"][0])
        self.assertIn("preview_route_graph", preview)
        self.assertIn("segments", preview["preview_route_graph"])

        with temporary_directory() as tmpdir:
            write_interest_catalog_artifacts(frame, tmpdir, config)
            self.assertTrue((Path(tmpdir) / "production_interest_bar_preview.json").exists())

    def test_map_export_generates_artifact_size_report_and_lightweight_share(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"scenario": "california_statewide_nature", "interest_profile": "nature_heavy"},
                "map_export": {"mode": "both", "lightweight_max_routes": 1},
            },
        )
        route_df = pd.DataFrame(
            [
                {
                    "route_key": "best",
                    "day": 1,
                    "stop_order": 1,
                    "attraction_name": "San Francisco",
                    "city": "San Francisco",
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "final_poi_value": 0.8,
                },
                {
                    "route_key": "best",
                    "day": 2,
                    "stop_order": 1,
                    "attraction_name": "Yosemite National Park",
                    "city": "Mariposa",
                    "latitude": 37.8651,
                    "longitude": -119.5383,
                    "final_poi_value": 0.96,
                    "nature_region": "Yosemite National Park",
                },
                {
                    "route_key": "comparison",
                    "day": 1,
                    "stop_order": 1,
                    "attraction_name": "Debug Route Stop",
                    "city": "Los Angeles",
                    "latitude": 34.0522,
                    "longitude": -118.2437,
                    "final_poi_value": 0.5,
                },
            ]
        )
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "comparison_type": "method",
                        "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                        "method": "hierarchical_bandit_gurobi_repair",
                        "method_display_name": "Hierarchical + Bandit + Small Gurobi Repair",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Preferred Bandit Stop",
                        "city": "San Francisco",
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "final_poi_value": 0.91,
                        "interest_adjusted_value": 0.0,
                        "hotel_name": "Optimizer Hotel",
                        "hotel_latitude": 37.78,
                        "hotel_longitude": -122.42,
                    },
                    {
                        "comparison_type": "method",
                        "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                        "method": "hierarchical_bandit_gurobi_repair",
                        "method_display_name": "Hierarchical + Bandit + Small Gurobi Repair",
                        "trip_days": 7,
                        "day": 2,
                        "stop_order": 1,
                        "attraction_name": "Preferred Bandit Nature Stop",
                        "city": "Mariposa",
                        "latitude": 37.8651,
                        "longitude": -119.5383,
                        "final_poi_value": 0.96,
                        "nature_region": "Yosemite National Park",
                    },
                    {
                        "comparison_type": "method",
                        "comparison_label": "Method · Hierarchical Greedy Baseline",
                        "method": "hierarchical_greedy_baseline",
                        "method_display_name": "Hierarchical Greedy Baseline",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Nondefault Greedy Stop",
                        "city": "Los Angeles",
                        "latitude": 34.0522,
                        "longitude": -118.2437,
                        "final_poi_value": 0.5,
                    },
                ]
            ).to_csv(output_dir / "production_method_route_stops.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "route_key": "balanced_7_day",
                        "comparison_label": "7-day balanced route",
                        "profile": "balanced",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Seven Day Stop",
                        "city": "San Francisco",
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "final_poi_value": 0.7,
                    },
                    {
                        "route_key": "balanced_9_day",
                        "comparison_label": "9-day balanced route",
                        "profile": "balanced",
                        "trip_days": 9,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Nine Day Stop",
                        "city": "Santa Barbara",
                        "latitude": 34.4208,
                        "longitude": -119.6982,
                        "final_poi_value": 0.8,
                    },
                ]
            ).to_csv(output_dir / "production_trip_length_route_stops.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "route_key": "matrix_balanced_coast",
                        "comparison_label": "Matrix · Balanced Coast",
                        "profile": "balanced",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Matrix San Francisco",
                        "city": "San Francisco",
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "final_poi_value": 0.81,
                    },
                    {
                        "route_key": "matrix_balanced_coast",
                        "comparison_label": "Matrix · Balanced Coast",
                        "profile": "balanced",
                        "trip_days": 7,
                        "day": 2,
                        "stop_order": 1,
                        "attraction_name": "Matrix Santa Barbara",
                        "city": "Santa Barbara",
                        "latitude": 34.4208,
                        "longitude": -119.6982,
                        "final_poi_value": 0.79,
                    },
                ]
            ).to_csv(output_dir / "production_route_matrix_route_stops.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "route_key": "interest_nature",
                        "comparison_label": "Interest · Nature Heavy",
                        "interest_profile": "nature_heavy",
                        "profile": "nature_heavy",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Interest Yosemite",
                        "city": "Mariposa",
                        "latitude": 37.8651,
                        "longitude": -119.5383,
                        "final_poi_value": 0.93,
                    },
                    {
                        "route_key": "interest_city",
                        "comparison_label": "Interest · City Heavy",
                        "interest_profile": "city_heavy",
                        "profile": "city_heavy",
                        "trip_days": 7,
                        "day": 1,
                        "stop_order": 1,
                        "attraction_name": "Interest Los Angeles",
                        "city": "Los Angeles",
                        "latitude": 34.0522,
                        "longitude": -118.2437,
                        "final_poi_value": 0.75,
                    },
                ]
            ).to_csv(output_dir / "production_interest_route_stops.csv", index=False)
            pd.DataFrame(
                [
                    {"route_layer": "coast_context", "leg_order": 1, "from": "San Francisco", "to": "Santa Barbara"},
                ]
            ).to_csv(output_dir / "production_intercity_legs.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "city": "San Francisco",
                        "hotel_name": "Debug Hotel",
                        "candidate_rank": 1,
                        "latitude": 37.78,
                        "longitude": -122.42,
                    }
                ]
            ).to_csv(output_dir / "production_hotel_selection_debug.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "name": "Must Go Pier",
                        "city": "Santa Barbara",
                        "latitude": 34.41,
                        "longitude": -119.69,
                        "final_poi_value": 0.88,
                    }
                ]
            ).to_csv(output_dir / "production_social_must_go_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "name": "Yosemite Valley",
                        "city": "Yosemite Valley",
                        "latitude": 37.7456,
                        "longitude": -119.5936,
                        "is_nature": True,
                        "nature_score": 0.95,
                        "scenic_score": 0.98,
                        "final_poi_value": 0.9,
                    }
                ]
            ).to_csv(output_dir / "production_enriched_poi_catalog.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "site_id": "yosemite",
                        "site_name": "Yosemite Valley",
                        "city": "Yosemite Valley",
                        "nature_region": "Yosemite National Park",
                        "latitude": 37.7456,
                        "longitude": -119.5936,
                        "route_id": "nature_site__yosemite__valley_loop",
                        "route_name": "Yosemite Valley Loop",
                        "route_type": "viewpoint_walk",
                        "distance_km": 2.4,
                        "duration_minutes": 55,
                        "difficulty": "easy",
                        "route_score": 0.82,
                        "source_confidence": 0.58,
                        "source": "curated_internal_route_fallback",
                        "source_url": "https://www.nps.gov/yose/",
                        "description": "Short internal Yosemite route for dashboard export testing.",
                        "geometry_source": "curated_geometry",
                        "fallback_used": True,
                        "point_count": 2,
                    }
                ]
            ).to_csv(output_dir / "production_nature_site_routes.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "route_id": "nature_site__yosemite__valley_loop",
                        "point_order": 1,
                        "latitude": 37.7456,
                        "longitude": -119.5936,
                        "name": "Yosemite Valley Loop",
                        "route_name": "Yosemite Valley Loop",
                        "site_name": "Yosemite Valley",
                        "nature_region": "Yosemite National Park",
                        "route_type": "viewpoint_walk",
                        "source_confidence": 0.58,
                    },
                    {
                        "route_id": "nature_site__yosemite__valley_loop",
                        "point_order": 2,
                        "latitude": 37.7554,
                        "longitude": -119.5963,
                        "name": "Yosemite Valley Loop",
                        "route_name": "Yosemite Valley Loop",
                        "site_name": "Yosemite Valley",
                        "nature_region": "Yosemite National Park",
                        "route_type": "viewpoint_walk",
                        "source_confidence": 0.58,
                    },
                ]
            ).to_csv(output_dir / "production_nature_site_route_points.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "site_id": "yosemite",
                        "site_name": "Yosemite Valley",
                        "city": "Yosemite Valley",
                        "nature_region": "Yosemite National Park",
                        "candidate": True,
                        "selected": True,
                        "route_count": 1,
                        "source_status": "curated_internal_route_fallback",
                        "fallback_used": True,
                        "missing_reason": "",
                    }
                ]
            ).to_csv(output_dir / "production_nature_site_route_audit.csv", index=False)
            pd.DataFrame(
                [{"route_key": "best", "layer_group": "selected", "method": "hierarchical_bandit_gurobi_repair"}]
            ).to_csv(output_dir / "production_map_route_debug.csv", index=False)
            (output_dir / "production_interest_bar_preview.json").write_text(
                json.dumps({"bar": {"nature": 0.55, "city": 0.2}, "top_boosted_pois": [{"name": "Yosemite"}]}),
                encoding="utf-8",
            )
            artifacts = export_map_artifacts(
                route_df,
                output_dir=output_dir,
                figure_dir=root / "figures",
                config=config,
            )
            share_html = artifacts["lightweight_share_map"].read_text(encoding="utf-8").lower()
            dashboard_root = artifacts["full_interactive_dashboard"].parent
            index_html = artifacts["full_interactive_dashboard"].read_text(encoding="utf-8")
            research_html = (dashboard_root / "research.html").read_text(encoding="utf-8")
            customer_html = (dashboard_root / "customer.html").read_text(encoding="utf-8")
            style_css = (dashboard_root / "assets" / "style.css").read_text(encoding="utf-8")
            map_js = (dashboard_root / "assets" / "map_controls.js").read_text(encoding="utf-8")
            dashboard_js = (dashboard_root / "assets" / "dashboard.js").read_text(encoding="utf-8")
            loader_js = (dashboard_root / "assets" / "data_loader.js").read_text(encoding="utf-8")
            evaluation_html = (dashboard_root / "evaluation.html").read_text(encoding="utf-8")
            route_index = json.loads((dashboard_root / "assets" / "route_index.json").read_text(encoding="utf-8"))
            metrics_payload = json.loads(
                (dashboard_root / "assets" / "dashboard_metrics.json").read_text(encoding="utf-8")
            )
            evaluation_payload = json.loads(
                (dashboard_root / "assets" / "evaluation_metrics.json").read_text(encoding="utf-8")
            )
            selected_route_geojson = json.loads(
                (dashboard_root / "assets" / "routes" / "selected_route.geojson").read_text(encoding="utf-8")
            )
            hotel_geojson = json.loads(
                (dashboard_root / "assets" / "routes" / "hotel_candidates.geojson").read_text(encoding="utf-8")
            )
            nature_geojson = json.loads(
                (dashboard_root / "assets" / "routes" / "nature_candidates.geojson").read_text(encoding="utf-8")
            )
            must_go_geojson = json.loads(
                (dashboard_root / "assets" / "routes" / "must_go_candidates.geojson").read_text(encoding="utf-8")
            )
            poi_payload = json.loads((dashboard_root / "assets" / "pois" / "selected_route_pois.json").read_text())
            playback_payload = json.loads((dashboard_root / "assets" / "playback_data.json").read_text())
            selected_hotels_payload = json.loads((dashboard_root / "assets" / "selected_hotels.json").read_text())
            hotel_payload = json.loads((dashboard_root / "assets" / "hotel_choices.json").read_text())
            nature_payload = json.loads((dashboard_root / "assets" / "nature_explore.json").read_text())
            nature_site_payload = json.loads((dashboard_root / "assets" / "nature_site_routes.json").read_text())
            generated_asset_names = {
                path.relative_to(dashboard_root).as_posix()
                for path in (dashboard_root / "assets").rglob("*")
                if path.is_file()
            }
            report = pd.read_csv(root / "outputs" / "production_map_artifact_size_report.csv")

        self.assertIn("lightweight_share_map", artifacts)
        self.assertIn("evaluation_dashboard", artifacts)
        self.assertIn("full_customer_dashboard", artifacts)
        self.assertIn("full_research_dashboard", artifacts)
        self.assertFalse(report.empty)
        self.assertIn('<div id="map">', index_html)
        self.assertIn("Research/Test dashboard", index_html)
        self.assertIn("dashboard-mode-toggle", index_html)
        self.assertIn('data-mode-section="customer"', index_html)
        self.assertIn('data-mode-section="research"', index_html)
        self.assertIn("Research/Test dashboard", research_html)
        self.assertIn('data-dashboard-mode="research"', research_html)
        self.assertIn("dashboard-mode-toggle", research_html)
        self.assertIn("assets/data_loader.js", index_html)
        self.assertIn("evaluation.html", index_html)
        self.assertIn("diagnostic-panel", index_html)
        self.assertIn("dashboard-drag-handle", index_html)
        self.assertIn("dashboard-collapse", index_html)
        self.assertIn("Playback", index_html)
        self.assertIn("Active stop details", index_html)
        self.assertIn("Hotel choices", index_html)
        self.assertIn("Nature explore", index_html)
        self.assertIn("Nature site routes", index_html)
        self.assertIn("Interest bars", index_html)
        self.assertIn("Saved optimized route", index_html)
        self.assertIn("Preview only route", index_html)
        self.assertIn('data-map-layer-toggle="terrain"', index_html)
        self.assertIn('data-map-layer-toggle="roads"', index_html)
        self.assertIn('data-map-layer-toggle="labels"', index_html)
        self.assertIn('data-map-layer-toggle="weather_risk"', index_html)
        self.assertNotIn("checked readonly", index_html)
        self.assertNotIn("Load optional layers", index_html)
        self.assertNotIn("alert(", index_html)
        self.assertIn('data-dashboard-mode="customer"', customer_html)
        self.assertIn("Customer trip planner", customer_html)
        self.assertIn("dashboard-mode-toggle", customer_html)
        self.assertIn("Plan your trip", customer_html)
        self.assertIn("customer-trip-controls", customer_html)
        self.assertIn("Trip summary", customer_html)
        self.assertIn("Hotel choices", customer_html)
        self.assertIn("Interest bars", customer_html)
        self.assertIn("evaluation.html", customer_html)
        self.assertIn("Debug summary", customer_html)
        self.assertIn("route-selector", customer_html)
        self.assertIn('data-mode-section="research"', customer_html)
        self.assertIn("height: 100vh", style_css)
        self.assertIn(".dashboard-panel", style_css)
        self.assertIn(".dashboard-panel.collapsed", style_css)
        self.assertIn(".dashboard-header", style_css)
        self.assertIn(".dashboard-mode-label", style_css)
        self.assertIn('[data-dashboard-mode="customer"] [data-mode-section="research"]', style_css)
        self.assertIn(".customer-control-grid", style_css)
        self.assertIn(".playback-progress", style_css)
        self.assertIn(".hotel-card", style_css)
        self.assertIn(".nature-card", style_css)
        self.assertIn(".nature-route-mini-card", style_css)
        self.assertIn(".interest-axis", style_css)
        self.assertIn('L.map("map"', map_js)
        self.assertIn("drawDefaultRoute", map_js)
        self.assertIn("renderRouteSelector", map_js)
        self.assertIn("runQuickAction", map_js)
        self.assertIn("loadOptionalLayers", map_js)
        self.assertIn("zoomVisibleRoutes", map_js)
        self.assertIn("playRouteAnimation", map_js)
        self.assertIn("pauseRouteAnimation", map_js)
        self.assertIn("requestAnimationFrame", map_js)
        self.assertIn("setPlaybackStop", map_js)
        self.assertIn("isPlayableRoute", map_js)
        self.assertIn("firstPlayableRouteId", map_js)
        self.assertIn("routeRecord?.marker_only", map_js)
        self.assertIn("drawSelectedHotels", map_js)
        self.assertIn("toggleSelectedHotels", map_js)
        self.assertIn("toggleHotelCandidates", map_js)
        self.assertIn("drawLivePreviewRoute", map_js)
        self.assertIn("Preview only - rerun pipeline to save", map_js)
        self.assertIn("clearLivePreviewRoute", map_js)
        self.assertIn("initializeBaseMapLayers", map_js)
        self.assertIn("bindMapLayerToggles", map_js)
        self.assertIn("buildLabelLayer", map_js)
        self.assertIn("buildWeatherRiskLayer", map_js)
        self.assertIn("numbered-route-stop", map_js)
        self.assertIn("data-map-layer-toggle", map_js)
        self.assertIn("nature_site_routes", map_js)
        self.assertIn("renderCustomerControls(index)", map_js)
        self.assertIn("window.dashboardMap", map_js)
        self.assertNotIn("alert(", map_js)
        self.assertIn("renderActiveStopDetail", dashboard_js)
        self.assertIn("renderCityDetails", dashboard_js)
        self.assertIn("renderHotelChoices", dashboard_js)
        self.assertIn("setPreferredHotel", dashboard_js)
        self.assertIn("clearPreferredHotel", dashboard_js)
        self.assertIn("Export preferences", dashboard_js)
        self.assertIn("preferred_hotels", dashboard_js)
        self.assertIn("Limited hotel data", dashboard_js)
        self.assertIn("window.localStorage", dashboard_js)
        self.assertIn("renderNatureExplore", dashboard_js)
        self.assertIn("natureRoutesForPlace", dashboard_js)
        self.assertIn("bindNatureRouteButtons", dashboard_js)
        self.assertIn("Inside this site", dashboard_js)
        self.assertIn("data-show-nature-layer", dashboard_js)
        self.assertIn("bindDashboardShellControls", dashboard_js)
        self.assertIn("data-interest-axis", dashboard_js)
        self.assertIn("normalizeInterestBarValues", dashboard_js)
        self.assertIn("candidateAllowedByWeights", dashboard_js)
        self.assertIn("dominantPreviewAxis", dashboard_js)
        self.assertIn("previewScore", dashboard_js)
        self.assertIn("Default route is computed by Python optimization using nature-aware value weights", dashboard_js)
        self.assertIn("National park candidates stay in candidate/context layers", dashboard_js)
        self.assertIn("Build live preview route", dashboard_js)
        self.assertIn("buildLivePreviewRoute", dashboard_js)
        self.assertIn("window.setTimeout(buildLivePreviewRoute", dashboard_js)
        self.assertIn("renderCustomerControls", dashboard_js)
        self.assertIn("Loaded saved ${kind} route artifact", dashboard_js)
        self.assertIn("customer_visible", dashboard_js)
        self.assertIn("dashboard-mode-toggle", dashboard_js)
        self.assertIn("applyDashboardMode", dashboard_js)
        self.assertIn("Route state", dashboard_js)
        self.assertIn("Browser preview is approximate", dashboard_js)
        self.assertIn("data-toggle-hotel-candidates", dashboard_js)
        self.assertIn("Method Evaluation Dashboard", evaluation_html)
        self.assertIn("assets/evaluation_metrics.js", evaluation_html)
        self.assertIn("dashboardIsFileMode", loader_js)
        self.assertIn("loadScriptOnce", loader_js)
        self.assertIn("loadDashboardAsset", loader_js)
        self.assertIn("loadDebugSummary", loader_js)
        self.assertIn("loadInterestPreview", loader_js)
        self.assertIn("loadPlaybackData", loader_js)
        self.assertIn("loadCityDetails", loader_js)
        self.assertIn("loadSelectedHotels", loader_js)
        self.assertIn("loadHotelChoices", loader_js)
        self.assertIn("loadNatureExplore", loader_js)
        self.assertIn("loadNatureSiteRoutes", loader_js)
        self.assertIn("[dashboard-loader]", loader_js)
        self.assertIn("[dashboard-assets]", loader_js)
        self.assertIn("[dashboard-error]", loader_js)
        self.assertIn("if (level !== 'error')", loader_js)
        self.assertIn("assets/route_index.js", generated_asset_names)
        self.assertIn("assets/dashboard_metrics.json", generated_asset_names)
        self.assertIn("assets/dashboard_metrics.js", generated_asset_names)
        self.assertIn("assets/debug_summary.json", generated_asset_names)
        self.assertIn("assets/debug_summary.js", generated_asset_names)
        self.assertIn("assets/interest_preview.json", generated_asset_names)
        self.assertIn("assets/interest_preview.js", generated_asset_names)
        self.assertIn("assets/playback_data.json", generated_asset_names)
        self.assertIn("assets/playback_data.js", generated_asset_names)
        self.assertIn("assets/city_details.json", generated_asset_names)
        self.assertIn("assets/city_details.js", generated_asset_names)
        self.assertIn("assets/selected_hotels.json", generated_asset_names)
        self.assertIn("assets/selected_hotels.js", generated_asset_names)
        self.assertIn("assets/hotel_choices.json", generated_asset_names)
        self.assertIn("assets/hotel_choices.js", generated_asset_names)
        self.assertIn("assets/nature_explore.json", generated_asset_names)
        self.assertIn("assets/nature_explore.js", generated_asset_names)
        self.assertIn("assets/nature_site_routes.json", generated_asset_names)
        self.assertIn("assets/nature_site_routes.js", generated_asset_names)
        self.assertIn("assets/routes/nature_site_routes.geojson", generated_asset_names)
        self.assertIn("assets/pois/nature_site_routes_pois.json", generated_asset_names)
        self.assertIn("assets/evaluation_metrics.json", generated_asset_names)
        self.assertIn("assets/evaluation_metrics.js", generated_asset_names)
        self.assertIn("assets/routes/selected_route.geojson", generated_asset_names)
        self.assertIn("assets/routes/selected_route.js", generated_asset_names)
        self.assertIn("assets/pois/selected_route_pois.json", generated_asset_names)
        self.assertIn("assets/pois/selected_route_pois.js", generated_asset_names)
        self.assertEqual(route_index["contract_version"], "core-route-index-v1")
        self.assertEqual(metrics_payload["scenario"], "california_statewide_nature")
        self.assertEqual(metrics_payload["scenario_label"], "California Statewide Nature")
        self.assertEqual(metrics_payload["default_route_method"], "hierarchical_bandit_gurobi_repair")
        self.assertEqual(metrics_payload["route_state_label"], "Saved optimized route")
        self.assertEqual(metrics_payload["preview_state_label"], "Preview only - rerun pipeline to save")
        self.assertIn("anchor_audit", metrics_payload)
        self.assertNotIn("default_route", route_index)
        self.assertNotIn("pois", route_index)
        self.assertGreater(len(route_index["routes"]), 1)
        required_route_fields = {
            "id",
            "label",
            "default",
            "optional",
            "family",
            "selector_group",
            "geojson",
            "geojson_js",
            "pois",
            "pois_js",
            "customer_visible",
            "research_only",
            "customer_control_group",
        }
        self.assertTrue(
            required_route_fields.issubset(route_index["routes"][0]),
            route_index["routes"][0],
        )
        route_families = {route["family"] for route in route_index["routes"]}
        self.assertTrue(
            {
                "selected",
                "route_matrix",
                "trip_length",
                "method",
                "context",
                "hotel",
                "must_go",
                "nature",
                "nature_detail",
            }.issubset(route_families),
            route_families,
        )
        self.assertTrue(route_index["routes"][0]["default"])
        self.assertEqual(route_index["routes"][0]["family"], "selected")
        self.assertEqual(route_index["routes"][0]["selector_group"], "default")
        self.assertNotIn("path", route_index["routes"][0])
        self.assertTrue(any(route["optional"] for route in route_index["routes"]))
        route_by_id = {route["id"]: route for route in route_index["routes"]}
        self.assertTrue(route_by_id["selected_route"]["playable"])
        self.assertTrue(route_by_id["selected_route"]["customer_visible"])
        self.assertFalse(route_by_id["selected_route"]["research_only"])
        self.assertEqual(route_by_id["selected_route"]["customer_control_group"], "saved_route")
        self.assertFalse(route_by_id["hotel_candidates"]["playable"])
        self.assertTrue(route_by_id["hotel_candidates"]["marker_only"])
        self.assertFalse(route_by_id["hotel_candidates"]["customer_visible"])
        self.assertTrue(route_by_id["hotel_candidates"]["research_only"])
        self.assertTrue(route_by_id["nature_candidates"]["marker_only"])
        self.assertFalse(route_by_id["nature_site_routes"]["playable"])
        self.assertFalse(route_by_id["nature_site_routes"]["marker_only"])
        self.assertTrue(route_by_id["nature_site_routes"]["customer_visible"])
        self.assertTrue(route_by_id["must_go_candidates"]["marker_only"])
        self.assertTrue(any(route.get("customer_control_group") == "trip_days" for route in route_index["routes"]))
        self.assertTrue(any(route.get("customer_control_group") == "interest" for route in route_index["routes"]))
        for payload in [hotel_geojson, nature_geojson, must_go_geojson]:
            self.assertFalse(
                any(feature.get("geometry", {}).get("type") == "LineString" for feature in payload["features"])
            )
        self.assertTrue(
            any(feature.get("geometry", {}).get("type") == "LineString" for feature in selected_route_geojson["features"])
        )
        self.assertTrue(any("days_7" in route.get("quick_groups", []) for route in route_index["routes"]))
        self.assertTrue(any("balanced" in route.get("quick_groups", []) for route in route_index["routes"]))
        self.assertIn("full_customer_dashboard", set(report["artifact_type"]))
        self.assertIn("full_research_dashboard", set(report["artifact_type"]))
        self.assertEqual(poi_payload[0]["name"], "Preferred Bandit Stop")
        self.assertAlmostEqual(poi_payload[0]["display_utility"], 0.91)
        self.assertEqual(poi_payload[0]["optimization_value_source"], "final_poi_value")
        for field in {
            "description",
            "why_selected",
            "expected_duration_minutes",
            "weather_risk",
            "source_confidence",
            "type",
            "city_or_anchor",
        }:
            self.assertIn(field, poi_payload[0])
        self.assertIn("selected_route", playback_payload["routes"])
        self.assertNotIn("hotel_candidates", playback_payload["routes"])
        self.assertNotIn("nature_candidates", playback_payload["routes"])
        self.assertNotIn("nature_site_routes", playback_payload["routes"])
        self.assertNotIn("must_go_candidates", playback_payload["routes"])
        self.assertEqual(playback_payload["routes"]["selected_route"]["stops"][0]["name"], "Preferred Bandit Stop")
        self.assertIn("description", playback_payload["routes"]["selected_route"]["stops"][0])
        self.assertTrue(selected_hotels_payload["items"])
        self.assertEqual(selected_hotels_payload["items"][0]["hotel_name"], "Optimizer Hotel")
        self.assertIn("San Francisco", hotel_payload["cities"])
        self.assertTrue(hotel_payload["cities"]["San Francisco"][0]["hotel_name"])
        self.assertTrue(nature_payload["items"])
        self.assertEqual(nature_payload["items"][0]["name"], "Yosemite Valley")
        self.assertTrue(nature_site_payload["available"])
        self.assertEqual(nature_site_payload["routes"][0]["route_name"], "Yosemite Valley Loop")
        self.assertIn("share-map-data", share_html)
        self.assertIn("l.map('map'", share_html)
        self.assertIn("preferred bandit stop", share_html)
        self.assertNotIn("route_matrix", share_html)
        self.assertNotIn("debug route stop", share_html)
        self.assertNotIn("candidate layers", share_html)
        self.assertIn("full_dashboard_index", set(report["artifact_type"]))
        self.assertIn("route_geojson", set(report["artifact_type"]))
        self.assertIn("route_js_fallback", set(report["artifact_type"]))
        self.assertIn("poi_json", set(report["artifact_type"]))
        self.assertIn("poi_js_fallback", set(report["artifact_type"]))
        self.assertTrue(report["notes"].astype(str).str.contains("optional_layer").any())
        self.assertEqual(
            {
                "hierarchical_gurobi_pipeline",
                "hierarchical_greedy_baseline",
                "hierarchical_bandit_gurobi_repair",
            },
            {method["method"] for method in evaluation_payload["methods"]},
        )

    def test_dashboard_validation_script_passes_and_fails_actionably(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={"map_export": {"mode": "both", "lightweight_max_routes": 1}},
        )
        route_df = pd.DataFrame(
            [
                {
                    "route_key": "best",
                    "day": 1,
                    "stop_order": 1,
                    "attraction_name": "San Francisco",
                    "city": "San Francisco",
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "final_poi_value": 0.8,
                },
                {
                    "route_key": "best",
                    "day": 2,
                    "stop_order": 1,
                    "attraction_name": "Yosemite National Park",
                    "city": "Mariposa",
                    "latitude": 37.8651,
                    "longitude": -119.5383,
                    "final_poi_value": 0.96,
                },
            ]
        )
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "route_id": "nature_site__yosemite__valley_loop__1",
                        "site_id": "yosemite",
                        "site_name": "Yosemite National Park",
                        "city": "Mariposa",
                        "nature_region": "Yosemite National Park",
                        "latitude": 37.7456,
                        "longitude": -119.5936,
                        "route_name": "Yosemite Valley Loop",
                        "route_type": "scenic_drive",
                        "distance_km": 18.0,
                        "duration_minutes": 75,
                        "difficulty": "easy",
                        "source": "curated_internal_route_fallback",
                        "source_url": "https://www.nps.gov/yose/index.htm",
                        "source_confidence": 0.72,
                        "route_score": 0.86,
                        "fallback_used": True,
                        "description": "Curated Yosemite Valley internal route detail.",
                    }
                ]
            ).to_csv(output_dir / "production_nature_site_routes.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "route_id": "nature_site__yosemite__valley_loop__1",
                        "site_id": "yosemite",
                        "site_name": "Yosemite National Park",
                        "route_name": "Yosemite Valley Loop",
                        "point_order": 1,
                        "latitude": 37.7456,
                        "longitude": -119.5936,
                    },
                    {
                        "route_id": "nature_site__yosemite__valley_loop__1",
                        "site_id": "yosemite",
                        "site_name": "Yosemite National Park",
                        "route_name": "Yosemite Valley Loop",
                        "point_order": 2,
                        "latitude": 37.7430,
                        "longitude": -119.6020,
                    },
                ]
            ).to_csv(output_dir / "production_nature_site_route_points.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "site_id": "yosemite",
                        "site_name": "Yosemite National Park",
                        "city": "Mariposa",
                        "nature_region": "Yosemite National Park",
                        "candidate": True,
                        "selected": True,
                        "route_count": 1,
                        "source_status": "curated_internal_route_fallback",
                        "fallback_used": True,
                        "missing_reason": "",
                    }
                ]
            ).to_csv(output_dir / "production_nature_site_route_audit.csv", index=False)
            export_map_artifacts(route_df, output_dir=output_dir, figure_dir=root / "figures", config=config)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_dashboard_export.py"),
                "--figure-dir",
                str(root / "figures"),
                "--output-dir",
                str(output_dir),
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("PASSED", passed.stdout)

            (
                root / "figures" / "full_interactive_dashboard" / "assets" / "routes" / "selected_route.geojson"
            ).write_text("", encoding="utf-8")
            failed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("route GeoJSON", failed.stdout)

    def test_production_notebook_reloads_dashboard_exporter(self):
        notebook = json.loads(
            (REPO_ROOT / "notebook" / "production_system_blueprint.ipynb").read_text(encoding="utf-8")
        )
        notebook_text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

        self.assertIn("import itinerary_system.map_exporter as map_exporter_module", notebook_text)
        self.assertLess(notebook_text.index("map_exporter_module"), notebook_text.index("map_renderer_module"))
        self.assertIn("import itinerary_system.map_exporter as map_exporter", notebook_text)
        self.assertIn("map_exporter = importlib.reload(map_exporter)", notebook_text)
        self.assertIn('print("Using map exporter:", map_exporter.__file__)', notebook_text)
        self.assertIn('NATURE_DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "nature_trip_config.yaml"', notebook_text)
        self.assertIn('CONFIG_PATH = Path(os.getenv("TRIP_CONFIG_PATH", NATURE_DEMO_CONFIG_PATH))', notebook_text)
        self.assertIn("validate_dashboard_export.py", notebook_text)
        self.assertIn("validate_nature_route_pipeline.py", notebook_text)
        self.assertIn('"--strict"', notebook_text)

    def test_generated_large_outputs_are_ignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("notebook/*_executed.ipynb", gitignore)
        self.assertIn("results/cache/", gitignore)
        self.assertIn("results/outputs/*.csv", gitignore)
        self.assertIn("results/outputs/*.json", gitignore)
        self.assertIn("results/outputs/*.html", gitignore)
        self.assertIn("results/figures/*.html", gitignore)
        self.assertIn("results/figures/lightweight_share_map.html", gitignore)
        self.assertIn("results/figures/full_interactive_dashboard/", gitignore)
        self.assertIn("results/outputs/production_map_artifact_size_report.csv", gitignore)
        self.assertIn("results/quality/", gitignore)

    def test_quality_tooling_files_exist(self):
        self.assertTrue((REPO_ROOT / "pyproject.toml").exists())
        self.assertTrue((REPO_ROOT / ".pre-commit-config.yaml").exists())
        self.assertTrue((REPO_ROOT / ".github" / "workflows" / "quality.yml").exists())
        self.assertTrue((REPO_ROOT / "scripts" / "find_dead_code.py").exists())

    def test_utility_model_adds_bayesian_ucb_without_sparse_penalty(self):
        config = load_trip_config(CONFIG_PATH)
        enriched_df = pd.DataFrame(
            [
                {
                    "name": "Sparse Famous Landmark",
                    "city": "Los Angeles",
                    "latitude": 34.1,
                    "longitude": -118.3,
                    "category": "landmark",
                    "source_list": "curated_social_must_go",
                    "source_score": 8.0,
                    "yelp_rating": 0.0,
                    "yelp_review_count": 0,
                    "social_score": 0.95,
                    "must_go_weight": 0.90,
                    "social_must_go": True,
                    "corridor_fit": 0.5,
                    "detour_minutes": 20,
                    "data_confidence": 0.20,
                    "wikipedia_pageview_score": 0.5,
                },
                {
                    "name": "Well Covered Local Museum",
                    "city": "Santa Barbara",
                    "latitude": 34.42,
                    "longitude": -119.70,
                    "category": "museum",
                    "source_list": "openstreetmap_overpass|yelp",
                    "source_score": 5.0,
                    "yelp_rating": 4.5,
                    "yelp_review_count": 200,
                    "social_score": 0.2,
                    "must_go_weight": 0.0,
                    "social_must_go": False,
                    "corridor_fit": 0.9,
                    "detour_minutes": 3,
                    "data_confidence": 0.75,
                    "wikipedia_pageview_score": 0.2,
                },
            ]
        )

        scored_df, utility_scores, audit_df = apply_utility_models(enriched_df, None, config)

        self.assertIn("utility_bayesian_ucb", scored_df.columns)
        self.assertIn("utility_posterior_std", scored_df.columns)
        self.assertGreater(scored_df.loc[scored_df["name"].eq("Sparse Famous Landmark"), "final_poi_value"].iloc[0], 0)
        self.assertGreater(
            scored_df.loc[scored_df["name"].eq("Sparse Famous Landmark"), "utility_posterior_std"].iloc[0],
            0,
        )
        self.assertFalse(utility_scores.empty)
        self.assertEqual(audit_df["sparse_data_policy"].iloc[0], "uncertainty_bonus_not_value_penalty")

    def test_mmr_and_submodular_diversity_reduce_duplicate_categories(self):
        config = load_trip_config(CONFIG_PATH, overrides={"diversity": {"mmr_lambda": 0.55}})
        candidate_df = pd.DataFrame(
            [
                {
                    "name": "Museum A",
                    "city": "San Francisco",
                    "category": "museum",
                    "latitude": 37.1,
                    "longitude": -122.1,
                    "final_poi_value": 1.0,
                    "source_list": "yelp",
                },
                {
                    "name": "Museum B",
                    "city": "San Francisco",
                    "category": "museum",
                    "latitude": 37.11,
                    "longitude": -122.11,
                    "final_poi_value": 0.98,
                    "source_list": "yelp",
                },
                {
                    "name": "View A",
                    "city": "San Francisco",
                    "category": "viewpoint",
                    "latitude": 37.8,
                    "longitude": -122.4,
                    "final_poi_value": 0.92,
                    "source_list": "osm",
                },
                {
                    "name": "Park A",
                    "city": "San Francisco",
                    "category": "park",
                    "latitude": 37.7,
                    "longitude": -122.4,
                    "final_poi_value": 0.90,
                    "source_list": "osm",
                },
            ]
        )

        selected_df, log_df = mmr_select_candidates(candidate_df, config, candidate_size=3)

        pure_top = candidate_df.sort_values("final_poi_value", ascending=False).head(3)
        self.assertGreaterEqual(submodular_diversity(selected_df), submodular_diversity(pure_top))
        self.assertFalse(log_df.empty)
        self.assertIn("max_similarity_to_selected", log_df.columns)

    def test_trip_days_override_changes_hierarchical_allocation_target(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"trip_days": 5},
                "optimization": {"max_cities": 3, "max_days_per_base_city": 3},
            },
        )

        plans = candidate_plans(config, city_summary_frame())
        self.assertTrue(plans)
        self.assertTrue(all(sum(plan["days_by_city"].values()) == 5 for plan in plans))

        best_plan, plan_df = solve_hierarchical_trip_with_gurobi(config, city_summary_frame())
        self.assertEqual(sum(best_plan["days_by_city"].values()), 5)
        self.assertIn("solver_status", best_plan)
        self.assertFalse(plan_df.empty)

    def test_hierarchical_plan_allows_pass_through_cities_without_hotels(self):
        config = load_trip_config(CONFIG_PATH)

        best_plan, _ = solve_hierarchical_trip_with_gurobi(config, city_summary_frame())

        self.assertEqual(sum(best_plan["days_by_city"].values()), config.trip_days)
        self.assertTrue(best_plan["pass_through_cities"])
        self.assertLess(len(best_plan["overnight_bases"]), len(set(best_plan["city_sequence"])))
        for city in best_plan["pass_through_cities"]:
            self.assertNotIn(city, best_plan["overnight_bases"])

    def test_budget_uses_soft_budget_hard_cap_and_overnight_only_hotels(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {"trip_days": 4},
                "budget": {
                    "user_budget": 1000,
                    "hard_budget_multiplier": 1.10,
                    "food_cost_per_day": 10,
                    "attraction_cost_per_day": 5,
                    "local_transport_buffer_per_day": 0,
                },
            },
        )
        trip = {
            "days_by_city": {
                "San Francisco": 1,
                "Santa Cruz": 1,
                "Monterey": 1,
                "Los Angeles": 1,
            },
            "intercity_drive_minutes": 0,
        }
        hotel_catalog = pd.DataFrame(
            [
                {"city": "San Francisco", "nightly_price": 100},
                {"city": "Santa Cruz", "nightly_price": 120},
                {"city": "Monterey", "nightly_price": 140},
                {"city": "Los Angeles", "nightly_price": 500},
            ]
        )

        budget_df = estimate_budget_range(
            trip=trip,
            hotels_df=pd.DataFrame(),
            osm_city_hotel_catalog_df=hotel_catalog,
            output_dir=REPO_ROOT / "results" / "outputs",
            config=config,
            write_output=False,
        )

        hotel_expected = float(budget_df.loc[budget_df["component"] == "hotel", "expected"].iloc[0])
        soft_budget = float(budget_df.loc[budget_df["component"] == "soft_budget", "expected"].iloc[0])
        hard_budget = float(budget_df.loc[budget_df["component"] == "hard_budget", "expected"].iloc[0])

        self.assertEqual(hotel_expected, 360.0)
        self.assertEqual(soft_budget, 1000.0)
        self.assertGreaterEqual(hard_budget, 1100.0)

    def test_bandit_arms_and_social_weights_come_from_config(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "bandit": {"arms": ["balanced", "low_detour"]},
                "social": {
                    "social_score_weight": 2.20,
                    "must_go_bonus_weight": 1.70,
                    "corridor_fit_weight": 0.60,
                    "detour_penalty_weight": 0.02,
                },
            },
        )

        strategies = route_search_strategies_from_config(config)

        self.assertEqual(set(strategies), {"balanced", "low_detour"})
        self.assertAlmostEqual(strategies["balanced"]["social_weight"], 0.44)
        self.assertAlmostEqual(strategies["balanced"]["must_go_weight"], 0.40)
        self.assertAlmostEqual(strategies["balanced"]["corridor_weight"], 0.24)
        self.assertAlmostEqual(strategies["balanced"]["detour_penalty"], 0.02)

    def test_small_gurobi_route_oracle_returns_solver_or_fallback_status(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "time": {"daily_time_budget_minutes": 240},
                "optimization": {"max_pois_per_day": 2, "gurobi_time_limit_seconds": 5},
            },
        )
        candidate_df = pd.DataFrame(
            [
                {
                    "name": "Golden Gate Bridge",
                    "city": "San Francisco",
                    "latitude": 37.8199,
                    "longitude": -122.4783,
                    "category": "viewpoint",
                    "final_poi_value": 2.5,
                    "social_score": 0.98,
                },
                {
                    "name": "Ferry Building",
                    "city": "San Francisco",
                    "latitude": 37.7955,
                    "longitude": -122.3937,
                    "category": "market",
                    "final_poi_value": 1.4,
                    "social_score": 0.40,
                },
                {
                    "name": "Lombard Street",
                    "city": "San Francisco",
                    "latitude": 37.8021,
                    "longitude": -122.4187,
                    "category": "landmark",
                    "final_poi_value": 1.7,
                    "social_score": 0.55,
                },
            ]
        )

        result = solve_enriched_route_with_gurobi(candidate_df, config, candidate_size=3)

        self.assertIn("solver_status", result)
        self.assertLessEqual(len(result["selected_pois"]), 2)
        self.assertIn("runtime_seconds", result)
        self.assertIn("epsilon_detour_minutes", result)
        self.assertIn("diversity_score", result)

    def test_bandit_stress_benchmark_records_open_enriched_scope(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "bandit": {
                    "arms": ["balanced"],
                    "seeds": [7],
                    "episode_grid": [2],
                    "candidate_size_grid": [3],
                    "top_k_gurobi_repairs": 1,
                },
                "time": {"daily_time_budget_minutes": 240},
                "optimization": {"max_pois_per_day": 2, "gurobi_time_limit_seconds": 5},
            },
        )
        enriched_df = pd.DataFrame(
            [
                {
                    "name": "Golden Gate Bridge",
                    "city": "San Francisco",
                    "latitude": 37.8199,
                    "longitude": -122.4783,
                    "category": "viewpoint",
                    "final_poi_value": 2.5,
                    "social_score": 0.98,
                    "social_must_go": True,
                    "must_go_weight": 0.95,
                    "corridor_fit": 0.8,
                    "detour_minutes": 5,
                    "data_confidence": 0.7,
                },
                {
                    "name": "Ferry Building",
                    "city": "San Francisco",
                    "latitude": 37.7955,
                    "longitude": -122.3937,
                    "category": "market",
                    "final_poi_value": 1.4,
                    "social_score": 0.40,
                    "social_must_go": False,
                    "must_go_weight": 0.0,
                    "corridor_fit": 0.5,
                    "detour_minutes": 10,
                    "data_confidence": 0.6,
                },
                {
                    "name": "Santa Cruz Wharf",
                    "city": "Santa Cruz",
                    "latitude": 36.9595,
                    "longitude": -122.0251,
                    "category": "waterfront",
                    "final_poi_value": 1.8,
                    "social_score": 0.55,
                    "social_must_go": False,
                    "must_go_weight": 0.0,
                    "corridor_fit": 0.7,
                    "detour_minutes": 8,
                    "data_confidence": 0.6,
                },
            ]
        )
        budget_df = pd.DataFrame(
            [
                {"component": "buffered_total", "expected": 900.0},
                {"component": "soft_budget", "expected": 1000.0},
                {"component": "hard_budget", "expected": 1300.0},
            ]
        )
        arms_df = build_bandit_arms(
            hierarchical_candidates=[
                {
                    "gateway_start": "San Francisco",
                    "gateway_end": "Los Angeles",
                    "city_sequence": ["San Francisco", "Santa Cruz"],
                    "days_by_city": {"San Francisco": 1, "Santa Cruz": 1},
                    "intercity_drive_minutes": 120,
                    "objective": 1.2,
                }
            ],
            enriched_df=enriched_df,
            budget_df=budget_df,
            config=config,
        )

        runs_df, summary_df = run_bandit_gurobi_stress_benchmark(
            arms_df=arms_df,
            enriched_df=enriched_df,
            budget_df=budget_df,
            config=config,
        )

        self.assertFalse(runs_df.empty)
        self.assertFalse(summary_df.empty)
        self.assertTrue(runs_df["data_source_scope"].eq("open_enriched_catalog").all())
        self.assertIn("solver_status", runs_df.columns)


if __name__ == "__main__":
    unittest.main()
