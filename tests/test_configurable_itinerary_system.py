import sys
import unittest
from pathlib import Path

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
from itinerary_system.budget import estimate_budget_range
from itinerary_system.config import load_trip_config
from itinerary_system.data_enrichment import canonical_poi_columns
from itinerary_system.diversity import mmr_select_candidates, submodular_diversity
from itinerary_system.hierarchical_gurobi import candidate_plans, solve_hierarchical_trip_with_gurobi
from itinerary_system.route_gurobi_oracle import solve_enriched_route_with_gurobi
from itinerary_system.utility_model import apply_utility_models


CONFIG_PATH = REPO_ROOT / "configs" / "default_trip_config.yaml"


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
        self.assertTrue((config.to_frame()["parameter"] == "max_detour_minutes").any())
        self.assertEqual(config.get("utility", "method"), "bayesian_ucb")
        self.assertTrue(config.get("multi_objective", "use_epsilon_constraints"))
        self.assertEqual(config.get("diversity", "method"), "submodular_sqrt")
        self.assertIn("data_uncertainty", canonical_poi_columns())

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
                {"name": "Museum A", "city": "San Francisco", "category": "museum", "latitude": 37.1, "longitude": -122.1, "final_poi_value": 1.0, "source_list": "yelp"},
                {"name": "Museum B", "city": "San Francisco", "category": "museum", "latitude": 37.11, "longitude": -122.11, "final_poi_value": 0.98, "source_list": "yelp"},
                {"name": "View A", "city": "San Francisco", "category": "viewpoint", "latitude": 37.8, "longitude": -122.4, "final_poi_value": 0.92, "source_list": "osm"},
                {"name": "Park A", "city": "San Francisco", "category": "park", "latitude": 37.7, "longitude": -122.4, "final_poi_value": 0.90, "source_list": "osm"},
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
                {"name": "Golden Gate Bridge", "city": "San Francisco", "latitude": 37.8199, "longitude": -122.4783, "category": "viewpoint", "final_poi_value": 2.5, "social_score": 0.98},
                {"name": "Ferry Building", "city": "San Francisco", "latitude": 37.7955, "longitude": -122.3937, "category": "market", "final_poi_value": 1.4, "social_score": 0.40},
                {"name": "Lombard Street", "city": "San Francisco", "latitude": 37.8021, "longitude": -122.4187, "category": "landmark", "final_poi_value": 1.7, "social_score": 0.55},
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
                {"name": "Golden Gate Bridge", "city": "San Francisco", "latitude": 37.8199, "longitude": -122.4783, "category": "viewpoint", "final_poi_value": 2.5, "social_score": 0.98, "social_must_go": True, "must_go_weight": 0.95, "corridor_fit": 0.8, "detour_minutes": 5, "data_confidence": 0.7},
                {"name": "Ferry Building", "city": "San Francisco", "latitude": 37.7955, "longitude": -122.3937, "category": "market", "final_poi_value": 1.4, "social_score": 0.40, "social_must_go": False, "must_go_weight": 0.0, "corridor_fit": 0.5, "detour_minutes": 10, "data_confidence": 0.6},
                {"name": "Santa Cruz Wharf", "city": "Santa Cruz", "latitude": 36.9595, "longitude": -122.0251, "category": "waterfront", "final_poi_value": 1.8, "social_score": 0.55, "social_must_go": False, "must_go_weight": 0.0, "corridor_fit": 0.7, "detour_minutes": 8, "data_confidence": 0.6},
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
