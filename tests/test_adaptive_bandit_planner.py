import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import adaptive_bandit_planner
import gurobi_itinerary_solver


def build_synthetic_notebook_context():
    top100 = pd.DataFrame(
        {
            "name": ["Pier", "Museum", "Garden", "Aquarium", "Beach"],
            "utility": [9.0, 8.5, 7.8, 8.8, 7.2],
            "waiting_final": [20.0, 35.0, 15.0, 30.0, 10.0],
            "visit_duration_sim": [60.0, 75.0, 50.0, 70.0, 45.0],
            "cost": [15.0, 22.0, 10.0, 25.0, 0.0],
            "content_theme": ["landmark", "museum", "park", "zoo_aquarium", "beach"],
            "type": ["landmark", "museum", "park", "zoo_aquarium", "beach"],
            "city": ["Santa Barbara"] * 5,
        }
    )
    travel_time_matrix = np.asarray(
        [
            [0.0, 12.0, 10.0, 14.0, 16.0],
            [12.0, 0.0, 8.0, 9.0, 13.0],
            [10.0, 8.0, 0.0, 11.0, 7.0],
            [14.0, 9.0, 11.0, 0.0, 12.0],
            [16.0, 13.0, 7.0, 12.0, 0.0],
        ],
        dtype=float,
    )
    hotel_to_attr_time = np.asarray([[8.0, 10.0, 6.0, 11.0, 9.0]], dtype=float)
    hotel_to_hotel_time = np.asarray([[0.0]], dtype=float)
    hotels_df = pd.DataFrame(
        {
            "name": ["Central Hotel"],
            "nightly_price": [110.0],
            "rating_score": [4.5],
            "experience_score": [4.2],
        }
    )

    route_details = {
        "balanced": {
            "selected_by_day": {0: [0, 2], 1: [1], 2: [3]},
            "travel_edges_by_day": {0: [(0, 2)], 1: [], 2: []},
            "start_edges_by_day": {0: [(0, 0)], 1: [(0, 1)], 2: [(0, 3)]},
            "end_edges_by_day": {0: [(2, 0)], 1: [(1, 0)], 2: [(3, 0)]},
            "hotels_by_day": {0: 0, 1: 0, 2: 0},
            "transitions_by_day": {0: [(0, 0)], 1: [(0, 0)]},
            "day_used_by_day": {0: 1, 1: 1, 2: 1},
            "content_mix": {"landmark": 1, "park": 1, "museum": 1, "zoo_aquarium": 1},
            "status": "OPTIMAL",
        }
    }
    routes = {"balanced": [0, 1, 2, 3]}
    utility_norm = gurobi_itinerary_solver.safe_minmax(top100["utility"].to_numpy())
    waiting_norm = gurobi_itinerary_solver.safe_minmax(top100["waiting_final"].to_numpy())

    return {
        "top100_with_waiting_time": top100,
        "travel_time_matrix": travel_time_matrix,
        "hotel_to_attr_time": hotel_to_attr_time,
        "hotel_to_hotel_time": hotel_to_hotel_time,
        "hotels_df": hotels_df,
        "route_details": route_details,
        "routes": routes,
        "TOURIST_PROFILES": {"balanced": dict(gurobi_itinerary_solver.DEFAULT_TOURIST_PROFILES["balanced"])},
        "SOLVER_SETTINGS": {
            "daily_time_budget": 360.0,
            "total_budget": 350.0,
            "time_limit_seconds": 10.0,
            "mip_gap_target": 0.1,
            "max_travel": 90.0,
        },
        "MODEL_SETTINGS": {
            "daily_time_budget": 360.0,
            "total_budget": 350.0,
            "gamma": 0.02,
            "num_days": 3,
        },
        "K": 3,
        "avg_speed": 20.0,
        "distance_matrix": travel_time_matrix * 20.0 / 60.0,
        "hotel_to_attr": hotel_to_attr_time * 20.0 / 60.0,
        "hotel_to_hotel": hotel_to_hotel_time,
        "utility_norm": utility_norm,
        "waiting_time_norm": waiting_norm,
    }


class AdaptiveBanditPlannerTests(unittest.TestCase):
    def setUp(self):
        self.context = build_synthetic_notebook_context()
        self.state = adaptive_bandit_planner.build_adaptive_bandit_context(
            self.context,
            profile_name="balanced",
            seed=7,
        )

    def test_scenario_generation_is_deterministic(self):
        scenario_a = adaptive_bandit_planner._build_episode_scenario(
            self.state,
            scenario_mode="weather_shock",
            episode_seed=123,
        )
        scenario_b = adaptive_bandit_planner._build_episode_scenario(
            self.state,
            scenario_mode="weather_shock",
            episode_seed=123,
        )
        self.assertEqual(scenario_a.weather_multipliers_by_day, scenario_b.weather_multipliers_by_day)
        self.assertEqual(scenario_a.congestion_multipliers_by_day, scenario_b.congestion_multipliers_by_day)

    def test_reward_terms_match_expected_formula_without_noise(self):
        scenario = adaptive_bandit_planner.ScenarioConfig(
            scenario_mode="nominal",
            weather_multipliers_by_day=[1.0, 1.0, 1.0],
            congestion_multipliers_by_day=[1.0, 1.0, 1.0],
            wait_noise_std=0.0,
            travel_noise_std=0.0,
            utility_noise_std=0.0,
        )
        rng = np.random.default_rng(0)
        outcome = adaptive_bandit_planner._simulate_stop_outcome(
            self.state,
            scenario,
            day_idx=0,
            current_hotel_idx=0,
            current_attr_idx=None,
            attr_idx=0,
            theme_counts={},
            rng=rng,
        )
        expected_reward = (
            float(self.state.utility_norm[0])
            - float(self.state.tourist_profile["alpha"]) * (self.state.waiting_raw[0] / np.max(self.state.waiting_raw))
            - float(self.state.tourist_profile["beta"])
            * (self.state.hotel_to_attr_time[0, 0] / self.state.travel_scale)
            - float(self.state.model_settings["gamma"]) * float(self.state.cost_penalty[0])
            + float(self.state.tourist_profile["attraction_bonus"]) * 1.0
        )
        self.assertAlmostEqual(outcome["realized_reward"], expected_reward, places=6)

    def test_candidate_pool_respects_remaining_time_and_budget(self):
        candidates = gurobi_itinerary_solver.generate_receding_horizon_candidate_pool(
            self.context,
            profile_name="balanced",
            current_hotel_idx=0,
            current_attr_idx=None,
            remaining_time=130.0,
            remaining_budget=20.0,
            visited_attractions={1, 3},
            visited_theme_counts={"museum": 1},
            candidate_pool_size=3,
            solver_overrides={"backend": "heuristic"},
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertLessEqual(candidate["predicted_total_cost"], 20.0 + 1e-9)
            self.assertLessEqual(candidate["predicted_total_travel_time"], 130.0 + 1e-9)
            self.assertNotIn(1, candidate["selected_indices"])
            self.assertNotIn(3, candidate["selected_indices"])

    def test_linear_thompson_sampling_updates_matching_features(self):
        model = adaptive_bandit_planner.LinearThompsonSamplingModel(feature_dim=3)
        good_feature = np.asarray([1.0, 1.0, 0.0])
        other_feature = np.asarray([0.0, 0.0, 1.0])
        baseline_good = float(model.predict_scores(np.vstack([good_feature]))[0])
        baseline_other = float(model.predict_scores(np.vstack([other_feature]))[0])
        model.update(good_feature, reward=1.5)
        updated_good = float(model.predict_scores(np.vstack([good_feature]))[0])
        updated_other = float(model.predict_scores(np.vstack([other_feature]))[0])
        self.assertGreater(updated_good, baseline_good)
        self.assertEqual(baseline_other, updated_other)

    def test_ccb_subset_features_capture_aggregate_metrics(self):
        candidate = {
            "sequence": [0, 2],
            "predicted_total_travel_time": 30.0,
            "predicted_diversity_score": 1.0,
        }
        scenario = adaptive_bandit_planner._build_episode_scenario(
            self.state,
            scenario_mode="nominal",
            episode_seed=77,
        )
        features = adaptive_bandit_planner._build_subset_feature_vector(
            self.state,
            candidate,
            remaining_time=180.0,
            remaining_budget=200.0,
            visited_attractions=set(),
            theme_counts={},
            scenario=scenario,
            day_idx=0,
        )
        expected_utility_sum = float(self.state.utility_norm[[0, 2]].sum())
        expected_diversity = 1.0
        self.assertAlmostEqual(features[1], expected_utility_sum, places=6)
        self.assertAlmostEqual(features[5], expected_diversity, places=6)

    def test_integration_smoke_benchmark_runs_all_methods(self):
        outputs = adaptive_bandit_planner.run_adaptive_bandit_benchmark(
            self.context,
            profile_name="balanced",
            methods=("static", "greedy", "cts", "ccb"),
            n_episodes=1,
            scenario_modes=("nominal",),
            candidate_pool_size=3,
            seed=19,
            solver_overrides={"backend": "heuristic"},
        )
        summary = outputs["summary"]
        episode_log = outputs["episode_log"]
        step_log = outputs["step_log"]
        self.assertEqual(set(summary["method"]), {"static", "greedy", "cts", "ccb"})
        self.assertEqual(set(episode_log["method"]), {"static", "greedy", "cts", "ccb"})
        self.assertFalse(step_log.empty)
        self.assertIn("adaptation_gain_vs_static_mean", summary.columns)


if __name__ == "__main__":
    unittest.main()
