import tempfile
import unittest
from pathlib import Path

import pandas as pd

from itinerary_system.config import load_trip_config
from itinerary_system.utility_model import (
    SourceAblationReport,
    build_signal_matrix,
    build_source_masks,
    score_masked_weighted_utility,
    write_source_ablation_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default_trip_config.yaml"


def utility_config():
    return load_trip_config(
        CONFIG_PATH,
        overrides={
            "utility": {
                "method": "mcda_weighted",
                "missing_source_fallback_utility": 0.15,
                "mcda_weights": {
                    "base_score": 0.25,
                    "yelp_signal": 0.25,
                    "social_signal": 0.10,
                    "must_go_signal": 0.05,
                    "corridor_fit": 0.15,
                    "wikipedia_signal": 0.10,
                    "data_confidence": 0.10,
                    "weather_safety": 0.05,
                    "low_detour": 0.05,
                },
            }
        },
    )


class UtilityMissingnessTests(unittest.TestCase):
    def test_missing_yelp_is_not_scored_as_bad_yelp(self):
        config = utility_config()
        frame = pd.DataFrame(
            [
                {
                    "poi_id": "missing_yelp",
                    "name": "Missing Yelp",
                    "city": "A",
                    "source_list": "curated_seed|wikipedia",
                    "source_score": 8.0,
                    "social_score": 0.5,
                    "corridor_fit": 0.8,
                    "route_fit": 0.8,
                    "detour_minutes": 10.0,
                    "wikipedia_title": "Same",
                    "wikipedia_pageview_score": 0.5,
                    "weather_risk": 0.2,
                },
                {
                    "poi_id": "bad_yelp",
                    "name": "Bad Yelp",
                    "city": "A",
                    "source_list": "curated_seed|wikipedia|yelp",
                    "source_score": 8.0,
                    "yelp_rating": 1.0,
                    "yelp_review_count": 10,
                    "social_score": 0.5,
                    "corridor_fit": 0.8,
                    "route_fit": 0.8,
                    "detour_minutes": 10.0,
                    "wikipedia_title": "Same",
                    "wikipedia_pageview_score": 0.5,
                    "weather_risk": 0.2,
                },
                {
                    "poi_id": "good_yelp",
                    "name": "Good Yelp",
                    "city": "A",
                    "source_list": "curated_seed|wikipedia|yelp",
                    "source_score": 8.0,
                    "yelp_rating": 5.0,
                    "yelp_review_count": 100,
                    "social_score": 0.5,
                    "corridor_fit": 0.8,
                    "route_fit": 0.8,
                    "detour_minutes": 10.0,
                    "wikipedia_title": "Same",
                    "wikipedia_pageview_score": 0.5,
                    "weather_risk": 0.2,
                },
            ]
        )

        signals = build_signal_matrix(frame, config)
        scores = score_masked_weighted_utility(signals, config)

        self.assertFalse(bool(signals.loc[0, "has_yelp"]))
        self.assertTrue(bool(signals.loc[1, "has_yelp"]))
        self.assertGreater(float(scores.loc[0]), float(scores.loc[1]))
        self.assertGreater(float(scores.loc[2]), float(scores.loc[0]))

    def test_identical_non_yelp_pois_are_not_split_by_missing_yelp(self):
        config = utility_config()
        frame = pd.DataFrame(
            [
                {
                    "poi_id": "a",
                    "name": "A",
                    "source_list": "curated_seed|wikipedia",
                    "source_score": 7.0,
                    "wikipedia_title": "A",
                    "wikipedia_pageview_score": 0.3,
                    "weather_risk": 0.1,
                    "route_fit": 0.7,
                    "corridor_fit": 0.7,
                    "detour_minutes": 5.0,
                },
                {
                    "poi_id": "b",
                    "name": "B",
                    "source_list": "curated_seed|wikipedia",
                    "source_score": 7.0,
                    "wikipedia_title": "B",
                    "wikipedia_pageview_score": 0.3,
                    "weather_risk": 0.1,
                    "route_fit": 0.7,
                    "corridor_fit": 0.7,
                    "detour_minutes": 5.0,
                },
            ]
        )

        signals = build_signal_matrix(frame, config)
        scores = score_masked_weighted_utility(signals, config)

        self.assertAlmostEqual(float(scores.loc[0]), float(scores.loc[1]))
        self.assertTrue(signals["missing_source_list"].str.contains("yelp").all())

    def test_all_source_missing_row_uses_fallback_and_low_coverage(self):
        config = utility_config()
        frame = pd.DataFrame([{"poi_id": "unknown", "name": "Unknown"}])

        masks = build_source_masks(frame)
        signals = build_signal_matrix(frame, config)
        scores = score_masked_weighted_utility(signals, config)

        self.assertEqual(float(masks["source_coverage_score"].iloc[0]), 0.0)
        self.assertEqual(float(signals["active_source_weight"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(scores.iloc[0]), 0.15)
        self.assertEqual(set(signals["missing_source_list"].iloc[0].split("|")), {
            "osm",
            "yelp",
            "curated",
            "wikidata",
            "wikipedia",
            "weather",
            "route",
        })

    def test_coverage_confidence_and_uncertainty_remain_separate(self):
        config = utility_config()
        frame = pd.DataFrame(
            [
                {
                    "poi_id": "sparse",
                    "name": "Sparse",
                    "source_list": "curated_seed",
                    "source_score": 5.0,
                    "source_coverage_score": 0.25,
                    "model_uncertainty": 0.7,
                }
            ]
        )

        signals = build_signal_matrix(frame, config)

        self.assertAlmostEqual(float(signals["source_coverage_score"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(signals["data_confidence"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(signals["model_uncertainty"].iloc[0]), 0.7)
        self.assertAlmostEqual(float(signals["data_uncertainty"].iloc[0]), 0.7)

    def test_source_ablation_audit_is_deterministic(self):
        config = utility_config()
        frame = pd.DataFrame(
            [
                {
                    "poi_id": "poi_a",
                    "name": "POI A",
                    "source_list": "curated_seed|yelp|wikipedia",
                    "source_score": 8.0,
                    "yelp_rating": 4.0,
                    "yelp_review_count": 30,
                    "wikipedia_title": "POI A",
                    "wikipedia_pageview_score": 0.6,
                    "weather_risk": 0.2,
                    "route_fit": 0.7,
                    "corridor_fit": 0.7,
                    "detour_minutes": 8.0,
                }
            ]
        )

        first = SourceAblationReport.compute(frame, config)
        second = SourceAblationReport.compute(frame, config)
        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_source_ablation_audit(frame, Path(temp_dir), config)

        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first, written)
        self.assertEqual(first["source_family"].tolist(), sorted(first["source_family"].tolist()))


if __name__ == "__main__":
    unittest.main()
