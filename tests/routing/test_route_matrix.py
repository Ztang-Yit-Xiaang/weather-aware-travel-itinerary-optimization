import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from itinerary_system.config import load_trip_config
from itinerary_system.hierarchical_gurobi import candidate_plans
from itinerary_system.multi_objective_route import solve_multi_objective_route
from itinerary_system.routing import (
    RouteMatrix,
    RouteMatrixCell,
    RouteMatrixCellMissing,
    RouteMatrixMissing,
    RouteMatrixNotPublicationEligible,
    SolverRouteMatrixAdapter,
    geodesic_fallback_matrix,
    load_route_matrix_from_cache,
    route_anchor_key,
    route_result_for_sequence,
    validate_route_matrix,
    write_validated_route_matrix_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default_trip_config.yaml"


def road_cell(origin_id: str, destination_id: str, minutes: float) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin_id,
        destination_id=destination_id,
        distance_m=minutes * 1000.0,
        duration_s=minutes * 60.0,
        route_leg_id=f"road_{origin_id}_{destination_id}",
        road_validated=True,
        fallback_used=False,
        provider="unit_test_road_cache",
        context_snapshot_id="context_unit",
        geometry=((0.0, 0.0), (0.1, 0.1)),
        routing_status="cached_osrm_validated",
        geometry_source="validated_osrm_cache",
        distance_source="validated_osrm_cache",
        duration_source="validated_osrm_cache",
    )


def matrix_from_minutes(minutes_by_pair: dict[tuple[str, str], float]) -> RouteMatrix:
    cells = {
        (route_anchor_key(origin), route_anchor_key(destination)): road_cell(origin, destination, minutes)
        for (origin, destination), minutes in minutes_by_pair.items()
    }
    return RouteMatrix(
        matrix_id="unit_route_matrix",
        context_snapshot_id="context_unit",
        entity_ids=(),
        cells=cells,
    )


class RouteMatrixTests(unittest.TestCase):
    def test_empty_matrix_rejects_solver_usage(self):
        matrix = RouteMatrix(matrix_id="empty", context_snapshot_id="context_unit", entity_ids=(), cells={})
        adapter = SolverRouteMatrixAdapter(matrix, mode="publication")

        with self.assertRaises(RouteMatrixMissing):
            adapter.assert_publication_ready()

    def test_missing_cell_raises_clear_error(self):
        matrix = matrix_from_minutes({("a", "b"): 12.0})
        adapter = SolverRouteMatrixAdapter(matrix, mode="publication")

        with self.assertRaises(RouteMatrixCellMissing):
            adapter.travel_minutes("b", "a")

    def test_geodesic_fallback_matrix_is_demo_only(self):
        matrix = geodesic_fallback_matrix({"a": (0.0, 0.0), "b": (0.0, 0.1)})

        demo = SolverRouteMatrixAdapter(matrix, mode="demo")
        self.assertGreater(demo.travel_minutes("a", "b"), 0.0)
        route = demo.route_result(("a", "b"))
        self.assertTrue(route.fallback_used)
        self.assertFalse(route.road_validated)
        self.assertFalse(route.evaluation_eligible)

        strict = SolverRouteMatrixAdapter(matrix, mode="publication")
        with self.assertRaises(RouteMatrixNotPublicationEligible):
            strict.travel_minutes("a", "b")

    def test_validated_matrix_route_result_is_evaluation_eligible(self):
        matrix = matrix_from_minutes({("start", "poi_a"): 10.0, ("poi_a", "end"): 20.0})

        route = route_result_for_sequence(
            matrix,
            ("start", "poi_a", "end"),
            strict=True,
            solver_feasible=True,
            schedule_feasible=True,
            dataset_snapshot_valid=True,
        )

        self.assertTrue(route.road_validated)
        self.assertFalse(route.fallback_used)
        self.assertTrue(route.evaluation_eligible)
        self.assertEqual(route.total_duration_s, 1800.0)

    def test_route_matrix_validation_reports_missing_and_fallback_cells(self):
        matrix = geodesic_fallback_matrix({"a": (0.0, 0.0), "b": (0.0, 0.1)})

        report = validate_route_matrix(matrix, required_sequences=(("a", "b", "c"),), require_publication_ready=True)

        self.assertFalse(report.publication_ready)
        self.assertEqual(report.required_leg_count, 2)
        self.assertEqual(report.present_leg_count, 1)
        self.assertEqual(report.fallback_leg_count, 1)
        self.assertEqual(report.missing_leg_count, 1)
        self.assertIn("route_matrix_not_publication_ready", report.errors)

    def test_write_validated_route_matrix_artifacts(self):
        matrix = matrix_from_minutes({("start", "poi_a"): 10.0, ("poi_a", "end"): 20.0})

        with tempfile.TemporaryDirectory() as temp_dir:
            report = write_validated_route_matrix_artifacts(
                matrix,
                temp_dir,
                required_sequences=(("start", "poi_a", "end"),),
                require_publication_ready=True,
            )
            matrix_path = Path(temp_dir) / "production_validated_route_matrix.csv"
            report_path = Path(temp_dir) / "production_validated_route_matrix_report.json"

        self.assertTrue(report.publication_ready)
        self.assertEqual(report.road_validated_leg_count, 2)
        self.assertTrue(matrix_path.exists())
        self.assertTrue(report_path.exists())

    def test_load_route_matrix_from_cache_reads_validated_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_options.csv"
            pd.DataFrame(
                [
                    {
                        "route_option_id": "road_start_poi_a",
                        "context_snapshot_id": "context_unit",
                        "origin_id": "start",
                        "destination_id": "poi_a",
                        "geometry": "[[0.0,0.0],[0.1,0.1]]",
                        "distance_m": 1000.0,
                        "duration_s": 600.0,
                        "routing_source": "unit_cache",
                        "road_validated": True,
                        "fallback_used": False,
                        "geometry_source": "validated_osrm_cache",
                        "distance_source": "validated_osrm_cache",
                        "duration_source": "validated_osrm_cache",
                    }
                ]
            ).to_csv(path, index=False)

            matrix = load_route_matrix_from_cache(path, "context_unit")

        self.assertEqual(matrix.duration_minutes("start", "poi_a", strict=True), 10.0)
        self.assertEqual(matrix.leg("start", "poi_a", strict=True).provider, "unit_cache")

    def test_build_validated_route_matrix_script_requires_publication_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "route_options.csv"
            pd.DataFrame(
                [
                    {
                        "route_option_id": "fallback_start_poi_a",
                        "context_snapshot_id": "context_unit",
                        "origin_id": "start",
                        "destination_id": "poi_a",
                        "geometry": "[[0.0,0.0],[0.1,0.1]]",
                        "distance_m": 1000.0,
                        "duration_s": 600.0,
                        "routing_source": "curated_straight_demo",
                        "road_validated": False,
                        "fallback_used": True,
                        "geometry_source": "straight_waypoints",
                        "distance_source": "geodesic_proxy",
                        "duration_source": "geodesic_speed_proxy",
                    }
                ]
            ).to_csv(input_path, index=False)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_validated_route_matrix.py"),
                "--input",
                str(input_path),
                "--context-snapshot-id",
                "context_unit",
                "--output-dir",
                temp_dir,
                "--required-sequence",
                "start,poi_a",
                "--require-publication-ready",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED route matrix build", result.stdout)

    def test_publication_route_solver_requires_matrix(self):
        config = load_trip_config(CONFIG_PATH)
        candidates = pd.DataFrame(
            [
                {
                    "poi_id": "poi_a",
                    "name": "POI A",
                    "latitude": 0.0,
                    "longitude": 0.1,
                    "category": "park",
                    "final_poi_value": 50.0,
                    "social_score": 0.0,
                    "weather_risk": 0.0,
                }
            ]
        )

        with self.assertRaises(RouteMatrixMissing):
            solve_multi_objective_route(candidates, config, candidate_size=1, routing_mode="publication")

    def test_publication_route_solver_uses_matrix_totals(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "optimization": {"max_pois_per_day": 1},
                "multi_objective": {"use_epsilon_constraints": False, "secondary_travel_penalty": 0.0},
                "time": {"daily_time_budget_minutes": 720},
            },
        )
        candidates = pd.DataFrame(
            [
                {
                    "poi_id": "poi_a",
                    "name": "POI A",
                    "city": "Unit City",
                    "latitude": 10.0,
                    "longitude": 10.0,
                    "category": "park",
                    "final_poi_value": 100.0,
                    "social_score": 0.0,
                    "weather_risk": 0.0,
                    "detour_minutes": 0.0,
                }
            ]
        )
        matrix = matrix_from_minutes(
            {
                ("start_depot", "poi_a"): 10.0,
                ("poi_a", "end_depot"): 20.0,
                ("start_depot", "end_depot"): 90.0,
            }
        )

        result = solve_multi_objective_route(
            candidates,
            config,
            start_depot=(0.0, 0.0),
            end_depot=(1.0, 1.0),
            candidate_size=1,
            route_matrix=matrix,
            routing_mode="publication",
        )

        self.assertEqual(result["selected_pois"], ["POI A"])
        self.assertEqual(result["route_duration_source"], "route_matrix")
        self.assertTrue(result["route_road_validated"])
        self.assertFalse(result["route_fallback_used"])
        self.assertAlmostEqual(result["total_travel_minutes"], 30.0)
        self.assertEqual(result["route_result_total_duration_s"], 1800.0)

    def test_hierarchical_planner_accepts_injected_route_matrix(self):
        config = load_trip_config(
            CONFIG_PATH,
            overrides={
                "trip": {
                    "start_city_options": ["San Francisco"],
                    "end_city_options": ["Los Angeles"],
                    "trip_days": 5,
                },
                "nature": {"enabled": False},
                "optimization": {"max_cities": 3, "max_days_per_base_city": 3},
            },
        )
        city_summary = pd.DataFrame(
            [
                {"city": "San Francisco", "city_value_score": 0.95, "data_uncertainty": 0.10},
                {"city": "Santa Cruz", "city_value_score": 0.70, "data_uncertainty": 0.25},
                {"city": "Monterey", "city_value_score": 0.76, "data_uncertainty": 0.20},
                {"city": "San Luis Obispo", "city_value_score": 0.66, "data_uncertainty": 0.30},
                {"city": "Santa Barbara", "city_value_score": 0.88, "data_uncertainty": 0.15},
                {"city": "Los Angeles", "city_value_score": 0.98, "data_uncertainty": 0.12},
            ]
        )
        route_minutes = {
            ("San Francisco", "Santa Cruz"): 10.0,
            ("Santa Cruz", "Monterey"): 11.0,
            ("Monterey", "San Luis Obispo"): 12.0,
            ("San Luis Obispo", "Santa Barbara"): 13.0,
            ("Santa Barbara", "Los Angeles"): 14.0,
        }
        matrix = matrix_from_minutes(route_minutes)

        plans = candidate_plans(
            config,
            city_summary,
            route_matrix=matrix,
            routing_mode="publication",
        )

        self.assertTrue(plans)
        self.assertTrue(all(plan["route_duration_source"] == "route_matrix" for plan in plans))
        self.assertTrue(all(plan["route_road_validated"] for plan in plans))
        self.assertAlmostEqual(float(plans[0]["intercity_drive_minutes"]), sum(route_minutes.values()))


if __name__ == "__main__":
    unittest.main()
