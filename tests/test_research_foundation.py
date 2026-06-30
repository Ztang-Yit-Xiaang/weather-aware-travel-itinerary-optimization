import hashlib
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.artifact_metadata import artifact_metadata_matches, write_artifact_metadata
from itinerary_system.config import load_trip_config
from itinerary_system.data import load_dataset_bundle, validate_dataset_bundle
from itinerary_system.experiment_runner import PRODUCTION_ARTIFACT_FILES
from itinerary_system.phase0_exporter import PHASE0_ARTIFACT_FILES, write_phase0_research_artifacts
from itinerary_system.research_artifacts import PlanArtifact, PlannerRun, evaluate_phase0_plan
from itinerary_system.routing import (
    LEGACY_ROUTE_CACHE_AUDIT_FILENAME,
    OSRM_LOCAL_BASE_URL,
    OSRM_PUBLIC_BASE_URL,
    ROAD_ROUTE_CACHE_AUDIT_FILENAME,
    ROAD_ROUTE_CACHE_FILENAME,
    ROAD_ROUTE_REQUESTS_FILENAME,
    audit_legacy_route_cache,
    build_road_route_cache_from_artifacts,
    is_public_osrm_base_url,
    osrm_cache_key,
)
from itinerary_system.routing import RouteLegResult, RouteResult
from itinerary_system.utility_model import build_signal_matrix

from scripts.summarize_phase0_readiness import load_phase0_readiness, readiness_markdown
from scripts.check_route_source import check_route_source

CONFIG_PATH = REPO_ROOT / "configs" / "default_trip_config.yaml"


@contextmanager
def temporary_directory():
    base_dir = REPO_ROOT / "results" / "outputs" / "pytest_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_route() -> RouteResult:
    leg = RouteLegResult(
        origin_id="poi_golden_gate_bridge",
        destination_id="poi_ferry_building",
        geometry=((37.8199, -122.4783), (37.7955, -122.3937)),
        distance_m=9200.0,
        duration_s=1440.0,
        routing_status="ok",
        provider="osrm",
        geometry_source="osrm_road_geometry",
        distance_source="osrm_route_distance",
        duration_source="osrm_route_duration",
        road_validated=True,
    )
    return RouteResult(
        route_id="route_validated_demo",
        legs=(leg,),
        solver_feasible=True,
        schedule_feasible=True,
        dataset_snapshot_valid=True,
    )


def fallback_route() -> RouteResult:
    leg = RouteLegResult.approximate_geodesic(
        origin_id="poi_golden_gate_bridge",
        destination_id="poi_ferry_building",
        geometry=((37.8199, -122.4783), (37.7955, -122.3937)),
        distance_m=7600.0,
        duration_s=1200.0,
    )
    return RouteResult(
        route_id="route_fallback_demo",
        legs=(leg,),
        solver_feasible=True,
        schedule_feasible=True,
        dataset_snapshot_valid=True,
    )


def planner_run() -> PlannerRun:
    return PlannerRun(
        run_id="run_phase0_demo",
        planning_request_id="request_california_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        planner_specification_id="phase0_repair_v1",
        method_requested="weather_aware_repair",
        method_executed="weather_aware_repair",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        solver_backend="unit-test",
        solver_version="0",
        result_plan_id="plan_phase0_demo",
    )


def plan_artifact() -> PlanArtifact:
    return PlanArtifact(
        plan_id="plan_phase0_demo",
        source_run_id="run_phase0_demo",
        planning_request_id="request_california_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"poi_id": "poi_golden_gate_bridge", "day": 1},
            {"poi_id": "poi_ferry_building", "day": 1},
        ),
        sequence=("poi_golden_gate_bridge", "poi_ferry_building"),
        created_at="2026-06-28T00:00:00+00:00",
    )


def phase0_method_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "hierarchical_bandit_gurobi_repair",
                "method_display_name": "Hierarchical + Bandit + Small Gurobi Repair",
                "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                "trip_days": 7,
                "status": "FEASIBLE",
                "local_route_solver": "small_gurobi",
                "objective": 12.5,
                "solve_seconds": 0.2,
                "mip_gap": 0.0,
                "total_utility": 2.1,
                "total_travel_time": 45.0,
                "total_travel_distance_km": 14.0,
                "selected_attractions": 2,
            }
        ]
    )


def phase0_route_stops_frame() -> pd.DataFrame:
    common = {
        "comparison_type": "method",
        "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
        "method": "hierarchical_bandit_gurobi_repair",
        "trip_days": 7,
        "day": 1,
        "route_start_name": "SFO",
        "route_start_latitude": 37.6213,
        "route_start_longitude": -122.3790,
        "route_end_name": "Optimizer Hotel",
        "route_end_latitude": 37.78,
        "route_end_longitude": -122.42,
        "source_list": "curated_seed",
        "source_confidence": 0.7,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "stop_order": 1,
                "attraction_name": "Golden Gate Bridge",
                "city": "San Francisco",
                "latitude": 37.8199,
                "longitude": -122.4783,
            },
            {
                **common,
                "stop_order": 2,
                "attraction_name": "Ferry Building",
                "city": "San Francisco",
                "latitude": 37.7955,
                "longitude": -122.3937,
            },
        ]
    )


def phase0_road_route_cache_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "origin_label": "SFO",
                "destination_label": "Golden Gate Bridge",
                "geometry": json.dumps([[37.6213, -122.3790], [37.73, -122.45], [37.8199, -122.4783]]),
                "distance_m": 31000,
                "duration_s": 2400,
                "provider": "local_osrm_cache",
                "routing_profile": "driving",
                "routing_status": "ok",
                "geometry_source": "validated_osrm_cache",
                "distance_source": "validated_osrm_cache",
                "duration_source": "validated_osrm_cache",
                "road_validated": True,
            },
            {
                "origin_label": "Golden Gate Bridge",
                "destination_label": "Ferry Building",
                "geometry": json.dumps([[37.8199, -122.4783], [37.807, -122.42], [37.7955, -122.3937]]),
                "distance_m": 9200,
                "duration_s": 1440,
                "provider": "local_osrm_cache",
                "routing_profile": "driving",
                "routing_status": "ok",
                "geometry_source": "validated_osrm_cache",
                "distance_source": "validated_osrm_cache",
                "duration_source": "validated_osrm_cache",
                "road_validated": True,
            },
            {
                "origin_label": "Ferry Building",
                "destination_label": "Optimizer Hotel",
                "geometry": json.dumps([[37.7955, -122.3937], [37.79, -122.405], [37.78, -122.42]]),
                "distance_m": 2800,
                "duration_s": 540,
                "provider": "local_osrm_cache",
                "routing_profile": "driving",
                "routing_status": "ok",
                "geometry_source": "validated_osrm_cache",
                "distance_source": "validated_osrm_cache",
                "duration_source": "validated_osrm_cache",
                "road_validated": True,
            },
        ]
    )


def write_phase0_osrm_cache_files(output_dir: Path) -> None:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    legs = [
        ("SFO", (37.6213, -122.3790), "Golden Gate Bridge", (37.8199, -122.4783), 31000, 2400),
        ("Golden Gate Bridge", (37.8199, -122.4783), "Ferry Building", (37.7955, -122.3937), 9200, 1440),
        ("Ferry Building", (37.7955, -122.3937), "Optimizer Hotel", (37.78, -122.42), 2800, 540),
    ]
    for _origin, left, _destination, right, distance_m, duration_s in legs:
        points = [left, right]
        key = osrm_cache_key(points)
        (cache_dir / f"open_osrm_route_{key}.json").write_text(
            json.dumps(
                {
                    "status": "osrm_live",
                    "latlon_geometry": [[left[0], left[1]], [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2], [right[0], right[1]]],
                    "raw": {
                        "routes": [
                            {
                                "distance": distance_m,
                                "duration": duration_s,
                                "geometry": {
                                    "coordinates": [[left[1], left[0]], [right[1], right[0]]],
                                },
                            }
                        ]
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class ResearchFoundationTests(unittest.TestCase):
    def test_clean_clone_snapshot_loads_and_gates_final_comparison(self):
        bundle = load_dataset_bundle(root=REPO_ROOT)
        report = validate_dataset_bundle(bundle)

        self.assertEqual(bundle.catalog_snapshot_id, "california_v1")
        self.assertEqual(bundle.context_snapshot_id, "context_static_demo_2026_06")
        self.assertEqual(report.errors, ())
        self.assertTrue(report.can_optimize)
        self.assertFalse(report.final_comparison_eligible)
        self.assertGreaterEqual(
            report.table_counts["poi_entities"],
            bundle.manifest["quality"]["minimum_poi_entities"],
        )
        self.assertTrue(any("non-road-validated" in warning for warning in report.warnings))

    def test_snapshot_manifest_hashes_match_files(self):
        bundle = load_dataset_bundle(root=REPO_ROOT)

        for filename, expected_hash in bundle.manifest["files"].items():
            path = bundle.snapshot_dir / filename
            self.assertTrue(path.exists(), filename)
            self.assertEqual(sha256(path), expected_hash, filename)
            self.assertEqual(bundle.file_hashes[filename], expected_hash, filename)

    def test_default_config_and_metadata_capture_dataset_and_run_identity(self):
        config = load_trip_config(CONFIG_PATH)

        self.assertEqual(config.get("data", "catalog_snapshot_id"), "california_v1")
        self.assertEqual(config.get("data", "context_snapshot_id"), "context_static_demo_2026_06")
        self.assertEqual(config.get("data", "refresh_policy"), "never")
        self.assertEqual(config.get("run", "role"), "demonstration")
        self.assertEqual(config.get("routing", "road_route_cache_path"), ROAD_ROUTE_CACHE_FILENAME)

        with temporary_directory() as output_dir:
            metadata_path = write_artifact_metadata(output_dir, config, artifact_files=["route.csv"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertTrue(artifact_metadata_matches(output_dir, config))
            self.assertEqual(metadata["catalog_snapshot_id"], "california_v1")
            self.assertEqual(metadata["context_snapshot_id"], "context_static_demo_2026_06")
            self.assertEqual(metadata["refresh_policy"], "never")
            self.assertEqual(metadata["run_role"], "demonstration")
            self.assertTrue(metadata["run_id"].startswith("auto_"))

            changed_context = load_trip_config(
                CONFIG_PATH,
                overrides={"data": {"context_snapshot_id": "context_retrieved_2026_07_01"}},
            )
            self.assertFalse(artifact_metadata_matches(output_dir, changed_context))

    def test_route_contract_blocks_geodesic_fallback_from_evaluation(self):
        route = fallback_route()

        self.assertFalse(route.road_validated)
        self.assertTrue(route.fallback_used)
        self.assertFalse(route.evaluation_eligible)
        self.assertEqual(route.total_distance_m, 7600.0)
        audit = route.to_audit_rows()[0]
        self.assertEqual(audit["geometry_source"], "straight_waypoints")
        self.assertEqual(audit["distance_source"], "geodesic_proxy")
        self.assertFalse(audit["road_validated"])

    def test_route_contract_allows_validated_road_route(self):
        route = validated_route()

        self.assertTrue(route.road_validated)
        self.assertFalse(route.fallback_used)
        self.assertTrue(route.evaluation_eligible)
        self.assertEqual(route.total_duration_s, 1440.0)

    def test_phase0_evaluator_separates_artifacts_from_final_eligibility(self):
        plan = plan_artifact()
        run = planner_run()

        eligible = evaluate_phase0_plan(
            plan=plan,
            planner_run=run,
            route_result=validated_route(),
            dataset_snapshot_valid=True,
        )
        self.assertEqual(eligible.artifact_grounding_status, "PASSED")
        self.assertEqual(eligible.hard_feasibility_status, "PASSED")
        self.assertEqual(eligible.comparison_eligibility, "eligible")
        self.assertEqual(eligible.conditional_reward, 1.0)

        ineligible = evaluate_phase0_plan(
            plan=plan,
            planner_run=run,
            route_result=fallback_route(),
            dataset_snapshot_valid=True,
        )
        self.assertEqual(ineligible.artifact_grounding_status, "PASSED")
        self.assertEqual(ineligible.hard_feasibility_status, "FAILED")
        self.assertEqual(ineligible.comparison_eligibility, "ineligible")
        self.assertIsNone(ineligible.conditional_reward)

    def test_utility_signals_keep_coverage_and_uncertainty_separate(self):
        config = load_trip_config(CONFIG_PATH)
        enriched_df = pd.DataFrame(
            [
                {
                    "name": "Sparse but Important Stop",
                    "city": "San Francisco",
                    "category": "viewpoint",
                    "source_list": "curated_seed",
                    "source_score": 8.0,
                    "source_coverage_score": 0.25,
                    "model_uncertainty": 0.70,
                    "weather_risk": 0.10,
                }
            ]
        )

        signal_df = build_signal_matrix(enriched_df, config)

        self.assertAlmostEqual(float(signal_df["source_coverage_score"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(signal_df["data_confidence"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(signal_df["model_uncertainty"].iloc[0]), 0.70)
        self.assertAlmostEqual(float(signal_df["data_uncertainty"].iloc[0]), 0.70)

    def test_phase0_exporter_writes_evidence_tables_and_blocks_unvalidated_routes(self):
        config = load_trip_config(CONFIG_PATH)
        method_df = pd.DataFrame(
            [
                {
                    "method": "hierarchical_bandit_gurobi_repair",
                    "method_display_name": "Hierarchical + Bandit + Small Gurobi Repair",
                    "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                    "trip_days": 7,
                    "status": "FEASIBLE",
                    "local_route_solver": "small_gurobi",
                    "objective": 12.5,
                    "solve_seconds": 0.2,
                    "mip_gap": 0.0,
                    "total_utility": 2.1,
                    "total_travel_time": 45.0,
                    "total_travel_distance_km": 14.0,
                    "selected_attractions": 2,
                }
            ]
        )
        route_stops_df = pd.DataFrame(
            [
                {
                    "comparison_type": "method",
                    "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                    "method": "hierarchical_bandit_gurobi_repair",
                    "trip_days": 7,
                    "day": 1,
                    "stop_order": 1,
                    "attraction_name": "Golden Gate Bridge",
                    "city": "San Francisco",
                    "latitude": 37.8199,
                    "longitude": -122.4783,
                    "route_start_name": "SFO",
                    "route_start_latitude": 37.6213,
                    "route_start_longitude": -122.3790,
                    "route_end_name": "Optimizer Hotel",
                    "route_end_latitude": 37.78,
                    "route_end_longitude": -122.42,
                    "source_list": "curated_seed",
                    "source_confidence": 0.7,
                },
                {
                    "comparison_type": "method",
                    "comparison_label": "Method · Hierarchical + Bandit + Small Gurobi Repair",
                    "method": "hierarchical_bandit_gurobi_repair",
                    "trip_days": 7,
                    "day": 1,
                    "stop_order": 2,
                    "attraction_name": "Ferry Building",
                    "city": "San Francisco",
                    "latitude": 37.7955,
                    "longitude": -122.3937,
                    "route_start_name": "SFO",
                    "route_start_latitude": 37.6213,
                    "route_start_longitude": -122.3790,
                    "route_end_name": "Optimizer Hotel",
                    "route_end_latitude": 37.78,
                    "route_end_longitude": -122.42,
                    "source_list": "curated_seed",
                    "source_confidence": 0.7,
                },
            ]
        )

        with temporary_directory() as output_dir:
            outputs = write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=method_df,
                route_stops_df=route_stops_df,
            )

            for filename in PHASE0_ARTIFACT_FILES:
                self.assertTrue((output_dir / filename).exists(), filename)

            summary = pd.read_csv(output_dir / "production_phase0_evidence_summary.csv")
            evaluation = pd.read_csv(output_dir / "production_phase0_evaluation_reports.csv")
            route_audit = pd.read_csv(output_dir / "production_phase0_route_audit.csv")
            planner_runs = pd.read_csv(output_dir / "production_phase0_planner_runs.csv")
            dataset_report = json.loads((output_dir / "production_phase0_dataset_validation.json").read_text())
            plan_lines = (output_dir / "production_phase0_plan_artifacts.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(outputs["plan_artifact_count"], 1)
        self.assertEqual(dataset_report["catalog_snapshot_id"], "california_v1")
        self.assertFalse(bool(dataset_report["final_comparison_eligible"]))
        self.assertEqual(dataset_report["route_cache_coverage"]["road_route_requested_leg_count"], 0)
        self.assertEqual(dataset_report["route_cache_coverage"]["road_route_validated_leg_count"], 0)
        self.assertEqual(len(plan_lines), 1)
        self.assertEqual(summary["comparison_eligibility"].iloc[0], "ineligible")
        self.assertFalse(bool(summary["route_road_validated"].iloc[0]))
        self.assertTrue(bool(summary["route_fallback_used"].iloc[0]))
        self.assertEqual(int(summary["route_validated_leg_count"].iloc[0]), 0)
        self.assertEqual(int(summary["route_fallback_leg_count"].iloc[0]), 3)
        self.assertAlmostEqual(float(summary["route_road_validation_coverage"].iloc[0]), 0.0)
        self.assertEqual(evaluation["artifact_grounding_status"].iloc[0], "PASSED")
        self.assertEqual(evaluation["hard_feasibility_status"].iloc[0], "FAILED")
        self.assertEqual(planner_runs["catalog_snapshot_id"].iloc[0], "california_v1")
        self.assertTrue(route_audit["geometry_source"].eq("straight_waypoints").all())
        self.assertTrue(route_audit["fallback_used"].astype(bool).all())

    def test_production_metadata_tracks_phase0_artifacts(self):
        self.assertTrue(set(PHASE0_ARTIFACT_FILES).issubset(set(PRODUCTION_ARTIFACT_FILES)))

    def test_phase0_validator_accepts_consistent_artifacts_and_rejects_false_eligible_claims(self):
        config = load_trip_config(CONFIG_PATH)

        with temporary_directory() as output_dir:
            write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_phase0_artifacts.py"),
                "--output-dir",
                str(output_dir),
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("PASSED", passed.stdout)

            evaluations = pd.read_csv(output_dir / "production_phase0_evaluation_reports.csv")
            evaluations["comparison_eligibility"] = "eligible"
            evaluations["hard_feasibility_status"] = "PASSED"
            evaluations.to_csv(output_dir / "production_phase0_evaluation_reports.csv", index=False)

            failed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("unvalidated or fallback route", failed.stdout)

    def test_phase0_exporter_uses_validated_road_route_cache_for_strict_eligibility(self):
        config = load_trip_config(CONFIG_PATH)

        with temporary_directory() as output_dir:
            phase0_road_route_cache_frame().to_csv(output_dir / ROAD_ROUTE_CACHE_FILENAME, index=False)
            write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_phase0_artifacts.py"),
                "--output-dir",
                str(output_dir),
                "--require-final-eligible",
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

            summary = pd.read_csv(output_dir / "production_phase0_evidence_summary.csv")
            evaluation = pd.read_csv(output_dir / "production_phase0_evaluation_reports.csv")
            route_audit = pd.read_csv(output_dir / "production_phase0_route_audit.csv")

        self.assertEqual(summary["comparison_eligibility"].iloc[0], "eligible")
        self.assertTrue(bool(summary["route_road_validated"].iloc[0]))
        self.assertFalse(bool(summary["route_fallback_used"].iloc[0]))
        self.assertEqual(int(summary["route_validated_leg_count"].iloc[0]), 3)
        self.assertEqual(int(summary["route_fallback_leg_count"].iloc[0]), 0)
        self.assertAlmostEqual(float(summary["route_road_validation_coverage"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(summary["road_route_validation_coverage"].iloc[0]), 1.0)
        self.assertTrue(str(summary["road_route_cache_hash"].iloc[0]))
        self.assertEqual(evaluation["hard_feasibility_status"].iloc[0], "PASSED")
        self.assertTrue(route_audit["road_validated"].astype(bool).all())
        self.assertFalse(route_audit["fallback_used"].astype(bool).any())
        self.assertTrue(route_audit["geometry_source"].eq("validated_osrm_cache").all())

    def test_road_route_cache_builder_audits_missing_cache_without_validation(self):
        with temporary_directory() as output_dir:
            result = build_road_route_cache_from_artifacts(
                output_dir=output_dir,
                route_stops_df=phase0_route_stops_frame(),
            )
            audit = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_AUDIT_FILENAME)
            cache = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_FILENAME)
            requests = pd.read_csv(output_dir / ROAD_ROUTE_REQUESTS_FILENAME)

        self.assertFalse(result.complete)
        self.assertTrue(cache.empty)
        self.assertEqual(len(audit), 3)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(result.request_df), 3)
        self.assertTrue(audit["status"].eq("missing_osrm_cache").all())
        self.assertFalse(audit["road_validated"].astype(bool).any())
        self.assertTrue(requests["osrm_route_url"].str.contains("/route/v1/driving/").all())
        self.assertTrue(requests["osrm_route_url"].str.startswith(OSRM_LOCAL_BASE_URL).all())
        self.assertTrue(requests["cache_path"].str.contains("open_osrm_route_").all())

    def test_route_fetch_policy_blocks_public_osrm_without_explicit_approval(self):
        calls = []

        def fake_fetcher(points, base_url, timeout_seconds):
            calls.append((points, base_url, timeout_seconds))
            return {}

        with temporary_directory() as output_dir:
            with self.assertRaisesRegex(ValueError, "public OSRM fetch requires"):
                build_road_route_cache_from_artifacts(
                    output_dir=output_dir,
                    route_stops_df=phase0_route_stops_frame(),
                    fetch_missing=True,
                    osrm_base_url=OSRM_PUBLIC_BASE_URL,
                    osrm_fetcher=fake_fetcher,
                )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_road_route_cache.py"),
                "--output-dir",
                str(output_dir),
                "--route-stops",
                str(output_dir / "production_method_route_stops.csv"),
                "--fetch-missing",
                "--osrm-base-url",
                OSRM_PUBLIC_BASE_URL,
            ]
            phase0_route_stops_frame().to_csv(output_dir / "production_method_route_stops.csv", index=False)
            failed = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertTrue(is_public_osrm_base_url(OSRM_PUBLIC_BASE_URL))
        self.assertEqual(calls, [])
        self.assertEqual(failed.returncode, 2)
        self.assertIn("--allow-public-osrm", failed.stdout)

    def test_route_source_readiness_check_is_manifest_and_policy_first(self):
        with temporary_directory() as output_dir:
            empty_status, empty_messages = check_route_source(output_dir=output_dir)

            build_road_route_cache_from_artifacts(
                output_dir=output_dir,
                route_stops_df=phase0_route_stops_frame(),
            )
            status, messages = check_route_source(output_dir=output_dir)
            public_status, public_messages = check_route_source(
                output_dir=output_dir,
                osrm_base_url=OSRM_PUBLIC_BASE_URL,
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_route_source.py"),
                "--output-dir",
                str(output_dir),
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertEqual(empty_status, 1)
        self.assertIn("missing or empty", "\n".join(empty_messages))
        self.assertEqual(status, 0)
        self.assertIn("Probe: skipped", "\n".join(messages))
        self.assertEqual(public_status, 2)
        self.assertIn("--allow-public-osrm", "\n".join(public_messages))
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertIn("PASSED route-source readiness precheck", passed.stdout)

    def test_road_route_cache_builder_converts_cached_osrm_for_strict_phase0(self):
        config = load_trip_config(CONFIG_PATH)

        with temporary_directory() as output_dir:
            write_phase0_osrm_cache_files(output_dir)
            result = build_road_route_cache_from_artifacts(
                output_dir=output_dir,
                route_stops_df=phase0_route_stops_frame(),
            )
            self.assertTrue(result.complete)
            self.assertEqual(len(result.cache_df), 3)
            self.assertTrue(result.audit_df["road_validated"].astype(bool).all())

            write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_phase0_artifacts.py"),
                "--output-dir",
                str(output_dir),
                "--require-final-eligible",
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

            route_audit = pd.read_csv(output_dir / "production_phase0_route_audit.csv")

        self.assertTrue(route_audit["road_validated"].astype(bool).all())
        self.assertTrue(route_audit["geometry_source"].eq("cached_osrm_route_geometry").all())

    def test_road_route_cache_builder_fetches_missing_osrm_when_enabled(self):
        calls = []

        def fake_fetcher(points, base_url, timeout_seconds):
            calls.append((points, base_url, timeout_seconds))
            left, right = points[0], points[-1]
            return {
                "routes": [
                    {
                        "distance": 1234.0 + len(calls),
                        "duration": 321.0 + len(calls),
                        "geometry": {
                            "coordinates": [
                                [left[1], left[0]],
                                [(left[1] + right[1]) / 2, (left[0] + right[0]) / 2],
                                [right[1], right[0]],
                            ]
                        },
                    }
                ]
            }

        with temporary_directory() as output_dir:
            result = build_road_route_cache_from_artifacts(
                output_dir=output_dir,
                route_stops_df=phase0_route_stops_frame(),
                fetch_missing=True,
                osrm_base_url="https://osrm.test",
                request_timeout_seconds=7,
                osrm_fetcher=fake_fetcher,
            )
            cache_files = sorted((output_dir / "cache").glob("open_osrm_route_*.json"))
            audit = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_AUDIT_FILENAME)
            cache = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_FILENAME)
            requests = pd.read_csv(output_dir / ROAD_ROUTE_REQUESTS_FILENAME)

        self.assertTrue(result.complete)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(cache_files), 3)
        self.assertEqual(len(cache), 3)
        self.assertEqual(len(requests), 3)
        self.assertTrue(audit["fetch_attempted"].astype(bool).all())
        self.assertTrue(audit["fetch_status"].eq("fetched_osrm_response").all())
        self.assertTrue(audit["status"].eq("fetched_osrm_validated").all())
        self.assertTrue(cache["road_validated"].astype(bool).all())

    def test_build_road_route_cache_script_writes_validated_cache(self):
        with temporary_directory() as output_dir:
            write_phase0_osrm_cache_files(output_dir)
            phase0_route_stops_frame().to_csv(output_dir / "production_method_route_stops.csv", index=False)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_road_route_cache.py"),
                "--output-dir",
                str(output_dir),
                "--require-complete",
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("PASSED", passed.stdout)
            self.assertIn(str(output_dir / ROAD_ROUTE_REQUESTS_FILENAME), passed.stdout)
            cache = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_FILENAME)
            audit = pd.read_csv(output_dir / ROAD_ROUTE_CACHE_AUDIT_FILENAME)
            requests = pd.read_csv(output_dir / ROAD_ROUTE_REQUESTS_FILENAME)

        self.assertEqual(len(cache), 3)
        self.assertEqual(len(requests), 3)
        self.assertTrue(cache["road_validated"].astype(bool).all())
        self.assertTrue(audit["road_validated"].astype(bool).all())

    def test_legacy_route_cache_audit_does_not_claim_incomplete_geometry_as_valid(self):
        with temporary_directory() as output_dir:
            build_road_route_cache_from_artifacts(
                output_dir=output_dir,
                route_stops_df=phase0_route_stops_frame(),
            )
            request = pd.read_csv(output_dir / ROAD_ROUTE_REQUESTS_FILENAME).iloc[0]
            key = f"{request['origin_latitude']:.5f},{request['origin_longitude']:.5f}|{request['destination_latitude']:.5f},{request['destination_longitude']:.5f}"
            legacy_cache_path = output_dir / "cache" / "production_road_route_cache.json"
            legacy_cache_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_cache_path.write_text(
                json.dumps(
                    {
                        key: {
                            "path": [
                                [request["origin_latitude"], request["origin_longitude"]],
                                [
                                    (request["origin_latitude"] + request["destination_latitude"]) / 2,
                                    (request["origin_longitude"] + request["destination_longitude"]) / 2,
                                ],
                                [request["destination_latitude"], request["destination_longitude"]],
                            ],
                            "mode": "legacy_osrm_path",
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = audit_legacy_route_cache(output_dir=output_dir, legacy_cache_path=legacy_cache_path)
            audit = pd.read_csv(output_dir / LEGACY_ROUTE_CACHE_AUDIT_FILENAME)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "audit_legacy_route_cache.py"),
                "--output-dir",
                str(output_dir),
                "--legacy-cache",
                str(legacy_cache_path),
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertIn("Legacy geometry matches: 1/3", passed.stdout)
        self.assertEqual(result["geometry_available_count"], 1)
        self.assertEqual(result["conversion_eligible_count"], 0)
        self.assertEqual(len(audit), 3)
        self.assertEqual(audit["legacy_match_type"].iloc[0], "exact")
        self.assertTrue(bool(audit["legacy_geometry_available"].iloc[0]))
        self.assertFalse(bool(audit["conversion_eligible"].iloc[0]))
        self.assertIn("legacy_duration_missing", audit["blocking_reason"].iloc[0])

    def test_phase0_readiness_summary_reports_blocked_and_ready_states(self):
        config = load_trip_config(CONFIG_PATH)
        with temporary_directory() as output_dir:
            write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            )
            blocked = load_phase0_readiness(output_dir)
            blocked_markdown = readiness_markdown(blocked)

            phase0_road_route_cache_frame().to_csv(output_dir / ROAD_ROUTE_CACHE_FILENAME, index=False)
            write_phase0_research_artifacts(
                output_dir=output_dir,
                config=config,
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            )
            ready = load_phase0_readiness(output_dir)
            ready_markdown = readiness_markdown(ready)

        self.assertFalse(blocked["strict_comparison_ready"])
        self.assertEqual(blocked["eligible_evaluation_count"], 0)
        self.assertIn("route_not_road_validated", blocked_markdown)
        self.assertTrue(ready["strict_comparison_ready"])
        self.assertEqual(ready["eligible_evaluation_count"], 1)
        self.assertIn("Strict comparison ready: `True`", ready_markdown)

    def test_phase0_evidence_pipeline_reports_current_blocker_without_strict_failure(self):
        with temporary_directory() as output_dir:
            quality_dir = output_dir / "quality"
            phase0_method_frame().to_csv(output_dir / "production_method_comparison.csv", index=False)
            phase0_route_stops_frame().to_csv(output_dir / "production_method_route_stops.csv", index=False)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_phase0_evidence_pipeline.py"),
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(output_dir),
                "--quality-dir",
                str(quality_dir),
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

            readiness = load_phase0_readiness(output_dir)

        self.assertIn("PASSED Phase 0 evidence pipeline", passed.stdout)
        self.assertIn("Road-route cache coverage: `0/3`", passed.stdout)
        self.assertIn("Strict comparison ready: `False`", passed.stdout)
        self.assertFalse(readiness["strict_comparison_ready"])
        self.assertEqual(readiness["road_route_missing_leg_count"], 3)
        self.assertTrue((output_dir / ROAD_ROUTE_REQUESTS_FILENAME).exists())
        self.assertTrue((quality_dir / "phase0_readiness_summary.md").exists())
        self.assertTrue((quality_dir / "phase0_method_readiness.csv").exists())

    def test_phase0_evidence_pipeline_passes_strict_mode_with_osrm_cache(self):
        with temporary_directory() as output_dir:
            quality_dir = output_dir / "quality"
            phase0_method_frame().to_csv(output_dir / "production_method_comparison.csv", index=False)
            phase0_route_stops_frame().to_csv(output_dir / "production_method_route_stops.csv", index=False)
            write_phase0_osrm_cache_files(output_dir)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_phase0_evidence_pipeline.py"),
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(output_dir),
                "--quality-dir",
                str(quality_dir),
                "--require-final-eligible",
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

            readiness = load_phase0_readiness(output_dir)

        self.assertIn("PASSED Phase 0 evidence pipeline", passed.stdout)
        self.assertIn("Road-route cache coverage: `3/3`", passed.stdout)
        self.assertIn("Strict comparison ready: `True`", passed.stdout)
        self.assertTrue(readiness["strict_comparison_ready"])
        self.assertEqual(readiness["eligible_evaluation_count"], 1)


if __name__ == "__main__":
    unittest.main()
