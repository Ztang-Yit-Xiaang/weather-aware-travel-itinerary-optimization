from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from itinerary_system.research_artifacts import stable_content_hash


def _hashed_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = dict(payload)
    plan["content_hash"] = stable_content_hash(plan)
    return plan


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def product_run_factory(tmp_path: Path):
    def build(
        *,
        certificate_updates: dict[str, Any] | None = None,
        child_updates: dict[str, Any] | None = None,
        diff_updates: dict[str, Any] | None = None,
        benchmark_rows: list[dict[str, Any]] | None = None,
        benchmark_planner_runs: list[dict[str, Any]] | None = None,
    ) -> Path:
        run_dir = tmp_path / "run"
        parent = _hashed_plan(
            {
                "schema_version": "plan-v1",
                "plan_id": "parent",
                "parent_plan_id": None,
                "ordered_days": [{"day": 1}, {"day": 2}],
                "selected_stops": [
                    {
                        "stop_id": "a",
                        "name": "Alpha",
                        "city": "San Francisco",
                        "day": 1,
                        "stop_order": 1,
                        "latitude": 37.77,
                        "longitude": -122.42,
                    },
                    {
                        "stop_id": "b",
                        "name": "Beta",
                        "city": "Monterey",
                        "day": 2,
                        "stop_order": 1,
                        "latitude": 36.60,
                        "longitude": -121.89,
                    },
                ],
                "sequence": ["a", "b"],
                "modeled_metrics": {"total_travel_time": 120.0},
                "owned_constraints": [],
            }
        )
        child_payload = {
            "schema_version": "plan-v1",
            "plan_id": "child",
            "parent_plan_id": "parent",
            "ordered_days": [{"day": 1}, {"day": 2}],
            "selected_stops": [
                {
                    "stop_id": "a",
                    "name": "Alpha",
                    "city": "San Francisco",
                    "day": 1,
                    "stop_order": 1,
                    "latitude": 37.77,
                    "longitude": -122.42,
                },
                {
                    "stop_id": "c",
                    "name": "Gamma",
                    "city": "Carmel",
                    "day": 2,
                    "stop_order": 1,
                    "latitude": 36.55,
                    "longitude": -121.92,
                },
            ],
            "sequence": ["a", "c"],
            "modeled_metrics": {"total_travel_time": 105.0},
            "owned_constraints": [
                {
                    "target_id": "c",
                    "strength": "preferred",
                    "confirmed": True,
                }
            ],
        }
        child_payload.update(child_updates or {})
        child = _hashed_plan(child_payload)
        diff = {
            "diff_id": "diff",
            "parent_plan_id": "parent",
            "child_plan_id": "child",
            "added_stops": [{"stop_id": "c", "day": 2, "owner_strength": "preferred"}],
            "deleted_stops": [{"stop_id": "b", "day": 2, "owner_strength": "flexible"}],
            "day_moves": [],
            "time_shifts": [],
            "reorder_changes": [],
            "lodging_changes": [],
            "road_changes": [{"day": 2}],
            "unchanged_days": [1],
            "weighted_edit_cost": 3.0,
        }
        diff.update(diff_updates or {})
        certificate = {
            "certificate_id": "cert",
            "plan_id": "child",
            "plan_content_hash": child["content_hash"],
            "diff_id": "diff",
            "eligible": True,
            "comparison_eligibility": "eligible",
            "evaluation_status": "PASSED",
            "failure_count": 0,
            "failures": [],
            "metrics": {
                "preservation_rate": 0.5,
                "utility_retained": 0.9,
                "weather_risk_delta": 0.7,
            },
            "route_validation": {
                "required_leg_count": 1,
                "road_validated_leg_count": 1,
                "missing_leg_count": 0,
                "fallback_leg_count": 0,
                "source_bundle_id": "bundle",
                "source_content_sha256": "abc123",
            },
        }
        certificate.update(certificate_updates or {})
        request = {
            "request_id": "scenario",
            "kind": "weather_deterioration",
            "user_intent": "Avoid the weather-affected stop.",
            "confirmed_constraints": {
                "affected_days": [2],
                "target_stop_ids": ["b"],
                "disruption_family": "weather_deterioration",
                "synthetic": True,
                "observed": False,
            },
            "evidence_records": [{"evidence_id": "weather-1"}],
        }
        explanation = {
            "evidence_id": "explain",
            "parent_plan_id": "parent",
            "child_plan_id": "child",
            "diff_id": "diff",
            "claims": [
                {
                    "claim_id": "claim",
                    "rendered_text": "Day 2 changed because weather risk increased.",
                    "claim_type": "repair_reason",
                    "confidence": "artifact-backed",
                    "evidence_refs": ["weather-1"],
                    "unsupported": False,
                }
            ],
            "evidence_records": [
                {
                    "ref_id": "weather-1",
                    "source_type": "weather",
                    "source_id": "snapshot",
                    "field_path": "day.2",
                    "record_hash": "weatherhash",
                }
            ],
        }
        route_matrix = {
            "matrix_id": "matrix",
            "cells": [
                {
                    "origin_id": "a",
                    "destination_id": "b",
                    "geometry": [[-122.42, 37.77], [-121.89, 36.60]],
                    "road_validated": True,
                    "fallback_used": False,
                    "distance_m": 180000,
                    "duration_s": 7200,
                },
                {
                    "origin_id": "a",
                    "destination_id": "c",
                    "geometry": [[-122.42, 37.77], [-121.92, 36.55]],
                    "road_validated": True,
                    "fallback_used": False,
                    "distance_m": 175000,
                    "duration_s": 6300,
                },
            ],
        }
        planner_rows = [
            {
                "run_id": "planner",
                "planning_request_id": "scenario",
                "method_requested": "lexicographic_repair",
                "method_executed": "progressive_sequential_lexicographic_repair",
                "execution_status": "COMPLETED",
                "solver_certification": "HEURISTIC",
            }
        ]
        benchmark_rows = benchmark_rows or [
            {
                "scenario_id": "scenario",
                "method_id": "context_blind_solver",
                "status": "failed",
                "benchmark_ranking_eligible": False,
            },
            {
                "scenario_id": "scenario",
                "method_id": "progressive_sequential_lexicographic_repair",
                "status": "completed",
                "benchmark_ranking_eligible": True,
                "benchmark_planner_method_executed_ids": [
                    "progressive_sequential_lexicographic_repair"
                ],
                "preservation_weighted_edit_cost": 3.0,
                "quality_utility_retained": 0.9,
                "quality_route_validated": True,
            },
        ]
        benchmark_planner_runs = benchmark_planner_runs or [
            {
                "run_id": "exact",
                "planning_request_id": "scenario",
                "method_requested": "context_blind_solver",
                "method_executed": "context_blind_solver",
                "execution_status": "FAILED",
                "solver_status_raw": "candidate_limit_exceeded",
                "error_summary": "complete_candidate_limit_exceeded:50000",
            }
        ]
        artifacts = {
            "requests": ["requests/request.json"],
            "plans": ["plans/parent.json", "plans/child.json"],
            "diffs": ["diffs/diff.json"],
            "evaluations": ["evaluations/cert.json"],
            "explanations": ["explanations/explain.json"],
            "routing": ["routing/matrix.json"],
            "planner_runs": ["planner_runs.jsonl"],
            "metrics": ["metrics/metrics.json"],
            "benchmark_metrics": ["benchmark/metrics.jsonl"],
            "benchmark_planner_runs": ["benchmark/planner_runs.jsonl"],
        }
        manifest = {
            "run_id": "product-test",
            "status": "completed",
            "mode": "strict",
            "parent_plan_id": "parent",
            "pipeline_run": {"output_plan_ids": ["child"]},
            "artifacts": artifacts,
            "artifact_counts": {key: len(value) for key, value in artifacts.items()},
        }
        _write_json(run_dir / "requests/request.json", request)
        _write_json(run_dir / "plans/parent.json", parent)
        _write_json(run_dir / "plans/child.json", child)
        _write_json(run_dir / "diffs/diff.json", diff)
        _write_json(run_dir / "evaluations/cert.json", certificate)
        _write_json(run_dir / "explanations/explain.json", explanation)
        _write_json(run_dir / "routing/matrix.json", route_matrix)
        _write_json(run_dir / "metrics/metrics.json", {"accepted_radius": 4})
        _write_jsonl(run_dir / "planner_runs.jsonl", planner_rows)
        _write_jsonl(run_dir / "benchmark/metrics.jsonl", benchmark_rows)
        _write_jsonl(
            run_dir / "benchmark/planner_runs.jsonl", benchmark_planner_runs
        )
        _write_json(run_dir / "manifest.json", manifest)
        return run_dir

    return build
