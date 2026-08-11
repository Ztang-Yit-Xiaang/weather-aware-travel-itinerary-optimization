from __future__ import annotations

import pandas as pd

from itinerary_system.dashboard_evaluation import EVALUATION_METHODS, build_evaluation_metrics
from itinerary_system.map_exporter import (
    _evaluation_metrics,
    _write_evaluation_page,
)

CANONICAL_METHOD_IDS = (
    "context_blind_solver",
    "deterministic_context_aware_heuristic",
    "progressive_sequential_lexicographic_repair",
    "full_reoptimization",
)


def test_evaluation_metrics_do_not_invent_rows_when_evidence_is_missing(tmp_path):
    payload = _evaluation_metrics(tmp_path)

    assert tuple(method_id for method_id, _ in EVALUATION_METHODS) == CANONICAL_METHOD_IDS
    assert payload["available"] is False
    assert payload["data_status"] == "not_available"
    assert payload["methods"] == []
    assert payload["empty_message"]


def test_evaluation_metrics_expose_only_canonical_evaluator_owned_fields(tmp_path):
    pd.DataFrame(
        [
            {
                "method": "full_reoptimization",
                "status": "complete_candidate_limit_exceeded:50000",
                "benchmark_ranking_eligible": False,
                "preservation_weighted_edit_cost": 4.0,
                "quality_utility_retained": 0.8,
                "quality_weather_risk_delta": 0.2,
                "computation_runtime_seconds": 1.5,
            }
        ]
    ).to_csv(tmp_path / "production_method_comparison.csv", index=False)

    payload = _evaluation_metrics(tmp_path)

    assert payload["available"] is True
    assert [method["method"] for method in payload["methods"]] == ["full_reoptimization"]
    method = payload["methods"][0]
    assert method["ranking_eligible"] is False
    assert method["weighted_edit_cost"] == 4.0
    assert method["utility_retained"] == 0.8
    assert "comparison_score" not in method


def test_map_exporter_wrapper_matches_extracted_evaluation_component(tmp_path):
    comparison = pd.DataFrame(
        [
            {
                "method": "progressive_sequential_lexicographic_repair",
                "method_display_name": "Progressive repair",
                "status": "PASSED",
                "benchmark_ranking_eligible": True,
                "preservation_weighted_edit_cost": 2.5,
                "quality_utility_retained": 0.9,
                "quality_weather_risk_delta": 0.3,
                "computation_runtime_seconds": 0.75,
                "notes": "characterization fixture",
            }
        ]
    )
    route_stops = pd.DataFrame(
        [
            {"method": "progressive_sequential_lexicographic_repair", "stop_id": "stop-1"},
            {"method": "progressive_sequential_lexicographic_repair", "stop_id": "stop-2"},
        ]
    )
    comparison.to_csv(tmp_path / "production_method_comparison.csv", index=False)
    route_stops.to_csv(tmp_path / "production_method_route_stops.csv", index=False)

    assert _evaluation_metrics(tmp_path) == build_evaluation_metrics(comparison, route_stops)

def test_evaluation_page_inverts_lower_is_better_bars_and_escapes_data(tmp_path):
    root = tmp_path / "dashboard"
    assets = root / "assets"
    assets.mkdir(parents=True)
    written = []
    _write_evaluation_page(root, assets, _evaluation_metrics(tmp_path), written)

    html = (root / "evaluation.html").read_text(encoding="utf-8")
    assert "field.higher_is_better ? (value - min) / span : (max - value) / span" in html
    assert "Lower is better" in html
    assert "escapeHtml(method.label)" in html
    assert "payload.empty_message" in html