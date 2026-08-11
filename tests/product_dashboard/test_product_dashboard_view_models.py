from __future__ import annotations

from itinerary_system.product_dashboard_adapter import load_product_dashboard_source
from itinerary_system.product_dashboard_view_models import (
    build_product_dashboard_view_model,
)


def test_customer_and_research_views_share_one_artifact_model(product_run_factory) -> None:
    model = build_product_dashboard_view_model(
        load_product_dashboard_source(product_run_factory())
    )

    assert model["repair"]["status"] == "Eligible repair"
    assert model["timeline"][0]["states"] == ["unchanged"]
    assert {"affected", "changed"} <= set(model["timeline"][1]["states"])
    assert model["research"]["lineage"]["parent_plan_id"] == "parent"
    assert model["research"]["lineage"]["child_plan_id"] == "child"
    assert model["interaction"]["enabled"] is False


def test_exact_failures_have_reasons_and_never_receive_ranks(product_run_factory) -> None:
    model = build_product_dashboard_view_model(
        load_product_dashboard_source(product_run_factory())
    )
    exact = next(
        row for row in model["alternatives"] if row["method_id"] == "context_blind_solver"
    )

    assert exact["display_status"] == "Exact search incomplete"
    assert "candidate_limit_exceeded" in exact["failure_reason"]
    assert exact["ranking_eligible"] is False
    assert "rank" not in exact


def test_comparison_metrics_keep_owner_and_directionality(product_run_factory) -> None:
    model = build_product_dashboard_view_model(
        load_product_dashboard_source(product_run_factory())
    )
    metrics = {metric["id"]: metric for metric in model["comparison"]}

    assert metrics["weighted_edit_cost"]["owner"] == "plan_diff"
    assert metrics["weighted_edit_cost"]["direction"] == "lower"
    assert metrics["utility_retained"]["owner"] == "independent_evaluator"
    assert metrics["route_validity"]["child"]["value"] == "1/1"


def test_map_has_original_repaired_and_text_alternative(product_run_factory) -> None:
    model = build_product_dashboard_view_model(
        load_product_dashboard_source(product_run_factory())
    )

    assert model["map"]["parent"]["segments"]
    assert model["map"]["child"]["segments"]
    assert model["map"]["evidence"]["label"] == "Weather Deterioration"
    assert model["map"]["child"]["stops"][1]["ownership_strength"] == "preferred"
    assert "Day 2 (affected)" in model["map_alternative"]


def test_missing_evaluator_metric_remains_unavailable_not_zero(
    product_run_factory,
) -> None:
    model = build_product_dashboard_view_model(
        load_product_dashboard_source(
            product_run_factory(
                certificate_updates={
                    "metrics": {
                        "preservation_rate": 0.5,
                        "utility_retained": None,
                        "weather_risk_delta": None,
                    }
                }
            )
        )
    )
    metrics = {metric["id"]: metric for metric in model["comparison"]}

    assert metrics["utility_retained"]["child"]["value"] is None
    assert metrics["utility_retained"]["child"]["state"] == "unavailable"
    assert {"null_metric", "unavailable_metric"} <= {
        state["id"] for state in model["truth_states"]
    }
