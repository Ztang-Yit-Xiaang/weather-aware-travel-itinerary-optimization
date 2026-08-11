from __future__ import annotations

import json

import pytest

from itinerary_system.product_dashboard_adapter import load_product_dashboard_source
from itinerary_system.product_dashboard_models import ProductDashboardValidationError


def test_loads_lineage_and_exact_incomplete_truth(product_run_factory) -> None:
    bundle = load_product_dashboard_source(product_run_factory())

    assert bundle.parent_plan["plan_id"] == "parent"
    assert bundle.child_plan["plan_id"] == "child"
    assert bundle.diff["child_plan_id"] == "child"
    assert "eligible_repair" in bundle.truth_states
    assert "exact_search_incomplete" in bundle.truth_states


def test_missing_declared_artifact_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory()
    (run_dir / "diffs/diff.json").unlink()

    with pytest.raises(ProductDashboardValidationError, match="missing_declared_artifact"):
        load_product_dashboard_source(run_dir)


def test_missing_parent_plan_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["plans"] = ["plans/child.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProductDashboardValidationError, match="missing_parent_plan"):
        load_product_dashboard_source(run_dir)


def test_missing_diff_for_child_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["diffs"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProductDashboardValidationError, match="missing_plan_diff"):
        load_product_dashboard_source(run_dir)


def test_manifest_path_escape_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"] = ["../outside.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProductDashboardValidationError, match="unsafe_artifact_path"):
        load_product_dashboard_source(run_dir)


def test_invalid_parent_content_hash_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory()
    path = run_dir / "plans/parent.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["content_hash"] = "wrong"
    path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ProductDashboardValidationError, match="parent_content_hash_mismatch"):
        load_product_dashboard_source(run_dir)


def test_child_parent_lineage_mismatch_is_rejected(product_run_factory) -> None:
    run_dir = product_run_factory(child_updates={"parent_plan_id": "other"})

    with pytest.raises(ProductDashboardValidationError, match="child_parent_lineage_mismatch"):
        load_product_dashboard_source(run_dir)


def test_certificate_lineage_mismatch_is_a_visible_truth_state(product_run_factory) -> None:
    bundle = load_product_dashboard_source(
        product_run_factory(certificate_updates={"plan_content_hash": "wrong"})
    )

    assert "certificate_mismatch" in bundle.truth_states
    assert "eligible_repair" not in bundle.truth_states


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_artifact_value_is_rejected(product_run_factory, value: float) -> None:
    run_dir = product_run_factory(diff_updates={"weighted_edit_cost": value})

    with pytest.raises(ProductDashboardValidationError, match="nonfinite_value"):
        load_product_dashboard_source(run_dir)


def test_infeasibility_is_not_mislabeled_as_exact_incomplete(product_run_factory) -> None:
    bundle = load_product_dashboard_source(
        product_run_factory(
            benchmark_planner_runs=[
                {
                    "planning_request_id": "scenario",
                    "method_requested": "context_blind_solver",
                    "method_executed": "context_blind_solver",
                    "execution_status": "FAILED",
                    "error_summary": "model infeasible after complete search",
                }
            ]
        )
    )

    assert "complete_infeasibility" in bundle.truth_states
    assert "exact_search_incomplete" not in bundle.truth_states


def test_booked_change_preserves_permission_provenance(product_run_factory) -> None:
    bundle = load_product_dashboard_source(
        product_run_factory(
            diff_updates={
                "added_stops": [
                    {"stop_id": "c", "day": 2, "owner_strength": "booked"}
                ]
            }
        )
    )

    assert "permission_required" in bundle.truth_states
