from dataclasses import replace
from pathlib import Path

import pytest

from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_dashboard_view_models import build_product_dashboard_view_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle():
    return load_product_demo_package(ROOT, ROOT / "runs/california-coast-product-demo-v2").primary_bundle


@pytest.mark.parametrize(("changes", "count"), [
    ({"duration_changes": [{"day": 7}]}, 1),
    ({"time_window_changes": [{"day": 6}]}, 1),
    ({"role_changes": [{"day": 6}]}, 1),
    ({"reorder_changes": [{"day": 4}, {"day": 4}], "road_changes": [{"day": 4}]}, 1),
    ({"added_stops": [{"day": 2}], "deleted_stops": [{"day": 3}]}, 2),
    ({"day_moves": [{"from_day": 2, "to_day": 5}]}, 2),
    ({"lodging_changes": [{"day": 3}], "time_shifts": [{"day": 3}]}, 1),
    ({"duration_changes": [{"day": 6}, {"day": 7}], "role_changes": [{"day": 6}]}, 2),
    ({}, 0),
])
def test_repair_changed_days_come_from_diff_not_disruption_request(bundle, changes, count):
    # The issue scope deliberately differs from the canonical edit scope.
    diff = {"added_stops": [], "deleted_stops": [], "unchanged_days": [], **changes}
    value = build_product_dashboard_view_model(replace(bundle, diff=diff,
        request={"confirmed_constraints": {"affected_days": [1, 2, 3]}}))
    assert value["repair"]["changed"]["affected_day_count"] == count
    assert value["issue"]["affected_days"] == [1, 2, 3]
