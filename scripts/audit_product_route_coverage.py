"""Audit every declared product-demo route leg and itinerary stop in order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.route_coverage import audit_route_coverage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = ROOT / "runs" / "california-coast-product-demo-v2"


def build_report(repository_root: Path, demo_directory: Path) -> dict[str, Any]:
    package = load_product_demo_package(repository_root, demo_directory)
    bundle = package.primary_bundle
    plans = (
        bundle.parent_plan,
        bundle.child_plan,
        *(plan for plan, _label in package.additional_plans),
    )
    cell_index = {
        (str(cell["origin_id"]), str(cell["destination_id"])): cell
        for cell in bundle.route_matrix["cells"]
    }
    reports = [
        audit_route_coverage(plan, package.route_legs_by_plan[plan["plan_id"]], cell_index)
        for plan in plans
    ]
    return {
        "schema_version": "product-route-coverage-audit-v1",
        "status": "complete" if all(report.complete for report in reports) else "incomplete",
        "plans": [report.as_dict() for report in reports],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-directory", type=Path, default=DEFAULT_DEMO)
    args = parser.parse_args()
    report = build_report(ROOT, args.demo_directory.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
