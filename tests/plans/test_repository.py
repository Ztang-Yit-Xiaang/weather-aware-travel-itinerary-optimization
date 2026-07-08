import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from itinerary_system.plans import (
    PlanNotFound,
    PlanRepository,
    PlanRepositoryConflict,
    load_plan,
    save_plan_append_only,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def demo_plan(plan_id: str = "plan_demo") -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id=plan_id,
        source_run_id="run_demo",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"stop_id": "poi_a", "day": 1, "stop_order": 1},
            {"stop_id": "poi_b", "day": 1, "stop_order": 2},
        ),
        day_assignments={"poi_a": 1, "poi_b": 1},
        sequence=("poi_a", "poi_b"),
        ordered_days=({"day": 1, "stop_ids": ("poi_a", "poi_b")},),
        route_ids_by_day={1: "route_day_1"},
        created_at="2026-07-07T00:00:00+00:00",
    )


class PlanRepositoryTests(unittest.TestCase):
    def test_save_load_and_verify_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = demo_plan()

            path = save_plan_append_only(plan, root)
            loaded = load_plan(path)
            repository = PlanRepository(root)

            self.assertEqual(path, root / "plan_demo.json")
            self.assertEqual(loaded.to_record(include_content_hash=False), plan.to_record(include_content_hash=False))
            self.assertTrue(repository.verify_hash(plan.plan_id))

    def test_append_only_save_is_idempotent_for_identical_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = PlanRepository(temp_dir)
            plan = demo_plan()

            first = repository.save(plan)
            second = repository.save(plan)

            self.assertEqual(first, second)

    def test_append_only_save_rejects_same_id_different_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = PlanRepository(temp_dir)
            plan = demo_plan()
            repository.save(plan)
            changed = replace(plan, selected_stops=(*plan.selected_stops, {"stop_id": "poi_c", "day": 2}))

            with self.assertRaises(PlanRepositoryConflict):
                repository.save(changed)

    def test_missing_plan_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = PlanRepository(temp_dir)

            with self.assertRaises(PlanNotFound):
                repository.load("missing")


if __name__ == "__main__":
    unittest.main()
