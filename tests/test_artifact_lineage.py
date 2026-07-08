import unittest
from dataclasses import replace

from itinerary_system.research_artifacts import (
    PlanArtifact,
    PlanArtifactV2,
    PlannerRun,
    create_child_plan_after_mutation,
    detect_post_solve_mutation,
    evaluate_phase0_plan,
    invalidate_certificate,
    mark_solver_certificate_invalidated,
    plan_artifact_from_v1,
)
from itinerary_system.routing import RouteLegResult, RouteResult


def planner_run() -> PlannerRun:
    return PlannerRun(
        run_id="run_child",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        planner_specification_id="phase0_test",
        method_requested="solver",
        method_executed="solver",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        result_plan_id="plan_child",
    )


def v1_plan() -> PlanArtifact:
    return PlanArtifact(
        plan_id="plan_parent",
        source_run_id="run_parent",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"poi_id": "poi_a", "day": 1},
            {"poi_id": "poi_b", "day": 1},
        ),
        sequence=("poi_a", "poi_b"),
        created_at="2026-07-06T00:00:00+00:00",
    )


def validated_route() -> RouteResult:
    leg = RouteLegResult(
        origin_id="poi_a",
        destination_id="poi_b",
        geometry=((1.0, 2.0), (3.0, 4.0)),
        distance_m=1000.0,
        duration_s=600.0,
        routing_status="ok",
        provider="unit",
        geometry_source="unit",
        distance_source="unit",
        duration_source="unit",
        road_validated=True,
    )
    return RouteResult(
        route_id="route_demo",
        legs=(leg,),
        solver_feasible=True,
        schedule_feasible=True,
        dataset_snapshot_valid=True,
    )


class ArtifactLineageTests(unittest.TestCase):
    def test_v1_plan_migrates_to_v2_with_ordered_days(self):
        migrated = plan_artifact_from_v1(v1_plan())

        self.assertIsInstance(migrated, PlanArtifactV2)
        self.assertEqual(migrated.schema_version, "plan-artifact-v2")
        self.assertEqual(migrated.ordered_days, ({"day": 1, "stop_ids": ("poi_a", "poi_b")},))
        self.assertTrue(migrated.content_hash)

    def test_v2_content_hash_changes_for_route_and_certificate_fields(self):
        base = plan_artifact_from_v1(v1_plan())
        with_route = replace(base, route_ids_by_day={1: "route_one"})
        with_certificate = replace(base, certificate_id="cert_one")

        self.assertNotEqual(base.content_hash, with_route.content_hash)
        self.assertNotEqual(base.content_hash, with_certificate.content_hash)

    def test_post_solve_mutation_detects_material_plan_content_change(self):
        parent = plan_artifact_from_v1(v1_plan())
        changed = replace(
            parent,
            selected_stops=(*parent.selected_stops, {"poi_id": "poi_c", "day": 1}),
        )

        report = detect_post_solve_mutation(parent, changed)

        self.assertTrue(report.material_change)
        self.assertTrue(report.requires_child_run)
        self.assertIn("selected_stops", report.changed_fields)
        self.assertIn("selected_stops_changed", report.reason_codes)

    def test_create_child_plan_after_mutation_rebinds_lineage(self):
        parent = plan_artifact_from_v1(v1_plan())
        changed = replace(parent, selected_stops=(*parent.selected_stops, {"poi_id": "poi_c", "day": 2}))
        run = planner_run()

        child = create_child_plan_after_mutation(parent, changed, run)

        self.assertNotEqual(child.plan_id, parent.plan_id)
        self.assertEqual(child.parent_plan_id, parent.plan_id)
        self.assertEqual(child.source_run_id, run.run_id)
        self.assertEqual(child.planning_request_id, run.planning_request_id)
        self.assertIsNone(child.certificate_id)

    def test_invalidated_solver_certificate_blocks_phase0_evaluation(self):
        plan = plan_artifact_from_v1(v1_plan())
        run = mark_solver_certificate_invalidated(planner_run(), reason="unit_mutation")

        evaluation = evaluate_phase0_plan(
            plan=plan,
            planner_run=run,
            route_result=validated_route(),
            dataset_snapshot_valid=True,
        )

        self.assertEqual(run.solver_certification, "INVALIDATED_AFTER_EDIT")
        self.assertEqual(evaluation.hard_feasibility_status, "FAILED")
        self.assertEqual(evaluation.comparison_eligibility, "ineligible")

    def test_invalidate_certificate_clears_certificate_and_marks_change(self):
        plan = replace(plan_artifact_from_v1(v1_plan()), certificate_id="cert_parent")

        invalidated = invalidate_certificate(plan)

        self.assertIsNone(invalidated.certificate_id)
        self.assertEqual(invalidated.change_components["certificate_invalidated"], 1.0)


if __name__ == "__main__":
    unittest.main()
