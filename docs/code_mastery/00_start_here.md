# Start Here

This guide teaches the current repository, not the July 6/8 snapshot and not a
generic Python architecture.

## Recommended Reading Order

1. [Current repository map](01_current_repository_map.md)
2. [Authoritative entry points](02_authoritative_entry_points.md)
3. [Pipeline and immutable artifacts](03_pipeline_and_immutable_artifacts.md)
4. [Parent, child, and diff](04_parent_child_plan_and_diff.md)
5. [Route matrix](05_route_matrix_and_road_validation.md)
6. [Repair and exact baselines](06_repair_methods_and_exact_baselines.md)
7. [Independent evaluator](07_independent_evaluator_and_metrics.md)
8. [Explanations](08_explanations_and_failure_evidence.md)
9. [Dashboards/renderers](09_dashboard_and_renderer_architecture.md)
10. [Interaction boundary](10_interaction_scaffold_and_permission_boundary.md)
11. [Testing](11_how_to_test_the_project.md)
12. [Debugging](12_debugging_playbook.md)
13. [Compatibility boundaries](13_legacy_compatibility_boundaries.md)
14. [Current gates](14_current_gate_map.md)
15. [Glossary](15_glossary.md)

## The One-Sentence Mental Model

The package turns a frozen parent itinerary plus a confirmed disruption into a
new immutable child, then independently evaluates the child and stores a typed
diff, certificate, route evidence, and explanation under a non-overwritable run
directory.

## Three Rules to Remember

1. The parent is never edited in place.
2. Solver success is not independent eligibility.
3. Evidence completeness is not the same as every benchmark method completing.

> **Beginner note / 初学者提示:** A “contract” here means a rule that code,
> tests, and artifacts agree to enforce. It is stronger than a comment or plan.

## Current Learning Check

- Which object records the accepted itinerary? `PlanArtifactV2`.
- Which object records change? `PlanDiff`.
- Which object decides display/ranking eligibility? The independent
  `PlanEvaluationCertificate`, not a planner score.
- Which component supplies road validity? `RouteMatrix` and its evidence bundle.
- Why is E3.1 blocked? Both exact methods refuse the real universe at the
  50,000-candidate safety cap.

