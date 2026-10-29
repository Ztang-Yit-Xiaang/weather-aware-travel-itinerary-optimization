"""Registered, independently evaluated owned-rule inputs for API/browser tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from itinerary_system.evaluation import PlanEvaluator, PlanEvaluatorConfig
from itinerary_system.plans import OwnedConstraint, compute_plan_diff
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler, _plan_artifact
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.research_artifacts import stable_content_hash
from scripts import build_coherent_product_demo as builder

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "california_coast_coherent_demo_v1"


def build_owned_product_registry(
    root: Path,
    *,
    protected_attribute: str | None = None,
    source_package: Path | None = None,
    existence_target: str | None = None,
    repository_root: Path = ROOT,
) -> Path:
    """Export actual fixture files; no API/compiler/evaluator boundary is mocked."""
    package = load_product_demo_package(
        repository_root, source_package or ROOT / "runs/california-coast-coherent-demo-v1"
    )
    base = package.primary_bundle
    compiler = FrozenDraftCompiler(base.parent_plan, package.evidence_bundles, repository_root=ROOT)
    matrix, day_config, _ = compiler._runtime_inputs()
    duration = {"mode": "exact", "preferred_minutes": 45, "minimum_minutes": 45, "maximum_minutes": 45}
    window = {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": "09:00",
        "latest_departure": "14:00",
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }
    rules = tuple(
        OwnedConstraint(
            constraint_id=f"owned_{attribute}",
            origin="user",
            scope="stop",
            target_id=target,
            relation=attribute,
            value=value,
            confirmed=True,
            strength="locked" if protected_attribute == attribute else "soft",
            relaxation_policy="never" if protected_attribute == attribute else "auto_with_penalty",
        ).to_record()
        for attribute, target, value in (
            ("duration", "griffith_observatory", duration),
            ("time_window", "golden_gate_bridge", window),
            ("role", "golden_gate_bridge", "activity"),
        )
    )
    if existence_target is not None:
        assert any(
            (stop.get("stop_id") or stop.get("poi_id")) == existence_target
            for stop in base.parent_plan["selected_stops"]
        )
        rules += (
            OwnedConstraint(
                constraint_id="owned_interest_existence",
                origin="user",
                scope="stop",
                target_id=existence_target,
                relation="must_keep",
                value=True,
                confirmed=True,
                strength="hard",
                relaxation_policy="explicit_only",
            ).to_record(),
        )

    def owned_plan(record, role, parent_id=None):
        plan = _plan_artifact(record)
        stops = deepcopy(list(plan.selected_stops))
        for stop in stops:
            if (stop.get("stop_id") or stop.get("poi_id")) == "griffith_observatory":
                stop.update(duration_constraint=duration, visit_duration_minutes=45)
            if (stop.get("stop_id") or stop.get("poi_id")) == "golden_gate_bridge":
                stop.update(
                    time_window_constraint=window,
                    itinerary_role="activity",
                    itinerary_role_source="user_declared_itinerary_role",
                )
        identity = stable_content_hash({"role": role, "original": plan.content_hash, "rules": rules})
        return replace(
            plan,
            plan_id=f"plan_{identity}",
            parent_plan_id=parent_id,
            selected_stops=tuple(stops),
            owned_constraints=rules,
        )

    parent = owned_plan(base.parent_plan, "parent")
    children = {
        bundle.metrics["role"]: owned_plan(bundle.child_plan, bundle.metrics["role"], parent.plan_id)
        for bundle in package.evidence_bundles.values()
    }
    planner_runs = {role: builder._planner_run(role, child) for role, child in children.items()}
    config = PlanEvaluatorConfig(
        strict_routes=True,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
        enforce_opening_windows=False,
        start_anchor_by_day=dict(day_config.start_anchor_by_day),
        end_anchor_by_day=dict(day_config.end_anchor_by_day),
        require_lodging_assignments=True,
    )
    diffs, certificates, explanations = {}, {}, {}
    for role, child in children.items():
        diffs[role] = compute_plan_diff(parent, child).to_record()
        certificate = PlanEvaluator(
            route_matrix=matrix,
            planner_runs={planner_runs[role].run_id: planner_runs[role]},
            config=config,
            reference_plan=parent,
        ).evaluate_final_plan(child)
        assert certificate.comparison_eligibility == "eligible", certificate.failures
        certificates[role] = certificate.to_record()
        explanations[role] = builder._explanation(role, parent, child, diffs[role], certificate.content_hash)
    package_dir = root / "owned_product_package"
    paths = builder._write_artifacts(
        package_dir,
        parent=parent,
        children=children,
        diffs=diffs,
        certificates=certificates,
        explanations=explanations,
        matrix_record=base.route_matrix,
        request=builder._request(),
        planner_runs=planner_runs,
    )
    manifest = builder._manifest(
        parent=parent,
        children=children,
        diffs=diffs,
        certificates=certificates,
        explanations=explanations,
        matrix_record=base.route_matrix,
        paths=paths,
        matrix_file_sha256=hashlib.sha256((package_dir / paths["route_matrix"]).read_bytes()).hexdigest(),
    )
    manifest["artifacts_sha256"] = {
        path.relative_to(package_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_dir.rglob("*"))
        if path.is_file()
    }
    builder._write_json(package_dir / "manifest.json", manifest)
    # Require the real strict loader to validate the exported test inputs.
    load_product_demo_package(repository_root, package_dir)
    registry = json.loads((ROOT / "configs/product_app_registry.json").read_text(encoding="utf-8"))
    record = next(record for record in registry["runs"] if record["run_id"] == RUN_ID)
    record.update(
        relative_path=package_dir.relative_to(repository_root).as_posix(),
        default=True,
        manifest_sha256=hashlib.sha256((package_dir / "manifest.json").read_bytes()).hexdigest(),
    )
    registry["runs"] = [record]
    registry_path = root / "owned_registry.json"
    builder._write_json(registry_path, registry)
    return registry_path
