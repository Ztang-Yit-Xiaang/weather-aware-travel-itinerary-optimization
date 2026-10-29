"""Exact source and fixture universe for the E2E-1 product vertical slice."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTEST_SETUP_INPUTS = ("tests/conftest.py", "tests/product_app/conftest.py")


def _repository_files() -> tuple[str, ...]:
    paths: set[str] = set()
    for pattern in (
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "poetry.lock",
        "package-lock.json",
        "src/itinerary_system/**/*.py",
        "src/itinerary_system/product_app/static/**/*.css",
        "src/itinerary_system/product_app/static/**/*.html",
        "src/itinerary_system/product_app/static/**/*.js",
        "configs/product_app_registry.json",
        "configs/product_poi_catalogs/**/*.json",
        "runs/california-coast-coherent-demo-v1/**/*.json",
        "runs/california-coast-coherent-demo-v1/**/*.jsonl",
    ):
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.glob(pattern)
            if path.is_file()
        )
    registration = ROOT / "configs/product_route_delta.json"
    if registration.is_file():
        descriptor = (ROOT / json.loads(registration.read_text(encoding="utf-8"))["descriptor_path"]).resolve()
        if not descriptor.is_relative_to((ROOT / "results/route_deltas").resolve()) or not descriptor.is_file():
            raise ValueError("e2e1_route_delta_source_path_invalid")
        paths.add(registration.relative_to(ROOT).as_posix())
        paths.update(path.relative_to(ROOT).as_posix() for path in descriptor.parent.rglob("*") if path.is_file())
    return tuple(sorted(paths))


PRODUCTION_SOURCE_INPUTS = _repository_files()


def deterministic_source_inputs(browser_test: str) -> tuple[str, ...]:
    browser_helpers = (
        ("tests/product_app/browser/test_e2e1_live_copilot_chat.py",)
        if browser_test in {
            "tests/product_app/browser/test_e2e1_vertical_slice_live.py",
            "tests/product_app/browser/test_map_initialization_recovery_live.py",
            "tests/product_app/browser/test_completed_preview_race_live.py",
            "tests/product_app/browser/test_workspace_stability_live.py",
            "tests/product_app/browser/test_direct_edit_journeys_live.py",
            "tests/product_app/browser/test_addition_continuation_live.py",
            "tests/product_app/browser/test_mobile_chat_journey_live.py",
            "tests/product_app/browser/test_waypoint_pointer_live.py",
            "tests/product_app/browser/test_build_coherence_live.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_copilot_combined_conflicts_live.py",
            "tests/product_app/browser/test_copilot_queued_preview_live.py",
            "tests/product_app/browser/test_owned_combined_journey_live.py",
            "tests/product_app/browser/test_capability_controls_live.py",
            "tests/product_app/browser/test_evaluated_variants_live.py",
            "tests/product_app/browser/test_webgl_recovery_live.py",
            "tests/product_app/browser/test_chat_runtime_closure_live.py",
            "tests/product_app/browser/test_same_plan_remount_live.py",
            "tests/product_app/browser/test_waypoint_contexts_live.py",
            "tests/product_app/browser/test_candidate_day_matrix_live.py",
            "tests/product_app/browser/test_bootstrap_recovery_live.py",
            "tests/product_app/browser/test_optional_evidence_fault_live.py",
            "tests/product_app/browser/test_preview_geography_fault_live.py",
        } else ()
    )
    extra_helpers = {
        "tests/product_app/browser/test_copilot_queued_preview_live.py": (
            "tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"),
        "tests/product_app/browser/test_owned_combined_journey_live.py": (
            "tests/product_app/owned_product_fixture.py", "scripts/build_coherent_product_demo.py",
            "tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/control_action_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py", "scripts/run_e2e1_self_managed.py",
            "scripts/verify_e2e1_self_managed.py", "scripts/verify_e2e1_owned_edits.py"),
        "tests/product_app/browser/test_preview_geography_fault_live.py": (
            "tests/product_app/browser/build_coherence_fixture.py", "tests/product_app/browser/test_optional_evidence_fault_live.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_bootstrap_recovery.py", "scripts/verify_e2e1_preview_geography_fault.py"),
        "tests/product_app/browser/test_optional_evidence_fault_live.py": (
            "tests/product_app/browser/build_coherence_fixture.py", "scripts/run_e2e1_self_managed.py",
            "scripts/verify_e2e1_self_managed.py", "scripts/verify_e2e1_bootstrap_recovery.py",
            "scripts/verify_e2e1_optional_evidence_fault.py"),
        "tests/product_app/browser/test_addition_continuation_live.py": (
            "scripts/verify_e2e1_control_actions.py",),
        "tests/product_app/browser/test_bootstrap_recovery_live.py": (
            "tests/product_app/browser/bootstrap_recovery_fixture.py", "tests/product_app/browser/build_coherence_fixture.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py", "scripts/verify_e2e1_bootstrap_recovery.py"),
        "tests/product_app/browser/test_e2e1_vertical_slice_live.py": (
            "tests/product_app/browser/waypoint_ui_fixture.py", "scripts/verify_e2e1_poi_controls.py"),
        "tests/product_app/browser/test_completed_preview_race_live.py": (
            "tests/product_app/browser/waypoint_ui_fixture.py", "scripts/verify_e2e1_network_observations.py"),
        "tests/product_app/browser/test_candidate_day_matrix_live.py": (
            "scripts/verify_e2e1_candidate_day_matrix.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_poi_controls.py", "tests/product_app/browser/waypoint_ui_fixture.py"),
        "tests/product_app/browser/test_waypoint_contexts_live.py": (
            "tests/product_app/browser/waypoint_ui_fixture.py", "scripts/verify_e2e1_waypoint_contexts.py",
            "scripts/verify_e2e1_waypoint_pointer.py", "scripts/verify_e2e1_overlay_release.py",
            "scripts/verify_e2e1_marker_overlay_release.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_network_observations.py"),
        "tests/product_app/browser/test_same_plan_remount_live.py": (
            "scripts/verify_e2e1_same_plan_remount.py", "scripts/verify_e2e1_self_managed.py"),
        "tests/product_app/browser/test_chat_runtime_closure_live.py": (
            "scripts/verify_e2e1_chat_closure.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_network_observations.py", "tests/product_app/browser/waypoint_ui_fixture.py"),
        "tests/product_app/browser/test_evaluated_variants_live.py": (
            "scripts/verify_e2e1_evaluated_variants.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_control_actions.py", "tests/product_app/browser/control_action_fixture.py",
            "scripts/e2e1_interaction_coverage.py", "scripts/e2e1_runtime_gate_coverage.py"),
        "tests/product_app/browser/test_webgl_recovery_live.py": (
            "scripts/verify_e2e1_webgl_recovery.py", "scripts/verify_e2e1_self_managed.py",
            "tests/product_app/browser/waypoint_ui_fixture.py"),
        "tests/product_app/browser/test_capability_controls_live.py": (
            "scripts/verify_e2e1_capability_controls.py", "scripts/verify_e2e1_self_managed.py",
            "scripts/verify_e2e1_control_actions.py", "tests/product_app/browser/control_action_fixture.py",
            "scripts/verify_e2e1_network_observations.py", "tests/product_app/browser/waypoint_ui_fixture.py",
            "scripts/e2e1_interaction_coverage.py", "scripts/e2e1_runtime_gate_coverage.py", "configs/e2e1_interaction_coverage.json"),
        "tests/product_app/browser/test_waypoint_pointer_live.py": (
            "tests/product_app/browser/waypoint_ui_fixture.py", "scripts/verify_e2e1_waypoint_pointer.py",
            "scripts/verify_e2e1_overlay_release.py", "scripts/verify_e2e1_marker_overlay_release.py",
            "scripts/verify_e2e1_network_observations.py"),
        "tests/product_app/browser/test_build_coherence_live.py": (
            "tests/product_app/browser/build_coherence_fixture.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"),
        "tests/product_app/browser/test_copilot_conversation_journeys_live.py": (
            "tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"),
        "tests/product_app/browser/test_copilot_combined_conflicts_live.py": (
            "tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"),
    }.get(browser_test, ())
    return (
        "scripts/e2e1_source_universe.py",
        "scripts/run_product_app.py",
        "scripts/run_e2e1_vertical_slice.py",
        "scripts/e2e1_provider_isolation.py",
        "scripts/e2e1_runtime_environment.py",
        "scripts/build_product_route_delta.py",
        "scripts/verify_e2e1_vertical_slice.py",
        *PYTEST_SETUP_INPUTS,
        browser_test,
        *browser_helpers,
        *extra_helpers,
        *PRODUCTION_SOURCE_INPUTS,
    )


def live_source_inputs(browser_test: str) -> tuple[str, ...]:
    return (
        "scripts/e2e1_source_universe.py",
        "scripts/run_product_app.py",
        "scripts/run_e2e1_vertical_slice.py",
        "scripts/run_e2e1_live_copilot_chat.py",
        "scripts/verify_e2e1_live_copilot_chat.py",
        "scripts/run_live_copilot_prompt_evaluation_v2.py",
        "scripts/e2e1_runtime_environment.py",
        "scripts/build_product_route_delta.py",
        *PYTEST_SETUP_INPUTS,
        browser_test,
        *PRODUCTION_SOURCE_INPUTS,
    )
