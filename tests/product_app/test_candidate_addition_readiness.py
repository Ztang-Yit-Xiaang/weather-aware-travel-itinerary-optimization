"""R03: authentic product additions are scoped to directed route evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.workspace import WorkspaceError

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ("pfeiffer_big_sur_state_park", "point_lobos_state_natural_reserve")


@pytest.fixture(scope="module")
def package():
    return load_product_demo_package(ROOT, ROOT / "runs/california-coast-coherent-demo-v1")


@pytest.fixture
def compiler(package):
    return FrozenDraftCompiler(package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT)


def test_product_add_capability_is_disabled_until_a_scoped_route_is_ready(compiler, tmp_path):
    result = compiler.capabilities()
    (tmp_path / "exact_capabilities.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    addition = result["operations"]["add_candidate"]
    assert addition["enabled"] is addition["preview_executable"] is False
    assert addition["status"] == "blocked_missing_route_evidence"
    assert len(addition["candidate_day_readiness"]) == 14
    assert all(row["preview_executable"] is False for row in addition["candidate_day_readiness"])
    assert "add_candidate" not in result["executable_combinations"][0]["operations"]
    assert result["operations"]["replace_nearby"]["preview_executable"] is True
    assert {row["candidate_id"] for row in result["candidate_choices"]} == set(CANDIDATES)


@pytest.mark.parametrize("candidate_id", CANDIDATES)
@pytest.mark.parametrize("day", range(1, 8))
def test_all_actual_product_additions_remain_truthfully_blocked(compiler, candidate_id, day, tmp_path):
    draft = [{"type": "add_candidate", "target": candidate_id, "parameters": {"day": day}}]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    output = {
        "input": {"operations": draft, "accepted_plan_id": compiler.parent_plan_id},
        "output": {
            "state": result.state,
            "reason": result.reason,
            "runtime_execution": result.runtime_execution,
            "blocking_findings": result.blocking_findings,
            "child_plan": result.child_plan,
            "evaluation_certificate": result.certificate,
        },
        "provider_mode": "not_used_by_compiler",
    }
    (tmp_path / "exact_addition_io.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    assert result.state == "blocked"
    assert result.reason == "blocked_missing_route_evidence"
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.child_plan is result.certificate is None
    readiness = compiler.candidate_addition_readiness(candidate_id, day)
    assert readiness["preview_executable"] is False
    assert readiness["status"] == "blocked_missing_route_evidence"
    assert readiness["parent_plan_id"] == compiler.parent_plan_id
    assert readiness["parent_plan_content_hash"] == compiler.parent_plan["content_hash"]
    assert readiness["context_snapshot_id"] == compiler.parent_plan["context_snapshot_id"]
    assert readiness["comparison_eligibility"] == "not_evaluated"
    parent_stops = next(row["stop_ids"] for row in compiler.parent_plan["ordered_days"] if row["day"] == day)
    assert readiness["route_sequence"][1:-1] == [*parent_stops, candidate_id]
    assert any(
        row["origin_id"] == parent_stops[-1]
        and row["destination_id"] == candidate_id
        and row["code"] == "missing_route_matrix_cell"
        for row in readiness["blocking_findings"]
    )
    output["readiness"] = readiness
    (tmp_path / "exact_addition_io.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize(
    "change,code",
    [
        ({"road_validated": False}, "route_leg_not_road_validated"),
        ({"duration_s": None}, "route_leg_not_road_validated"),
        ({"geometry": ()}, "route_geometry_missing"),
        ({"geometry": ((91.0, -120.0), (36.0, -120.0))}, "route_geometry_invalid"),
        ({"geometry": ((float("nan"), -120.0), (36.0, -120.0))}, "route_geometry_invalid"),
        ({"context_snapshot_id": "wrong_context"}, "route_context_mismatch"),
    ],
)
def test_readiness_rejects_invalid_existing_egress_leg(compiler, change, code):
    matrix, day_config, trip_config = compiler._runtime_inputs()
    pair = (CANDIDATES[0], day_config.end_anchor_by_day[5])
    cells = dict(matrix.cells)
    cells[pair] = replace(cells[pair], **change)
    compiler._runtime_cache = (replace(matrix, cells=cells), day_config, trip_config)
    readiness = compiler.candidate_addition_readiness(CANDIDATES[0], 5)
    assert readiness["preview_executable"] is False
    assert any(
        row["code"] == code and (row["origin_id"], row["destination_id"]) == pair
        for row in readiness["blocking_findings"]
    )


def test_readiness_checks_unchanged_prefix_and_does_not_reuse_stale_result(compiler):
    before = compiler.candidate_addition_readiness(CANDIDATES[0], 5)
    matrix, day_config, trip_config = compiler._runtime_inputs()
    pair = (day_config.start_anchor_by_day[5], "bixby_creek_bridge_viewpoint")
    assert any((row["origin_id"], row["destination_id"]) == pair and row["ready"] for row in before["required_legs"])
    compiler._runtime_cache = (
        replace(matrix, cells={key: value for key, value in matrix.cells.items() if key != pair}),
        day_config,
        trip_config,
    )
    after = compiler.candidate_addition_readiness(CANDIDATES[0], 5)
    assert any(
        (row["origin_id"], row["destination_id"]) == pair and row["code"] == "missing_route_matrix_cell"
        for row in after["blocking_findings"]
    )


def test_readiness_rejects_matrix_from_another_context(compiler):
    matrix, day_config, trip_config = compiler._runtime_inputs()
    compiler._runtime_cache = (replace(matrix, context_snapshot_id="wrong_context"), day_config, trip_config)
    readiness = compiler.candidate_addition_readiness(CANDIDATES[0], 5)
    assert readiness["preview_executable"] is False
    assert readiness["status"] == "blocked_invalid_route_evidence"
    assert any(row["code"] == "route_context_mismatch" for row in readiness["blocking_findings"])


@pytest.mark.parametrize(
    "candidate_id,day,reason",
    [
        ("unregistered", 5, "draft_candidate_not_registered"),
        (CANDIDATES[0], 0, "draft_destination_day_invalid"),
        (CANDIDATES[0], 8, "draft_destination_day_invalid"),
        (CANDIDATES[0], True, "draft_destination_day_invalid"),
    ],
)
def test_readiness_does_not_accept_unregistered_candidates_or_invalid_days(compiler, candidate_id, day, reason):
    with pytest.raises(WorkspaceError, match=reason):
        compiler.candidate_addition_readiness(candidate_id, day)


def test_replacement_still_reaches_actual_evaluator(compiler):
    result = compiler.compile(
        [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": CANDIDATES[0]},
            }
        ],
        accepted_plan_id=compiler.parent_plan_id,
    )
    assert result.state == "eligible"
    assert result.runtime_execution["evaluation_executed"] is True
    assert result.certificate["plan_id"] == result.child_plan["plan_id"]
