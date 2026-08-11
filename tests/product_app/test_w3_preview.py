from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.workspace import WorkspaceError

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
PARENT_ID = "plan_e1c4f803691e3188"
RECOMMENDED_ID = "plan_f5ee52459659dcb5"
LOW_DRIVING_ID = "plan_8aa919c8323dbac0"


def package_and_compiler() -> tuple[object, FrozenDraftCompiler]:
    package = load_product_demo_package(ROOT, RUN)
    compiler = FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        package.evidence_bundles,
        repository_root=ROOT,
    )
    return package, compiler


def operation(
    operation_type: str,
    target: str,
    parameters: dict | None = None,
) -> DraftOperationV1:
    return DraftOperationV1(
        operation_id=f"operation_{operation_type}",
        type=operation_type,
        target=target,
        parameters=parameters or {},
        source="test",
    )


@pytest.mark.parametrize(
    ("candidate_id", "expected_plan_id"),
    [
        ("bixby_creek_bridge_viewpoint", RECOMMENDED_ID),
        ("santa_barbara_museum_of_natural_history_sea_center", LOW_DRIVING_ID),
    ],
)
def test_exact_registered_replacement_compiles_to_its_own_evaluated_child(
    candidate_id: str,
    expected_plan_id: str,
) -> None:
    _, compiler = package_and_compiler()

    compiled = compiler.compile(
        [operation("replace_nearby", "golden_gate_bridge", {"candidate_id": candidate_id})],
        accepted_plan_id=PARENT_ID,
    )

    assert compiled.state == "eligible"
    assert compiled.child_plan["plan_id"] == expected_plan_id
    assert compiled.certificate["comparison_eligibility"] == "eligible"
    assert compiled.certificate["route_validation"]["publication_ready"] is True
    assert compiled.diff["child_plan_id"] == expected_plan_id


def test_capabilities_are_explicit_and_keep_parent_and_candidate_inventories_separate() -> None:
    _, compiler = package_and_compiler()

    capabilities = compiler.capabilities()

    assert capabilities["schema_version"] == "draft-capabilities-v1"
    assert set(capabilities["operations"]) == {
        "keep_stop",
        "lock_stop",
        "mark_flexible",
        "move_day",
        "route_feedback",
        "replace_nearby",
        "add_candidate",
    }
    assert all(row["enabled"] for row in capabilities["operations"].values())
    assert all(row["preview_executable"] for row in capabilities["operations"].values())
    assert all(row["status"] == "deterministic_pipeline" for row in capabilities["operations"].values())
    assert capabilities["executable_combinations"][0]["cardinality"] == 32
    assert "golden_gate_bridge" in capabilities["parent_targets"]
    candidate_ids = {row["candidate_id"] for row in capabilities["candidate_choices"]}
    assert candidate_ids == {
        "bixby_creek_bridge_viewpoint",
        "santa_barbara_museum_of_natural_history_sea_center",
    }
    assert candidate_ids.isdisjoint(capabilities["parent_targets"])
    assert all(row["label"] for row in capabilities["candidate_choices"])


def test_large_route_runtime_is_lazy_and_reused_without_mutating_source() -> None:
    _, compiler = package_and_compiler()
    assert compiler._runtime_cache is None

    compiler.compile(
        [operation("mark_flexible", "golden_gate_bridge")],
        accepted_plan_id=PARENT_ID,
    )
    cached = compiler._runtime_cache
    compiler.compile(
        [operation("route_feedback", "selected_route", {"preference": "reduce_driving"})],
        accepted_plan_id=PARENT_ID,
    )

    assert compiler._runtime_cache is cached


def test_semantically_identical_drafts_are_idempotent_across_session_metadata() -> None:
    _, compiler = package_and_compiler()
    first = DraftOperationV1(
        operation_id="operation_session_a",
        type="route_feedback",
        target="selected_route",
        parameters={"preference": "reduce_driving"},
        source="map",
    )
    second = DraftOperationV1(
        operation_id="operation_session_b",
        type="route_feedback",
        target="selected_route",
        parameters={"preference": "reduce_driving"},
        source="copilot",
    )

    compiled_first = compiler.compile([first], accepted_plan_id=PARENT_ID)
    compiled_second = compiler.compile([second], accepted_plan_id=PARENT_ID)

    assert compiled_first.source_request_id == compiled_second.source_request_id
    assert compiled_first.child_plan["plan_id"] == compiled_second.child_plan["plan_id"]
    assert compiled_first.child_plan["content_hash"] == compiled_second.child_plan["content_hash"]


@pytest.mark.parametrize(
    ("draft", "expected_state", "expected_reason"),
    [
        ([operation("keep_stop", "griffith_observatory")], "ineligible", "no_feasible_evaluated_child"),
        ([operation("lock_stop", "griffith_observatory")], "ineligible", "no_feasible_evaluated_child"),
        ([operation("mark_flexible", "golden_gate_bridge")], "eligible", None),
        ([operation("move_day", "griffith_observatory", {"day": 4})], "ineligible", "no_feasible_evaluated_child"),
        ([operation("route_feedback", "selected_route", {"preference": "reduce_driving"})], "eligible", None),
        ([operation("replace_nearby", "golden_gate_bridge", {"candidate_id": "bixby_creek_bridge_viewpoint"})], "eligible", None),
        ([operation("add_candidate", "bixby_creek_bridge_viewpoint", {"day": 7})], "eligible", None),
    ],
)
def test_all_canonical_operations_run_the_real_pipeline_and_report_truthful_state(
    draft: list[DraftOperationV1],
    expected_state: str,
    expected_reason: str | None,
) -> None:
    _, compiler = package_and_compiler()

    compiled = compiler.compile(draft, accepted_plan_id=PARENT_ID)

    assert compiled.state == expected_state
    assert compiled.reason == expected_reason
    if compiled.child_plan:
        assert compiled.child_plan["parent_plan_id"] == PARENT_ID
        assert compiled.certificate["plan_id"] == compiled.child_plan["plan_id"]


@pytest.mark.parametrize(
    ("draft", "code"),
    [
        ([operation("replace_nearby", "golden_gate_bridge", {"candidate_id": "unknown"})], "draft_candidate_not_registered"),
        ([operation("replace_nearby", "griffith_observatory", {"candidate_id": "bixby_creek_bridge_viewpoint"})], "draft_candidate_target_mismatch"),
        ([operation("keep_stop", "not_a_parent_stop")], "draft_target_not_in_parent"),
        ([operation("route_feedback", "selected_route", {"preference": "invent_a_route"})], "draft_route_feedback_invalid"),
        ([operation("replace_nearby", "golden_gate_bridge", {})], "draft_operation_parameters_invalid"),
    ],
)
def test_invalid_draft_contracts_fail_with_stable_codes(
    draft: list[DraftOperationV1],
    code: str,
) -> None:
    _, compiler = package_and_compiler()
    with pytest.raises(WorkspaceError, match=code) as raised:
        compiler.compile(draft, accepted_plan_id=PARENT_ID)
    assert raised.value.code == code
    assert raised.value.status_code == 409


def test_conflicts_are_reported_before_non_executable_status() -> None:
    _, compiler = package_and_compiler()
    replacement = operation(
        "replace_nearby",
        "golden_gate_bridge",
        {"candidate_id": "bixby_creek_bridge_viewpoint"},
    )

    with pytest.raises(WorkspaceError, match="draft_conflicts_with_replacement"):
        compiler.compile(
            [operation("lock_stop", "golden_gate_bridge"), replacement],
            accepted_plan_id=PARENT_ID,
        )
    with pytest.raises(WorkspaceError, match="draft_conflicting_stop_policy"):
        compiler.compile(
            [
                operation("keep_stop", "golden_gate_bridge"),
                operation("mark_flexible", "golden_gate_bridge"),
            ],
            accepted_plan_id=PARENT_ID,
        )
    with pytest.raises(WorkspaceError, match="draft_duplicate_operation"):
        compiler.compile([replacement, replacement], accepted_plan_id=PARENT_ID)


def test_candidate_certificate_and_diff_are_reverified_not_merely_looked_up() -> None:
    package, _ = package_and_compiler()
    recommended = package.evidence_bundles[RECOMMENDED_ID]
    damaged_certificate = dict(recommended.certificate)
    damaged_certificate["plan_content_hash"] = "wrong"
    compiler = FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        {RECOMMENDED_ID: replace(recommended, certificate=damaged_certificate)},
        repository_root=ROOT,
    )

    with pytest.raises(WorkspaceError, match="draft_pipeline_artifact_mismatch"):
        compiler.compile(
            [
                operation(
                    "replace_nearby",
                    "golden_gate_bridge",
                    {"candidate_id": "bixby_creek_bridge_viewpoint"},
                )
            ],
            accepted_plan_id=PARENT_ID,
        )


def test_service_preview_binds_revision_and_draft_without_mutating_artifacts_or_pointer(
    tmp_path: Path,
) -> None:
    parent_file = RUN / "alternatives" / "w2_weather_recommended_v2" / "plans" / f"{PARENT_ID}.json"
    child_file = RUN / "alternatives" / "w2_weather_low_driving_v2" / "plans" / f"{LOW_DRIVING_ID}.json"
    before = {path: sha256(path.read_bytes()).hexdigest() for path in (parent_file, child_file)}
    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path / "state")
    session, _, _ = service.create_session("california_coast_product_demo_v2")
    service.workspace.add_operation(
        session,
        {
            "expected_revision": 0,
            "type": "replace_nearby",
            "target": "golden_gate_bridge",
            "parameters": {"candidate_id": "santa_barbara_museum_of_natural_history_sea_center"},
            "source": "test",
        },
        valid_stop_ids=service.valid_stops(session.run_id),
        day_count=7,
    )

    proposal = service.preview(session.session_id, expected_revision=1)

    assert proposal["schema_version"] == "draft-preview-v1"
    assert proposal["state"] == "eligible"
    assert proposal["execution_mode"] == "deterministic_repair_pipeline"
    assert proposal["expected_revision"] == 1
    assert len(proposal["draft_content_hash"]) == 16
    assert proposal["parent_plan_id"] == PARENT_ID
    assert proposal["child_plan_id"] == LOW_DRIVING_ID
    assert proposal["certificate_id"] == "cert_5a6deef4c159d346"
    assert proposal["diff_id"] == "diff_ea97896a586cb3af"
    assert proposal["repair"]["result"].endswith("Sea Center.")
    assert proposal["route_validation"]["road_validated_leg_count"] == 16
    assert proposal["geography_plan"]["plan_id"] == LOW_DRIVING_ID
    assert proposal["geography_plan"]["role"] == "draft_preview"
    assert proposal["geography_plan"]["status"] == "ready"
    assert proposal["geography_plan"]["coverage"]["required_leg_count"] == 16
    assert all(
        feature["properties"]["role"] == "draft_preview"
        for collection in ("stops", "route_path", "validated_legs", "gaps")
        for feature in proposal["geography_plan"][collection]["features"]
    )
    assert proposal["evidence"]["comparison_eligibility"] == "eligible"
    assert session.revision == 2
    repeated = service.preview(session.session_id, expected_revision=2)
    assert repeated == proposal
    assert service.workspace.get(session.session_id).revision == 2
    assert not (tmp_path / "state" / "workspaces" / "california_coast_demo" / "pointer.json").exists()
    assert {path: sha256(path.read_bytes()).hexdigest() for path in before} == before


def test_route_feedback_preview_reuses_only_projection_equivalent_route_evidence(
    tmp_path: Path,
) -> None:
    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path / "state")
    session, _, _ = service.create_session("california_coast_product_demo_v2")
    service.workspace.add_operation(
        session,
        {
            "expected_revision": 0,
            "type": "route_feedback",
            "target": "selected_route",
            "parameters": {"preference": "reduce_contextual_risk"},
            "source": "test",
        },
        valid_stop_ids=service.valid_stops(session.run_id),
        day_count=7,
    )

    proposal = service.preview(session.session_id, expected_revision=1)
    geography = proposal["geography_plan"]

    assert proposal["state"] == "eligible"
    assert proposal["child_plan_id"] not in {RECOMMENDED_ID, LOW_DRIVING_ID}
    assert geography["plan_id"] == proposal["child_plan_id"]
    assert geography["content_hash"] == proposal["child_plan_content_hash"]
    assert geography["role"] == "draft_preview"
    assert geography["coverage"]["required_leg_count"] == 16
    assert geography["coverage"]["road_validated_leg_count"] == 16
    assert geography["coverage"]["gap_count"] == 0
    assert all(
        feature["properties"]["plan_id"] == proposal["child_plan_id"]
        and feature["properties"]["content_hash"] == proposal["child_plan_content_hash"]
        for collection in ("stops", "route_path", "validated_legs", "gaps")
        for feature in geography[collection]["features"]
    )


def test_service_preview_rejects_stale_revision_before_compilation(tmp_path: Path) -> None:
    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path / "state")
    session, _, _ = service.create_session("california_coast_product_demo_v2")
    service.workspace.add_operation(
        session,
        {
            "expected_revision": 0,
            "type": "route_feedback",
            "target": "selected_route",
            "parameters": {"preference": "reduce_contextual_risk"},
            "source": "test",
        },
        valid_stop_ids=service.valid_stops(session.run_id),
        day_count=7,
    )

    with pytest.raises(WorkspaceError, match="stale_session_revision"):
        service.preview(session.session_id, expected_revision=0)

    assert session.revision == 1
    assert session.proposal is None
