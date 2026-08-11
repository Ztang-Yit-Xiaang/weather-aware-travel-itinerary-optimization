from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
PARENT_ID = "plan_e1c4f803691e3188"


def _compiler(*, evidence_bundles: dict | None = None) -> FrozenDraftCompiler:
    package = load_product_demo_package(ROOT, RUN)
    return FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        evidence_bundles or package.evidence_bundles,
        repository_root=ROOT,
    )


def _operation(
    target: str = "surf_n_wear_s_beach_house",
    *,
    day: int = 4,
    sequence_index: int = 0,
) -> DraftOperationV1:
    return DraftOperationV1(
        operation_id=f"operation_{target}_{day}_{sequence_index}",
        type="set_stop_order",
        target=target,
        parameters={"day": day, "sequence_index": sequence_index},
        source="typed_stop_editor",
    )


def test_same_day_order_edit_builds_exact_child_diff_certificate_and_route_legs() -> None:
    package = load_product_demo_package(ROOT, RUN)
    parent = package.primary_bundle.parent_plan
    parent_before = deepcopy(parent)
    matrix_before = deepcopy(package.primary_bundle.route_matrix)
    compiler = _compiler()

    compiled = compiler.compile([_operation()], accepted_plan_id=PARENT_ID)

    assert compiled.state == "eligible"
    assert compiled.reason is None
    assert compiled.execution_mode == "typed_direct_edit_independent_evaluation"
    child = compiled.child_plan
    assert child["plan_id"] != PARENT_ID
    assert child["parent_plan_id"] == PARENT_ID
    assert child["content_hash"] != parent["content_hash"]
    assert child["sequence"][3:5] == ["surf_n_wear_s_beach_house", "stearns_wharf"]
    assert next(row for row in child["ordered_days"] if row["day"] == 4)["stop_ids"] == [
        "surf_n_wear_s_beach_house",
        "stearns_wharf",
    ]
    day_four = [row for row in child["selected_stops"] if row["day"] == 4]
    assert [(row["poi_id"], row["stop_order"]) for row in day_four] == [
        ("surf_n_wear_s_beach_house", 1),
        ("stearns_wharf", 2),
    ]
    assert child["day_assignments"]["surf_n_wear_s_beach_house"] == 4
    assert child["modeled_metrics"] == {"selected_attractions": 9.0}
    assert "total_utility" not in child["modeled_metrics"]
    assert "total_travel_time" not in child["modeled_metrics"]
    assert "total_travel_distance_km" not in child["modeled_metrics"]

    diff = compiled.diff
    assert diff["parent_plan_id"] == PARENT_ID
    assert diff["child_plan_id"] == child["plan_id"]
    assert {row["stop_id"] for row in diff["reorder_changes"]} == {
        "stearns_wharf",
        "surf_n_wear_s_beach_house",
    }
    assert len(diff["road_changes"]) == 1
    assert diff["road_changes"][0]["day"] == 4
    assert not diff["added_stops"]
    assert not diff["deleted_stops"]
    assert not diff["day_moves"]
    assert not diff["time_shifts"]

    certificate = compiled.certificate
    assert certificate["plan_id"] == child["plan_id"]
    assert certificate["plan_content_hash"] == child["content_hash"]
    assert certificate["comparison_eligibility"] == "eligible"
    assert certificate["evaluation_status"] == "PASSED"
    assert certificate["route_validation"]["required_leg_count"] == 16
    assert certificate["route_validation"]["road_validated_leg_count"] == 16
    assert certificate["route_validation"]["missing_leg_count"] == 0
    assert certificate["route_validation"]["fallback_leg_count"] == 0
    assert len(compiled.route_legs) == 16
    assert {
        (row["origin_id"], row["destination_id"])
        for row in compiled.route_legs
        if row["day"] == 4
    } == {
        ("the_line_la", "surf_n_wear_s_beach_house"),
        ("surf_n_wear_s_beach_house", "stearns_wharf"),
        ("stearns_wharf", "hotel_milo_santa_barbara"),
    }
    assert parent == parent_before
    assert package.primary_bundle.route_matrix == matrix_before


def test_semantically_identical_order_edits_are_deterministic() -> None:
    compiler = _compiler()
    first = compiler.compile([_operation()], accepted_plan_id=PARENT_ID)
    second_operation = _operation()
    second_operation = DraftOperationV1(
        operation_id="different_session_operation_id",
        type=second_operation.type,
        target=second_operation.target,
        parameters=second_operation.parameters,
        source="copilot_confirmed",
    )
    second = compiler.compile([second_operation], accepted_plan_id=PARENT_ID)

    assert first.source_request_id == second.source_request_id
    assert first.child_plan == second.child_plan
    assert first.diff == second.diff
    assert first.certificate["certificate_id"] == second.certificate["certificate_id"]


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        (_operation(day=5), "draft_order_day_mismatch"),
        (_operation(sequence_index=2), "draft_order_index_invalid"),
        (_operation(sequence_index=1), "draft_no_effect"),
    ],
)
def test_invalid_order_edits_fail_before_artifact_creation(
    operation: DraftOperationV1,
    code: str,
) -> None:
    with pytest.raises(WorkspaceError, match=code) as raised:
        _compiler().compile([operation], accepted_plan_id=PARENT_ID)
    assert raised.value.code == code
    assert raised.value.status_code == 409


def test_order_edit_cannot_be_mixed_with_registered_replacement_pipeline() -> None:
    replacement = DraftOperationV1(
        operation_id="operation_replacement",
        type="replace_nearby",
        target="golden_gate_bridge",
        parameters={"candidate_id": "bixby_creek_bridge_viewpoint"},
        source="test",
    )
    with pytest.raises(
        WorkspaceError,
        match="draft_evaluated_operation_combination_unsupported",
    ):
        _compiler().compile([_operation(), replacement], accepted_plan_id=PARENT_ID)

    duration = DraftOperationV1(
        operation_id="operation_duration",
        type="set_stop_duration",
        target="surf_n_wear_s_beach_house",
        parameters={
            "duration": {
                "mode": "exact",
                "preferred_minutes": 60,
                "minimum_minutes": 60,
                "maximum_minutes": 60,
            }
        },
        source="test",
    )
    with pytest.raises(
        WorkspaceError,
        match="draft_evaluated_operation_combination_unsupported",
    ):
        _compiler().compile([_operation(), duration], accepted_plan_id=PARENT_ID)


def test_reorder_fails_closed_when_one_required_reverse_leg_is_missing() -> None:
    package = load_product_demo_package(ROOT, RUN)
    damaged_bundles = {}
    for plan_id, bundle in package.evidence_bundles.items():
        matrix = deepcopy(bundle.route_matrix)
        matrix["cells"] = [
            row
            for row in matrix["cells"]
            if not (
                row["origin_id"] == "surf_n_wear_s_beach_house"
                and row["destination_id"] == "stearns_wharf"
            )
        ]
        damaged_bundles[plan_id] = bundle.__class__(
            **{**bundle.__dict__, "route_matrix": matrix}
        )

    with pytest.raises(WorkspaceError, match="draft_candidate_route_evidence_missing"):
        _compiler(evidence_bundles=damaged_bundles).compile(
            [_operation()],
            accepted_plan_id=PARENT_ID,
        )


def test_preview_api_binds_revision_draft_lineage_evidence_and_geography(tmp_path: Path) -> None:
    with TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=tmp_path / "state",
            additional_allowed_authorities=("testserver",),
        )
    ) as client:
        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        appended = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_order",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {"day": 4, "sequence_index": 0},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert appended.status_code == 200
        operation_id = appended.json()["operation"]["operation_id"]
        response = client.post(
            f"/api/sessions/{session['session_id']}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        restored = client.get(
            f"/api/sessions/{session['session_id']}",
            headers=headers,
        )

    assert response.status_code == 200
    response_payload = response.json()
    proposal = response_payload["proposal"]
    assert response_payload["session"]["revision"] == proposal["session_revision"]
    assert proposal["provenance"] == "independent_evaluated_direct_edit"
    assert proposal["execution_mode"] == "typed_direct_edit_independent_evaluation"
    assert proposal["session_revision"] == 2
    assert proposal["draft_operation_ids"] == [operation_id]
    assert len(proposal["draft_content_hash"]) == 16
    assert proposal["parent_plan_id"] == PARENT_ID
    assert proposal["parent_plan_content_hash"]
    assert proposal["child_plan_id"] == proposal["diff_identity"]["child_plan_id"]
    assert proposal["parent_plan_id"] == proposal["diff_identity"]["parent_plan_id"]
    assert proposal["diff_id"] == proposal["diff_identity"]["diff_id"]
    assert proposal["diff_content_hash"] == proposal["diff_identity"]["content_hash"]
    assert proposal["child_plan_id"] == proposal["certificate_identity"]["plan_id"]
    assert proposal["child_plan_content_hash"] == proposal["certificate_identity"][
        "plan_content_hash"
    ]
    assert proposal["certificate_id"] == proposal["certificate_identity"]["certificate_id"]
    assert proposal["certificate_content_hash"] == proposal["certificate_identity"][
        "content_hash"
    ]
    assert proposal["route_validation_identity"] == {
        key: proposal["route_validation"][key]
        for key in (
            "matrix_id",
            "context_snapshot_id",
            "source_bundle_id",
            "source_content_sha256",
        )
    }
    geography = proposal["geography_plan"]
    assert geography["plan_id"] == proposal["child_plan_id"]
    assert geography["content_hash"] == proposal["child_plan_content_hash"]
    assert geography["coverage"]["required_leg_count"] == 16
    assert geography["coverage"]["road_validated_leg_count"] == 16
    assert geography["coverage"]["gap_count"] == 0
    route_path = [
        feature["properties"]["node_id"]
        for feature in geography["route_path"]["features"]
    ]
    assert route_path.index("surf_n_wear_s_beach_house") < route_path.index("stearns_wharf")
    assert stable_content_hash(proposal["compiled_request"]["operations"]) != ""
    restored_session = restored.json()["session"]
    assert restored_session["revision"] == proposal["session_revision"]
    assert restored_session["proposal"] == proposal


@pytest.mark.parametrize(
    ("parameters", "code"),
    [
        ({"day": 5, "sequence_index": 0}, "draft_order_day_mismatch"),
        ({"day": 4, "sequence_index": 2}, "draft_order_index_invalid"),
        ({"day": 4, "sequence_index": True}, "draft_order_index_invalid"),
        ({"day": 4, "sequence_index": 1}, "draft_no_effect"),
    ],
)
def test_invalid_order_append_writes_nothing(
    tmp_path: Path,
    parameters: dict,
    code: str,
) -> None:
    with TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=tmp_path / code,
            additional_allowed_authorities=("testserver",),
        )
    ) as client:
        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        response = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_order",
                "target": "surf_n_wear_s_beach_house",
                "parameters": parameters,
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        restored = client.get(
            f"/api/sessions/{session['session_id']}",
            headers=headers,
        )

    assert response.status_code == 409
    assert response.json() == {"detail": code}
    assert restored.status_code == 200
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []


def test_mixed_order_impact_matches_compiler_rejection(tmp_path: Path) -> None:
    with TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=tmp_path / "mixed",
            additional_allowed_authorities=("testserver",),
        )
    ) as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        order = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_order",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {"day": 4, "sequence_index": 0},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert order.status_code == 200
        day = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 1,
                "type": "set_stop_day",
                "target": "griffith_observatory",
                "parameters": {"day": 4},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert day.status_code == 200
        impact = client.post(
            f"/api/sessions/{session_id}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 2},
        )
        preview = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 2},
        )

    assert impact.status_code == 200
    summary = impact.json()["summary"]
    assert summary == {
        "operation_count": 2,
        "evaluated_executable_count": 2,
        "draft_only_count": 0,
        "can_run_evaluated_preview": False,
        "blocking_codes": ["draft_evaluated_operation_combination_unsupported"],
    }
    assert preview.status_code == 409
    assert preview.json() == {
        "detail": "draft_evaluated_operation_combination_unsupported"
    }


def test_order_preview_does_not_mutate_frozen_product_artifacts(tmp_path: Path) -> None:
    paths = tuple(
        path
        for path in (RUN / "alternatives").rglob("*.json")
        if path.is_file()
    )
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    compiler = _compiler()

    compiler.compile([_operation()], accepted_plan_id=PARENT_ID)

    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
