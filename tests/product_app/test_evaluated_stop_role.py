from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.evaluated_stop_edits import (
    EvaluatedStopRoleCompiler,
    build_role_constraint_evidence,
)
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.service import _attach_role_proposal_evidence
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
PARENT_ID = "plan_e1c4f803691e3188"
TARGET = "stearns_wharf"
ROLE_SOURCE = "user_declared_itinerary_role"


def _operation(role: str = "meal", target: str = TARGET) -> DraftOperationV1:
    return DraftOperationV1(
        operation_id=f"operation_role_{target}_{role}",
        type="set_stop_role",
        target=target,
        parameters={"role": role},
        source="typed_stop_editor",
    )


def _compiler() -> FrozenDraftCompiler:
    package = load_product_demo_package(ROOT, RUN)
    return FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        package.evidence_bundles,
        repository_root=ROOT,
    )


def _direct(parent=None) -> EvaluatedStopRoleCompiler:
    compiler = _compiler()
    route_matrix, day_config, _ = compiler._runtime_inputs()
    return EvaluatedStopRoleCompiler(
        parent=parent or compiler._parent_artifact,
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
        day_start_time=day_config.day_start_time,
    )


def _client(state_root: Path) -> TestClient:
    return TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=state_root,
            additional_allowed_authorities=("testserver",),
        )
    )


def _constraint(
    *,
    relation: str,
    value,
    strength: str = "locked",
    relaxation_policy: str | None = None,
) -> dict:
    return {
        "constraint_id": f"constraint_{relation}_{strength}",
        "origin": "user",
        "strength": strength,
        "scope": "stop",
        "target_id": TARGET,
        "relation": relation,
        "value": value,
        "confirmed": True,
        "relaxation_policy": relaxation_policy
        or ("never" if strength == "locked" else "always"),
        "evidence_refs": [],
        "schema_version": "owned-constraint-v1",
    }


def test_role_edit_builds_v4_diff_and_only_changes_trip_role_fields() -> None:
    package = load_product_demo_package(ROOT, RUN)
    parent_before = deepcopy(package.primary_bundle.parent_plan)
    compiled = _compiler().compile([_operation()], accepted_plan_id=PARENT_ID)

    assert compiled.state == "eligible"
    child = compiled.child_plan
    assert child["plan_id"] != PARENT_ID
    assert child["parent_plan_id"] == PARENT_ID
    assert child["content_hash"] != parent_before["content_hash"]
    assert child["modeled_metrics"] == {}
    assert child["context_exposure_components"] == {}
    assert child["change_components"] == {}
    parent_by_id = {row["poi_id"]: row for row in parent_before["selected_stops"]}
    child_by_id = {row["poi_id"]: row for row in child["selected_stops"]}
    for stop_id, parent_stop in parent_by_id.items():
        child_stop = child_by_id[stop_id]
        if stop_id == TARGET:
            reduced = dict(child_stop)
            assert reduced.pop("itinerary_role") == "meal"
            assert reduced.pop("itinerary_role_source") == ROLE_SOURCE
            assert reduced == parent_stop
        else:
            assert child_stop == parent_stop

    diff = compiled.diff
    assert diff["schema_version"] == "plan-diff-v4"
    assert "duration_changes" not in diff
    assert "time_window_changes" not in diff
    assert diff["time_shifts"] == []
    assert diff["road_changes"] == []
    assert diff["role_changes"] == [
        {
            "stop_id": TARGET,
            "day": 4,
            "from_role": None,
            "to_role": "meal",
            "from_source": "unavailable",
            "to_source": ROLE_SOURCE,
            "owner_strength": "",
            "cost": 0.25,
        }
    ]
    assert diff["weighted_edit_cost"] == 0.25
    assert compiled.parent_route_legs == compiled.route_legs
    assert len(compiled.route_legs) == 16
    impact = compiled.schedule_impact
    assert impact["schema_version"] == "evaluated-role-impact-v1"
    payload = dict(impact)
    assert payload.pop("content_hash") == stable_content_hash(payload)
    assert all(impact["invariance"].values())
    assert impact["semantic_scope"] == {
        "itinerary_role_semantics": "trip_specific_user_declared_use",
        "place_identity_unchanged": True,
        "place_categories_unchanged": True,
        "route_schedule_effect": "none_for_supported_visit_roles",
        "semantic_fit_claim": "unavailable",
        "recommendation_claim": "unavailable",
    }
    assert impact["parent_schedule"]["metrics"] == impact["child_schedule"]["metrics"]
    assert package.primary_bundle.parent_plan == parent_before


def test_role_owned_constraints_are_attribute_independent_and_value_aware() -> None:
    parent = _compiler()._parent_artifact
    existence_locked = replace(
        parent,
        owned_constraints=(_constraint(relation="must_keep", value=True),),
    )
    existence_result = _direct(existence_locked).compile((_operation().as_dict(),))
    existence_change = existence_result.diff.role_changes[0]
    assert existence_change.owner_strength == ""
    assert existence_change.cost == 0.25

    role_locked_same = replace(
        parent,
        owned_constraints=(_constraint(relation="role", value="meal"),),
    )
    matching_result = _direct(role_locked_same).compile((_operation().as_dict(),))
    matching_change = matching_result.diff.role_changes[0]
    assert matching_change.owner_strength == "locked"
    assert matching_change.cost == 250.0

    role_locked_other = replace(
        parent,
        owned_constraints=(_constraint(relation="role", value="activity"),),
    )
    with pytest.raises(WorkspaceError, match="role_edit_permission_required"):
        _direct(role_locked_other).compile((_operation().as_dict(),))

    role_soft_other = replace(
        parent,
        owned_constraints=(
            _constraint(relation="role", value="activity", strength="soft"),
        ),
    )
    soft_result = _direct(role_soft_other).compile((_operation().as_dict(),))
    soft_change = soft_result.diff.role_changes[0]
    assert soft_change.owner_strength == "soft"
    assert soft_change.cost == 2.5
    assert [warning.code for warning in soft_result.certificate.warnings] == [
        "owned_role_constraint_unsatisfied",
        "opening_window_evidence_missing",
    ]

    role_strong_never = replace(
        parent,
        owned_constraints=(
            _constraint(
                relation="role",
                value="activity",
                strength="strong",
                relaxation_policy="never",
            ),
        ),
    )
    with pytest.raises(WorkspaceError, match="role_edit_permission_required"):
        _direct(role_strong_never).compile((_operation().as_dict(),))


def test_role_constraint_evidence_is_sanitized_deterministic_and_contradiction_safe() -> None:
    parent = _compiler()._parent_artifact
    constrained = replace(
        parent,
        owned_constraints=(
            _constraint(relation="must_keep", value=True),
            _constraint(
                relation="role",
                value="activity",
                strength="soft",
            ),
        ),
    )

    evidence = build_role_constraint_evidence(constrained)
    payload = dict(evidence)
    assert payload.pop("content_hash") == stable_content_hash(payload)
    assert evidence["parent_plan_id"] == constrained.plan_id
    assert evidence["parent_plan_content_hash"] == constrained.content_hash
    assert evidence["constraints"] == [
        {
            "constraint_id": "constraint_role_soft",
            "target_stop_id": TARGET,
            "required_role": "activity",
            "strength": "soft",
            "scope": "stop",
            "relation": "role",
            "relaxation_policy": "always",
            "permission_semantics": "weighted_mismatch_allowed",
        }
    ]
    assert "origin" not in evidence["constraints"][0]
    assert "evidence_refs" not in evidence["constraints"][0]

    contradictory = replace(
        parent,
        owned_constraints=(
            _constraint(relation="role", value="meal", strength="soft"),
            {
                **_constraint(
                    relation="itinerary_role",
                    value="activity",
                    strength="strong",
                ),
                "constraint_id": "other_role_constraint",
            },
        ),
    )
    with pytest.raises(
        WorkspaceError,
        match="draft_parent_role_constraint_invalid",
    ):
        build_role_constraint_evidence(contradictory)


def test_unsupported_structural_role_and_mixed_edits_fail_stable() -> None:
    compiler = _compiler()
    with pytest.raises(WorkspaceError, match="stop_role_evaluation_not_supported"):
        compiler.compile([_operation("lodging")], accepted_plan_id=PARENT_ID)
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
        source="typed_stop_editor",
    )
    with pytest.raises(
        WorkspaceError,
        match="draft_evaluated_operation_combination_unsupported",
    ):
        compiler.compile([_operation(), duration], accepted_plan_id=PARENT_ID)


def test_explicit_role_no_effect_and_malformed_parent_fail_closed() -> None:
    parent = _compiler()._parent_artifact
    stops = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row["itinerary_role"] = "meal"
            row["itinerary_role_source"] = ROLE_SOURCE
        stops.append(row)
    explicit_parent = replace(parent, selected_stops=tuple(stops))
    with pytest.raises(WorkspaceError, match="draft_no_effect"):
        _direct(explicit_parent).compile((_operation().as_dict(),))

    malformed = [dict(stop) for stop in parent.selected_stops]
    malformed[0]["itinerary_role"] = "meal"
    malformed[0]["itinerary_role_source"] = "place_category_inference"
    with pytest.raises(WorkspaceError, match="draft_parent_role_invalid"):
        _direct(replace(parent, selected_stops=tuple(malformed))).compile(
            (_operation().as_dict(),)
        )


def test_workspace_role_lock_rejects_before_append_but_existence_lock_does_not(
    tmp_path: Path,
) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    session, _ = store.create_session("run", "plan", 4)
    payload = {
        "expected_revision": 0,
        "type": "set_stop_role",
        "target": TARGET,
        "parameters": {"role": "meal"},
        "source": "test",
        "evidence_refs": [],
    }
    common = {
        "valid_stop_ids": {TARGET},
        "day_count": 7,
        "parent_stop_ids": {TARGET},
        "parent_day_by_stop": {TARGET: 4},
        "parent_order_by_day": {4: (TARGET,)},
        "parent_role_by_stop": {
            TARGET: {"itinerary_role": None, "itinerary_role_source": None}
        },
    }
    added = store.add_operation(
        session,
        payload,
        role_constraints_by_stop={},
        **common,
    )
    assert added.parameters == {"role": "meal"}
    assert session.revision == 1

    second, _ = store.create_session("run", "plan", 4)
    payload["expected_revision"] = 0
    with pytest.raises(WorkspaceError, match="role_edit_permission_required"):
        store.add_operation(
            second,
            payload,
            role_constraints_by_stop={
                TARGET: (
                    {
                        "constraint_id": "role_lock",
                        "strength": "hard",
                        "value": "activity",
                        "permission_semantics": (
                            "explicit_permission_required_for_mismatch"
                        ),
                    },
                )
            },
            **common,
        )
    assert second.revision == 0
    assert second.draft == []


def test_role_api_exposes_v4_impact_geography_and_w5_block(tmp_path: Path) -> None:
    with _client(tmp_path / "api") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        workspace_role_constraints = created["workspace"][
            "role_constraint_evidence"
        ]
        appended = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "meal"},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        impact = client.post(
            f"/api/sessions/{session_id}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        preview = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        accept = client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": 2},
        )

    assert appended.status_code == 200
    assert impact.status_code == 200
    impact_row = impact.json()["operations"][0]
    assert impact_row["changed_attributes"] == ["itinerary_role"]
    assert impact_row["feedback_tier"] == "evaluated"
    assert impact_row["preview_executable"] is True
    assert preview.status_code == 200
    proposal = preview.json()["proposal"]
    assert proposal["plan_diff"]["schema_version"] == "plan-diff-v4"
    assert proposal["role_impact"]["schema_version"] == "evaluated-role-impact-v1"
    assert proposal["role_impact"]["role_constraint_evidence"] == (
        workspace_role_constraints
    )
    assert proposal["decision_eligible"] is True
    assert proposal["ranking_eligible"] is False
    assert proposal["acceptance_eligible"] is False
    assert proposal["repair"]["tradeoffs"]["utility_retained"] is None
    assert proposal["repair"]["tradeoffs"]["weather_risk_delta"] is None
    stop_feature = next(
        feature
        for feature in proposal["geography_plan"]["stops"]["features"]
        if feature["properties"]["stop_id"] == TARGET
    )
    assert stop_feature["properties"]["role"] == "draft_preview"
    assert stop_feature["properties"]["itinerary_role"] == "meal"
    assert stop_feature["properties"]["itinerary_role_source"] == ROLE_SOURCE
    assert proposal["route_validation"]["required_leg_count"] == 16
    assert accept.status_code == 409
    assert accept.json() == {"detail": "acceptance_not_enabled_until_w5"}


def test_structural_role_api_stays_draft_only_with_specific_blocker(
    tmp_path: Path,
) -> None:
    with _client(tmp_path / "draft_only") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        appended = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "lodging"},
            },
        )
        impact = client.post(
            f"/api/sessions/{session_id}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        preview = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )

    assert appended.status_code == 200
    row = impact.json()["operations"][0]
    assert row["feedback_tier"] == "draft_only"
    assert row["preview_executable"] is False
    assert row["blocking_codes"] == ["stop_role_evaluation_not_supported"]
    assert impact.json()["summary"]["blocking_codes"] == [
        "stop_role_evaluation_not_supported"
    ]
    assert preview.status_code == 409
    assert preview.json() == {"detail": "stop_role_evaluation_not_supported"}


def test_role_api_stale_write_changes_nothing_and_restart_restores_proposal(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "persistent"
    with _client(state_root) as first:
        created = first.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        added = first.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "meal"},
            },
        )
        duplicate = first.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 1,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "meal"},
            },
        )
        stale = first.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "activity"},
            },
        )
        after_stale = first.get(
            f"/api/sessions/{session_id}", headers=headers
        ).json()["session"]
        preview = first.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )

    assert added.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "draft_conflicting_attribute_edits"}
    assert stale.status_code == 409
    assert stale.json() == {"detail": "stale_session_revision"}
    assert after_stale["revision"] == 1
    assert len(after_stale["draft"]) == 1
    assert after_stale["draft"][0]["parameters"] == {"role": "meal"}
    assert preview.status_code == 200
    saved_proposal = preview.json()["proposal"]

    with _client(state_root) as restarted:
        restored = restarted.get(
            f"/api/sessions/{session_id}", headers=headers
        )

    assert restored.status_code == 200, restored.json()
    restored_session = restored.json()["session"]
    assert restored_session["revision"] == 2
    assert restored_session["draft"] == after_stale["draft"]
    assert restored_session["proposal"] == saved_proposal


def test_restart_rejects_persisted_duplicate_role_operations_without_rewrite(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "tampered_duplicate"
    with _client(state_root) as first:
        created = first.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        added = first.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "meal"},
            },
        )
    assert added.status_code == 200

    snapshot_path = state_root / "sessions" / f"{session_id}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    duplicate = deepcopy(snapshot["session"]["draft"][0])
    duplicate["operation_id"] = "operation_" + "f" * 32
    snapshot["session"]["draft"].append(duplicate)
    snapshot["session"]["revision"] = 2
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    tampered_bytes = snapshot_path.read_bytes()

    with _client(state_root) as restarted:
        restored = restarted.get(
            f"/api/sessions/{session_id}", headers=headers
        )

    assert restored.status_code == 409
    assert restored.json() == {"detail": "draft_conflicting_attribute_edits"}
    assert snapshot_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize(
    "tamper",
    [
        "impact",
        "certificate",
        "certificate_hash",
        "certificate_timestamp",
        "diff",
        "geography",
    ],
)
def test_restart_revalidates_persisted_role_proposal_without_rewrite(
    tmp_path: Path,
    tamper: str,
) -> None:
    state_root = tmp_path / f"tampered_{tamper}"
    with _client(state_root) as first:
        created = first.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        first.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": TARGET,
                "parameters": {"role": "meal"},
            },
        ).raise_for_status()
        first.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        ).raise_for_status()

    snapshot_path = state_root / "sessions" / f"{session_id}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    proposal = snapshot["session"]["proposal"]
    if tamper == "impact":
        proposal["role_impact"]["semantic_scope"]["recommendation_claim"] = (
            "validated"
        )
        impact_payload = dict(proposal["role_impact"])
        impact_payload.pop("content_hash")
        forged_hash = stable_content_hash(impact_payload)
        proposal["role_impact"]["content_hash"] = forged_hash
        proposal["role_impact_identity"]["content_hash"] = forged_hash
    elif tamper == "certificate":
        proposal["certificate_role_evidence"]["warning_codes"].append(
            "forged_role_warning"
        )
        evidence_payload = dict(proposal["certificate_role_evidence"])
        evidence_payload.pop("content_hash")
        proposal["certificate_role_evidence"]["content_hash"] = (
            stable_content_hash(evidence_payload)
        )
    elif tamper == "certificate_hash":
        forged_hash = "0" * 16
        proposal["certificate_content_hash"] = forged_hash
        proposal["certificate_identity"]["content_hash"] = forged_hash
        proposal["certificate_role_evidence"][
            "certificate_content_hash"
        ] = forged_hash
        proposal["role_impact_identity"][
            "certificate_content_hash"
        ] = forged_hash
        evidence_payload = dict(proposal["certificate_role_evidence"])
        evidence_payload.pop("content_hash")
        proposal["certificate_role_evidence"]["content_hash"] = (
            stable_content_hash(evidence_payload)
        )
    elif tamper == "certificate_timestamp":
        certificate_record = proposal["certificate_role_evidence"][
            "certificate_record"
        ]
        certificate_record["evaluated_at"] = "2099-01-01T00:00:00+00:00"
        certificate_payload = dict(certificate_record)
        certificate_payload.pop("content_hash")
        forged_hash = stable_content_hash(certificate_payload)
        certificate_record["content_hash"] = forged_hash
        proposal["certificate_content_hash"] = forged_hash
        proposal["certificate_identity"]["content_hash"] = forged_hash
        proposal["certificate_role_evidence"][
            "certificate_content_hash"
        ] = forged_hash
        proposal["role_impact_identity"][
            "certificate_content_hash"
        ] = forged_hash
        evidence_payload = dict(proposal["certificate_role_evidence"])
        evidence_payload.pop("content_hash")
        proposal["certificate_role_evidence"]["content_hash"] = (
            stable_content_hash(evidence_payload)
        )
    elif tamper == "diff":
        proposal["plan_diff"]["role_changes"][0]["to_role"] = "activity"
        forged_hash = stable_content_hash(proposal["plan_diff"])
        proposal["diff_content_hash"] = forged_hash
        proposal["diff_identity"]["content_hash"] = forged_hash
    else:
        feature = next(
            row
            for row in proposal["geography_plan"]["stops"]["features"]
            if row["properties"]["stop_id"] == TARGET
        )
        feature["properties"]["itinerary_role"] = "activity"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    tampered_bytes = snapshot_path.read_bytes()

    with _client(state_root) as restarted:
        restored = restarted.get(
            f"/api/sessions/{session_id}", headers=headers
        )

    assert restored.status_code == 409
    assert restored.json() == {"detail": "draft_role_proposal_invalid"}
    assert snapshot_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize(
    "tamper",
    [
        "semantic_scope",
        "parent_role_source",
        "invariance",
        "route_evidence",
        "schedule",
        "certificate_codes",
        "role_constraint_evidence",
    ],
)
def test_role_proposal_rejects_coherently_rehashed_evidence(tamper: str) -> None:
    compiler = _direct()
    compiled = compiler.compile((_operation().as_dict(),))
    impact = deepcopy(compiled.schedule_impact)
    certificate = compiled.certificate.to_record()
    if tamper == "semantic_scope":
        impact["semantic_scope"]["recommendation_claim"] = "validated"
    elif tamper == "parent_role_source":
        impact["parent_role_states"][0]["itinerary_role_source"] = ROLE_SOURCE
    elif tamper == "invariance":
        impact["invariance"]["route_legs_unchanged"] = False
    elif tamper == "route_evidence":
        impact["route_leg_evidence"].pop()
    elif tamper == "schedule":
        metric = next(iter(impact["child_schedule"]["metrics"]))
        impact["child_schedule"]["metrics"][metric] += 1.0
        child_payload = dict(impact["child_schedule"])
        child_payload.pop("content_hash")
        impact["child_schedule"]["content_hash"] = stable_content_hash(
            child_payload
        )
    elif tamper == "certificate_codes":
        certificate["warnings"].append(
            {
                "code": "forged_role_warning",
                "message": "forged",
                "severity": "soft",
                "evidence_refs": [],
            }
        )
        certificate["warning_count"] += 1
        certificate["nonblocking_warning_count"] += 1
    else:
        impact["role_constraint_evidence"]["constraints"] = [
            {
                "constraint_id": "forged",
                "target_stop_id": TARGET,
                "required_role": "meal",
                "strength": "locked",
                "scope": "stop",
                "relation": "role",
                "relaxation_policy": "never",
                "permission_semantics": (
                    "explicit_permission_required_for_mismatch"
                ),
            }
        ]
        constraint_payload = dict(impact["role_constraint_evidence"])
        constraint_payload.pop("content_hash")
        impact["role_constraint_evidence"]["content_hash"] = stable_content_hash(
            constraint_payload
        )
    impact_payload = dict(impact)
    impact_payload.pop("content_hash")
    impact["content_hash"] = stable_content_hash(impact_payload)

    with pytest.raises(WorkspaceError, match="draft_role_evidence_mismatch"):
        _attach_role_proposal_evidence(
            {},
            role_impact=impact,
            diff=compiled.diff.to_record(),
            certificate=certificate,
            parent_plan=compiler._parent.to_record(),
            child_plan=compiled.child_plan.to_record(),
            route_matrix=compiler._route_matrix,
            expected_route_legs=compiled.route_legs,
        )
