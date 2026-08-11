from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def _client(state_root: Path) -> TestClient:
    return TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=state_root,
            additional_allowed_authorities=("testserver",),
        )
    )


def _session(client: TestClient) -> tuple[dict, dict[str, str], dict]:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    payload = response.json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}, payload["workspace"]


def _append(
    client: TestClient,
    session_id: str,
    headers: dict[str, str],
    revision: int,
    operation_type: str,
    target: str,
    parameters: dict,
):
    return client.post(
        f"/api/sessions/{session_id}/draft/operations",
        headers=headers,
        json={
            "expected_revision": revision,
            "type": operation_type,
            "target": target,
            "parameters": parameters,
            "source": "typed_stop_editor",
            "evidence_refs": [],
        },
    )


def test_typed_edit_capabilities_are_closed_and_truthful(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        _, _, workspace = _session(client)

    capabilities = workspace["typed_edit_capabilities"]
    assert capabilities["schema_version"] == "product-typed-edit-capabilities-v1"
    assert set(capabilities["operations"]) == {
        "set_stop_role",
        "set_stop_day",
        "set_stop_order",
        "set_stop_duration",
        "set_stop_time_window",
        "set_stop_commitment",
        "set_attribute_constraint",
        "change_route_preference",
        "report_route_issue",
    }
    assert capabilities["operations"]["set_stop_day"] == {
        "enabled": True,
        "feedback_tier": "evaluated",
        "preview_executable": True,
        "evaluated_repair": True,
        "blocking_code": None,
    }
    assert capabilities["operations"]["set_stop_order"] == {
        "enabled": True,
        "feedback_tier": "evaluated",
        "preview_executable": True,
        "evaluated_repair": True,
        "blocking_code": None,
        "supported_scope": "same_day",
        "sequence_index_base": 0,
    }
    assert capabilities["operations"]["set_stop_duration"] == {
        "enabled": True,
        "feedback_tier": "conditional",
        "preview_executable": True,
        "evaluated_repair": True,
        "blocking_code": None,
        "supported_evaluated_modes": ["exact"],
        "draft_only_modes": ["preferred", "minimum", "maximum", "range"],
        "unsupported_mode_blocking_code": "duration_mode_evaluation_not_supported",
        "scalar_plan_field": "visit_duration_minutes",
        "typed_plan_field": "duration_constraint",
    }
    assert capabilities["operations"]["set_stop_time_window"] == {
        "enabled": True,
        "feedback_tier": "evaluated",
        "preview_executable": True,
        "evaluated_repair": True,
        "blocking_code": None,
        "typed_plan_field": "time_window_constraint",
        "constraint_schema_version": "stop-time-window-constraint-v1",
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
        "combinable_operation_types": ["set_stop_time_window"],
    }
    assert capabilities["operations"]["set_stop_role"] == {
        "enabled": True,
        "feedback_tier": "conditional",
        "preview_executable": True,
        "evaluated_repair": True,
        "blocking_code": None,
        "supported_evaluated_roles": [
            "attraction",
            "activity",
            "meal",
            "rest_stop",
            "scenic_stop",
        ],
        "draft_only_roles": [
            "lodging",
            "transport_hub",
            "route_waypoint",
            "origin",
            "destination",
        ],
        "unsupported_role_blocking_code": "stop_role_evaluation_not_supported",
        "typed_plan_field": "itinerary_role",
        "typed_source_field": "itinerary_role_source",
        "typed_source_value": "user_declared_itinerary_role",
        "combinable_operation_types": ["set_stop_role"],
    }
    assert capabilities["operations"]["change_route_preference"]["enabled"] is False
    assert capabilities["operations"]["change_route_preference"]["blocking_code"] == (
        "route_preference_not_supported"
    )
    assert capabilities["operations"]["set_stop_commitment"]["protected_strengths"] == [
        "must_keep",
        "booked",
    ]


def test_changed_set_stop_day_runs_pipeline_and_reports_no_feasible_child_truthfully() -> None:
    package = load_product_demo_package(ROOT, ROOT / "runs" / "california-coast-product-demo-v2")
    compiler = FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        package.evidence_bundles,
        repository_root=ROOT,
    )
    compiled = compiler.compile(
        [
            DraftOperationV1(
                operation_id="operation_set_stop_day",
                type="set_stop_day",
                target="griffith_observatory",
                parameters={"day": 4},
                source="typed_stop_editor",
            )
        ],
        accepted_plan_id="plan_e1c4f803691e3188",
    )

    assert compiled.state == "ineligible"
    assert compiled.reason == "no_feasible_evaluated_child"
    assert compiled.child_plan is None
    assert compiled.diff is None
    assert compiled.certificate is None


def test_same_day_set_stop_day_is_rejected_before_append(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        session, headers, _ = _session(client)
        response = _append(
            client,
            session["session_id"],
            headers,
            0,
            "set_stop_day",
            "golden_gate_bridge",
            {"day": 7},
        )
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "draft_no_effect"
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []


def test_draft_only_edit_persists_impacts_and_undoes_without_certification(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    first = _client(state_root)
    with first:
        session, headers, _ = _session(first)
        session_id = session["session_id"]
        added = _append(
            first,
            session_id,
            headers,
            0,
            "set_stop_duration",
            "griffith_observatory",
            {
                "duration": {
                    "mode": "range",
                    "preferred_minutes": None,
                    "minimum_minutes": 60,
                    "maximum_minutes": 120,
                }
            },
        )
        assert added.status_code == 200
        operation_id = added.json()["operation"]["operation_id"]

        impact = first.post(
            f"/api/sessions/{session_id}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        assert impact.status_code == 200
        payload = impact.json()
        assert set(payload) == {
            "schema_version",
            "session_id",
            "session_revision",
            "parent_plan_id",
            "parent_plan_content_hash",
            "certification_status",
            "is_certified",
            "operations",
            "summary",
        }
        assert payload["certification_status"] == "not_certified"
        assert payload["is_certified"] is False
        assert payload["operations"] == [
            {
                "operation_id": operation_id,
                "type": "set_stop_duration",
                "target": "griffith_observatory",
                "feedback_tier": "draft_only",
                "preview_executable": False,
                "evaluated_repair": False,
                "changed_attributes": ["duration"],
                "blocking_codes": ["duration_mode_evaluation_not_supported"],
            }
        ]
        assert payload["summary"] == {
            "operation_count": 1,
            "evaluated_executable_count": 0,
            "draft_only_count": 1,
            "can_run_evaluated_preview": False,
            "blocking_codes": ["duration_mode_evaluation_not_supported"],
        }

        evaluated = first.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        assert evaluated.status_code == 409
        assert evaluated.json()["detail"] == "duration_mode_evaluation_not_supported"

    with _client(state_root) as restored:
        snapshot = restored.get(f"/api/sessions/{session_id}", headers=headers)
        assert snapshot.status_code == 200
        assert snapshot.json()["session"]["draft"][0]["operation_id"] == operation_id
        undone = restored.post(
            f"/api/sessions/{session_id}/draft/undo",
            headers=headers,
            json={"expected_revision": 1},
        )
        assert undone.status_code == 200
        assert undone.json()["undone"]["operation_id"] == operation_id
        assert undone.json()["session"]["draft"] == []


@pytest.mark.parametrize(
    ("operation_type", "parameters", "detail", "status"),
    [
        ("set_stop_role", {"role": "hotel"}, "invalid_stop_role", 422),
        (
            "set_stop_duration",
            {
                "duration": {
                    "mode": "exact",
                    "preferred_minutes": 10,
                    "minimum_minutes": 10,
                    "maximum_minutes": 10,
                }
            },
            "invalid_stop_duration",
            422,
        ),
        (
            "set_stop_time_window",
            {"earliest_arrival": "18:30", "latest_departure": "17:30"},
            "invalid_stop_time_window",
            422,
        ),
        (
            "set_stop_commitment",
            {"strength": "booked", "scope_lifetime": "current_repair_session"},
            "commitment_permission_required",
            409,
        ),
        (
            "change_route_preference",
            {"route_leg_id": "route_leg_unknown", "preference": "scenic"},
            "route_preference_not_supported",
            409,
        ),
    ],
)
def test_typed_edit_invalid_or_unsupported_values_fail_closed(
    tmp_path: Path,
    operation_type: str,
    parameters: dict,
    detail: str,
    status: int,
) -> None:
    with _client(tmp_path / operation_type) as client:
        session, headers, _ = _session(client)
        response = _append(
            client,
            session["session_id"],
            headers,
            0,
            operation_type,
            "griffith_observatory",
            parameters,
        )

    assert response.status_code == status
    assert response.json()["detail"] == detail


def test_route_issue_requires_exact_accepted_route_leg_and_remains_user_report(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        session, headers, workspace = _session(client)
        accepted = next(
            plan
            for plan in workspace["geography"]["plans"]
            if plan["plan_id"] == session["accepted_plan_id"]
        )
        route_leg_id = accepted["validated_legs"]["features"][0]["properties"]["route_leg_id"]
        response = _append(
            client,
            session["session_id"],
            headers,
            0,
            "report_route_issue",
            route_leg_id,
            {
                "route_leg_id": route_leg_id,
                "issue_type": "suspected_closure",
                "note": "User-reported observation; not independently verified.",
            },
        )
        assert response.status_code == 200
        impact = client.post(
            f"/api/sessions/{session['session_id']}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 1},
        ).json()
        assert impact["operations"][0]["changed_attributes"] == ["user_route_report"]
        assert impact["operations"][0]["evaluated_repair"] is False

        invalid = _append(
            client,
            session["session_id"],
            headers,
            1,
            "report_route_issue",
            "route_leg_foreign",
            {
                "route_leg_id": "route_leg_foreign",
                "issue_type": "suspected_closure",
                "note": None,
            },
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"] == "invalid_route_issue"


def test_attribute_constraints_are_independent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        session, headers, _ = _session(client)
        first = _append(
            client,
            session["session_id"],
            headers,
            0,
            "set_attribute_constraint",
            "griffith_observatory",
            {
                "attribute": "existence",
                "strength": "strong_preference",
                "value": True,
                "scope_lifetime": "current_repair_session",
            },
        )
        assert first.status_code == 200
        independent = _append(
            client,
            session["session_id"],
            headers,
            1,
            "set_attribute_constraint",
            "griffith_observatory",
            {
                "attribute": "day",
                "strength": "optional",
                "value": 4,
                "scope_lifetime": "current_draft_only",
            },
        )
        assert independent.status_code == 200
        conflict = _append(
            client,
            session["session_id"],
            headers,
            2,
            "set_attribute_constraint",
            "griffith_observatory",
            {
                "attribute": "day",
                "strength": "optional",
                "value": 5,
                "scope_lifetime": "current_draft_only",
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == "draft_conflicting_attribute_edits"


@pytest.mark.parametrize(
    ("first_type", "first_parameters"),
    [
        ("set_stop_day", {"day": 4}),
        (
            "set_attribute_constraint",
            {
                "attribute": "day",
                "strength": "optional",
                "value": 4,
                "scope_lifetime": "current_draft_only",
            },
        ),
    ],
)
def test_legacy_and_attribute_day_edits_conflict_with_typed_day_edits(
    tmp_path: Path,
    first_type: str,
    first_parameters: dict,
) -> None:
    with _client(tmp_path / first_type) as client:
        session, headers, _ = _session(client)
        first = _append(
            client,
            session["session_id"],
            headers,
            0,
            first_type,
            "griffith_observatory",
            first_parameters,
        )
        assert first.status_code == 200
        second_type = "move_day" if first_type == "set_stop_day" else "set_stop_day"
        second = _append(
            client,
            session["session_id"],
            headers,
            1,
            second_type,
            "griffith_observatory",
            {"day": 5},
        )

    assert second.status_code == 409
    assert second.json()["detail"] == "draft_conflicting_day_moves"


def test_legacy_and_typed_same_day_edits_are_semantic_duplicates(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        session, headers, _ = _session(client)
        first = _append(
            client,
            session["session_id"],
            headers,
            0,
            "set_stop_day",
            "griffith_observatory",
            {"day": 4},
        )
        assert first.status_code == 200
        duplicate = _append(
            client,
            session["session_id"],
            headers,
            1,
            "move_day",
            "griffith_observatory",
            {"day": 4},
        )

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "draft_duplicate_day_edit"
