from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.evaluated_stop_edits import EvaluatedStopDurationCompiler
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
PARENT_ID = "plan_e1c4f803691e3188"
TARGET = "stearns_wharf"


def _duration(mode: str = "exact", minutes: int = 90) -> dict:
    if mode == "exact":
        return {
            "mode": mode,
            "preferred_minutes": minutes,
            "minimum_minutes": minutes,
            "maximum_minutes": minutes,
        }
    if mode == "preferred":
        return {
            "mode": mode,
            "preferred_minutes": minutes,
            "minimum_minutes": None,
            "maximum_minutes": None,
        }
    raise AssertionError(mode)


def _operation(*, mode: str = "exact", minutes: int = 90, target: str = TARGET) -> DraftOperationV1:
    return DraftOperationV1(
        operation_id=f"operation_duration_{mode}_{minutes}_{target}",
        type="set_stop_duration",
        target=target,
        parameters={"duration": _duration(mode, minutes)},
        source="typed_stop_editor",
    )


def _compiler() -> FrozenDraftCompiler:
    package = load_product_demo_package(ROOT, RUN)
    return FrozenDraftCompiler(
        package.primary_bundle.parent_plan,
        package.evidence_bundles,
        repository_root=ROOT,
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


def _append(
    client: TestClient,
    session_id: str,
    headers: dict[str, str],
    *,
    revision: int,
    duration: dict,
    target: str = TARGET,
):
    return client.post(
        f"/api/sessions/{session_id}/draft/operations",
        headers=headers,
        json={
            "expected_revision": revision,
            "type": "set_stop_duration",
            "target": target,
            "parameters": {"duration": duration},
            "source": "typed_stop_editor",
            "evidence_refs": [],
        },
    )


def test_exact_duration_builds_versioned_diff_schedule_and_certificate() -> None:
    package = load_product_demo_package(ROOT, RUN)
    parent_before = deepcopy(package.primary_bundle.parent_plan)
    matrix_before = deepcopy(package.primary_bundle.route_matrix)

    compiled = _compiler().compile([_operation()], accepted_plan_id=PARENT_ID)

    assert compiled.state == "eligible"
    assert compiled.execution_mode == "typed_direct_edit_independent_evaluation"
    child = compiled.child_plan
    assert child["plan_id"] != PARENT_ID
    assert child["parent_plan_id"] == PARENT_ID
    assert child["content_hash"] != parent_before["content_hash"]
    assert child["created_at"] == ""
    target = next(row for row in child["selected_stops"] if row["poi_id"] == TARGET)
    assert target["visit_duration_minutes"] == 90
    assert target["duration_constraint"] == _duration()
    assert child["sequence"] == parent_before["sequence"]
    assert child["ordered_days"] == parent_before["ordered_days"]
    assert child["route_ids_by_day"] == parent_before["route_ids_by_day"]
    assert child["modeled_metrics"] == {"selected_attractions": 9.0}

    diff = compiled.diff
    assert diff["schema_version"] == "plan-diff-v2"
    assert not diff["time_shifts"]
    assert not diff["road_changes"]
    assert len(diff["duration_changes"]) == 1
    change = diff["duration_changes"][0]
    assert change == {
        "stop_id": TARGET,
        "day": 4,
        "from_constraint": None,
        "to_constraint": _duration(),
        "from_minutes": None,
        "to_minutes": 90.0,
        "from_source": "unavailable",
        "to_source": "visit_duration_minutes",
        "accounting_from_minutes": 45.0,
        "accounting_to_minutes": 90.0,
        "accounting_from_source": "configured_evaluator_default",
        "accounting_to_source": "visit_duration_minutes",
        "delta_minutes": 45.0,
        "owner_strength": "",
        "cost": 0.25,
    }

    certificate = compiled.certificate
    assert certificate["plan_id"] == child["plan_id"]
    assert certificate["plan_content_hash"] == child["content_hash"]
    assert certificate["comparison_eligibility"] == "eligible"
    assert certificate["evaluation_status"] == "PASSED_WITH_WARNINGS"
    assert [row["code"] for row in certificate["warnings"]] == [
        "opening_window_evidence_missing"
    ]
    assert certificate["route_validation"]["required_leg_count"] == 16
    assert certificate["route_validation"]["road_validated_leg_count"] == 16
    assert certificate["route_validation"]["missing_leg_count"] == 0
    assert certificate["route_validation"]["fallback_leg_count"] == 0
    assert compiled.parent_route_legs == compiled.route_legs

    schedule = compiled.schedule_impact
    assert schedule is not None
    payload = dict(schedule)
    assert payload.pop("content_hash") == stable_content_hash(payload)
    assert schedule["affected_days"] == [4]
    assert schedule["configured_default_visit_minutes"] == 45.0
    assert schedule["duration_cost_policy"]["base_change_cost"] == 0.25
    assert schedule["target_stop_ids"] == [TARGET]
    assert schedule["parent"]["status"] == "incomplete_evidence"
    assert schedule["child"]["status"] == "incomplete_evidence"
    assert len(schedule["parent"]["missing_opening_window_stop_ids"]) == 9
    assert schedule["parent"]["modeled_components"] == [
        "road_travel",
        "visit_duration",
        "opening_wait_when_window_available",
        "day_limit",
    ]
    assert schedule["parent"]["unavailable_components"] == [
        "parking_dropoff",
        "walking_transfer",
        "queue_wait",
        "service_buffer",
    ]
    assert schedule["parent"]["metrics"]["day_4_total_minutes"] == pytest.approx(206.835)
    assert schedule["child"]["metrics"]["day_4_total_minutes"] == pytest.approx(251.835)
    assert package.primary_bundle.parent_plan == parent_before
    assert package.primary_bundle.route_matrix == matrix_before


def test_duration_only_overrun_is_ineligible_with_exact_failed_accounting() -> None:
    package = load_product_demo_package(ROOT, RUN)
    parent_before = deepcopy(package.primary_bundle.parent_plan)
    compiled = _compiler().compile(
        [
            _operation(minutes=480, target="stearns_wharf"),
            _operation(minutes=480, target="surf_n_wear_s_beach_house"),
        ],
        accepted_plan_id=PARENT_ID,
    )

    assert compiled.state == "ineligible"
    assert compiled.reason == "independent_evaluation_failed"
    assert compiled.certificate["evaluation_status"] == "FAILED"
    assert compiled.certificate["comparison_eligibility"] == "ineligible"
    assert "day_time_exceeded" in {
        row["code"] for row in compiled.certificate["failures"]
    }
    assert compiled.certificate["route_validation"]["required_leg_count"] == 16
    assert compiled.certificate["route_validation"]["road_validated_leg_count"] == 16
    assert compiled.certificate["route_validation"]["missing_leg_count"] == 0
    assert compiled.certificate["route_validation"]["fallback_leg_count"] == 0
    assert len(compiled.diff["duration_changes"]) == 2
    assert compiled.schedule_impact is not None
    child_metrics = compiled.schedule_impact["child"]["metrics"]
    assert child_metrics["day_4_total_minutes"] == pytest.approx(1076.835)
    assert child_metrics["day_4_slack_minutes"] == 0.0
    assert child_metrics["day_4_overrun_minutes"] == pytest.approx(356.835)
    assert compiled.schedule_impact["child"]["status"] == "failed"
    assert package.primary_bundle.parent_plan == parent_before


def test_ineligible_duration_preview_api_preserves_evidence_and_w5_block(tmp_path: Path) -> None:
    with _client(tmp_path / "ineligible") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        for revision, target in enumerate(
            ("stearns_wharf", "surf_n_wear_s_beach_house")
        ):
            assert _append(
                client,
                session_id,
                headers,
                revision=revision,
                duration=_duration(minutes=480),
                target=target,
            ).status_code == 200
        response = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 2},
        )
        accept = client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": 3},
        )

    assert response.status_code == 200
    proposal = response.json()["proposal"]
    assert proposal["state"] == "ineligible"
    assert proposal["evaluation_status"] == "FAILED"
    assert proposal["eligibility"] == "ineligible"
    assert proposal["decision_eligible"] is False
    assert proposal["ranking_eligible"] is False
    assert proposal["acceptance_eligible"] is False
    assert proposal["acceptance_blocking_code"] == "acceptance_not_enabled_until_w5"
    assert len(proposal["plan_diff"]["duration_changes"]) == 2
    assert proposal["schedule_impact"]["child"]["status"] == "failed"
    assert proposal["schedule_impact"]["child"]["blocking_codes"] == [
        "day_time_exceeded"
    ]
    assert proposal["schedule_impact"]["child"]["metrics"][
        "day_4_overrun_minutes"
    ] == pytest.approx(356.835)
    assert proposal["certificate_schedule_evidence"]["failure_codes"] == [
        "day_time_exceeded"
    ]
    assert proposal["route_validation"]["required_leg_count"] == 16
    assert proposal["route_validation"]["road_validated_leg_count"] == 16
    assert accept.status_code == 409
    assert accept.json() == {"detail": "acceptance_not_enabled_until_w5"}


def test_missing_to_explicit_default_is_material_but_explicit_exact_same_is_no_effect() -> None:
    first = _compiler().compile([_operation(minutes=45)], accepted_plan_id=PARENT_ID)
    assert first.state == "eligible"
    assert first.diff["duration_changes"][0]["delta_minutes"] == 0.0

    compiler = _compiler()
    parent = compiler._parent_artifact
    stops = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row["duration_constraint"] = _duration(minutes=45)
            row["visit_duration_minutes"] = 45
        stops.append(row)
    explicit_parent = replace(parent, selected_stops=tuple(stops))
    route_matrix, day_config, _ = compiler._runtime_inputs()
    direct = EvaluatedStopDurationCompiler(
        parent=explicit_parent,
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
    )

    with pytest.raises(WorkspaceError, match="draft_no_effect"):
        direct.compile(({
            "type": "set_stop_duration",
            "target": TARGET,
            "parameters": {"duration": _duration(minutes=45)},
        },))


@pytest.mark.parametrize(
    "mutation",
    [
        {"duration_constraint": {"mode": "bad"}},
        {"duration_constraint": _duration(minutes=45)},
        {"visit_duration_minutes": 45, "duration_minutes": 60},
        {"visit_duration_minutes": True},
        {"visit_duration_minutes": float("nan")},
        {"visit_duration_minutes": float("inf")},
        {"visit_duration_minutes": 0},
    ],
)
def test_parent_duration_corruption_fails_closed(mutation: dict) -> None:
    compiler = _compiler()
    parent = compiler._parent_artifact
    stops = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row.update(mutation)
        stops.append(row)
    route_matrix, day_config, _ = compiler._runtime_inputs()
    direct = EvaluatedStopDurationCompiler(
        parent=replace(parent, selected_stops=tuple(stops)),
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
    )

    with pytest.raises(WorkspaceError, match="draft_parent_duration_invalid"):
        direct.compile(({
            "type": "set_stop_duration",
            "target": TARGET,
            "parameters": {"duration": _duration()},
        },))


def test_changed_child_canonicalizes_consistent_legacy_scalar_aliases() -> None:
    compiler = _compiler()
    parent = compiler._parent_artifact
    stops = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row["duration_minutes"] = 45
            row["service_minutes"] = 45
        stops.append(row)
    route_matrix, day_config, _ = compiler._runtime_inputs()
    direct = EvaluatedStopDurationCompiler(
        parent=replace(parent, selected_stops=tuple(stops)),
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
    )

    compiled = direct.compile(({
        "type": "set_stop_duration",
        "target": TARGET,
        "parameters": {"duration": _duration()},
    },))
    changed = next(row for row in compiled.child_plan.selected_stops if row.get("poi_id") == TARGET)
    assert changed["visit_duration_minutes"] == 90
    assert "duration_minutes" not in changed
    assert "service_minutes" not in changed


@pytest.mark.parametrize("value", [True, "45", float("nan"), float("inf"), 0, 14, 481])
def test_invalid_configured_duration_default_fails_closed(value) -> None:
    compiler = _compiler()
    route_matrix, day_config, _ = compiler._runtime_inputs()
    with pytest.raises(WorkspaceError, match="draft_evaluator_duration_config_invalid"):
        EvaluatedStopDurationCompiler(
            parent=compiler._parent_artifact,
            route_matrix=route_matrix,
            start_anchor_by_day=day_config.start_anchor_by_day,
            end_anchor_by_day=day_config.end_anchor_by_day,
            max_day_minutes=day_config.max_day_minutes,
            default_visit_minutes=value,
        )


@pytest.mark.parametrize(
    "duration",
    [
        {**_duration(), "preferred_minutes": True},
        {**_duration(), "preferred_minutes": 90.0},
        {**_duration(), "preferred_minutes": "90"},
        {**_duration(), "preferred_minutes": None},
        _duration(minutes=14),
        _duration(minutes=481),
        {**_duration(), "maximum_minutes": 91},
        {**_duration(), "extra": 1},
    ],
)
def test_invalid_exact_duration_append_writes_nothing(tmp_path: Path, duration: dict) -> None:
    with _client(tmp_path / str(len(repr(duration)))) as client:
        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        response = _append(
            client,
            session["session_id"],
            headers,
            revision=0,
            duration=duration,
        )
        restored = client.get(
            f"/api/sessions/{session['session_id']}",
            headers=headers,
        ).json()["session"]

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_stop_duration"}
    assert restored["revision"] == 0
    assert restored["draft"] == []


def test_nonexact_duration_is_draft_only_and_preview_fails_stably(tmp_path: Path) -> None:
    with _client(tmp_path / "preferred") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        appended = _append(
            client,
            session_id,
            headers,
            revision=0,
            duration=_duration("preferred"),
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
    assert impact.json()["operations"][0]["feedback_tier"] == "draft_only"
    assert impact.json()["summary"] == {
        "operation_count": 1,
        "evaluated_executable_count": 0,
        "draft_only_count": 1,
        "can_run_evaluated_preview": False,
        "blocking_codes": ["duration_mode_evaluation_not_supported"],
    }
    assert preview.status_code == 409
    assert preview.json() == {"detail": "duration_mode_evaluation_not_supported"}


def test_exact_and_draft_only_duration_modes_cannot_be_mixed() -> None:
    with pytest.raises(
        WorkspaceError,
        match="draft_evaluated_operation_combination_unsupported",
    ):
        _compiler().compile(
            [
                _operation(target="stearns_wharf"),
                _operation(mode="preferred", target="surf_n_wear_s_beach_house"),
            ],
            accepted_plan_id=PARENT_ID,
        )


def test_duration_preview_api_exposes_hash_bound_diff_and_schedule(tmp_path: Path) -> None:
    with _client(tmp_path / "proposal") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        appended = _append(
            client,
            session_id,
            headers,
            revision=0,
            duration=_duration(),
        )
        assert appended.status_code == 200
        impact = client.post(
            f"/api/sessions/{session_id}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        response = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )

    assert impact.json()["summary"]["can_run_evaluated_preview"] is True
    assert response.status_code == 200
    proposal = response.json()["proposal"]
    assert proposal["decision_eligible"] is True
    assert proposal["ranking_eligible"] is False
    assert proposal["acceptance_eligible"] is False
    assert proposal["repair"]["tradeoffs"]["utility_retained"] is None
    assert proposal["repair"]["tradeoffs"]["weather_risk_delta"] is None
    assert proposal["plan_diff"]["schema_version"] == "plan-diff-v2"
    assert proposal["plan_diff"]["duration_changes"]
    assert proposal["diff_identity"]["content_hash"] == stable_content_hash(
        proposal["plan_diff"]
    )
    assert proposal["schedule_impact_identity"] == {
        "content_hash": proposal["schedule_impact"]["content_hash"],
        "evaluator_version": proposal["schedule_impact"]["evaluator_version"],
        "route_matrix_id": proposal["schedule_impact"]["route_matrix_id"],
        "parent_plan_id": proposal["schedule_impact"]["parent"]["plan_id"],
        "parent_plan_content_hash": proposal["schedule_impact"]["parent"][
            "plan_content_hash"
        ],
        "child_plan_id": proposal["schedule_impact"]["child"]["plan_id"],
        "child_plan_content_hash": proposal["schedule_impact"]["child"][
            "plan_content_hash"
        ],
        "certificate_id": proposal["certificate_id"],
        "certificate_content_hash": proposal["certificate_content_hash"],
    }
    assert proposal["geography_plan"]["coverage"]["required_leg_count"] == 16
    assert proposal["geography_plan"]["coverage"]["gap_count"] == 0
    certificate_schedule = proposal["certificate_schedule_evidence"]
    certificate_payload = dict(certificate_schedule)
    assert certificate_payload.pop("content_hash") == stable_content_hash(certificate_payload)
    assert certificate_schedule["certificate_id"] == proposal["certificate_id"]
    assert certificate_schedule["warning_codes"] == ["opening_window_evidence_missing"]
    assert certificate_schedule["failure_codes"] == []
    assert certificate_schedule["schedule_metrics"][
        "schedule_missing_opening_window_count"
    ] == 9.0


def test_mixed_duration_draft_impact_matches_preview_rejection(tmp_path: Path) -> None:
    with _client(tmp_path / "mixed") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        assert _append(
            client,
            session_id,
            headers,
            revision=0,
            duration=_duration(),
        ).status_code == 200
        order = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 1,
                "type": "set_stop_order",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {"day": 4, "sequence_index": 0},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert order.status_code == 200
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

    assert impact.json()["summary"]["can_run_evaluated_preview"] is False
    assert impact.json()["summary"]["blocking_codes"] == [
        "draft_evaluated_operation_combination_unsupported"
    ]
    assert preview.status_code == 409
    assert preview.json() == {
        "detail": "draft_evaluated_operation_combination_unsupported"
    }


def test_workspace_explicit_exact_no_effect_and_stale_revision_write_nothing(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    session, _ = store.create_session("run", "plan", 4)
    exact = _duration(minutes=45)
    kwargs = {
        "valid_stop_ids": {TARGET},
        "day_count": 7,
        "parent_stop_ids": {TARGET},
        "parent_day_by_stop": {TARGET: 4},
        "parent_order_by_day": {4: (TARGET,)},
        "parent_duration_by_stop": {
            TARGET: {
                "duration_constraint": exact,
                "visit_duration_minutes": 45,
            }
        },
    }
    payload = {
        "expected_revision": 0,
        "type": "set_stop_duration",
        "target": TARGET,
        "parameters": {"duration": exact},
        "source": "test",
        "evidence_refs": [],
    }
    with pytest.raises(WorkspaceError, match="draft_no_effect"):
        store.add_operation(session, payload, **kwargs)
    assert session.revision == 0
    assert session.draft == []

    changed = deepcopy(payload)
    changed["parameters"] = {"duration": _duration(minutes=60)}
    changed["expected_revision"] = 1
    with pytest.raises(WorkspaceError, match="stale_session_revision"):
        store.add_operation(session, changed, **kwargs)
    assert session.revision == 0
    assert session.draft == []

    wrong_target = deepcopy(payload)
    wrong_target["target"] = "not_in_parent"
    with pytest.raises(WorkspaceError, match="invalid_draft_target"):
        store.add_operation(session, wrong_target, **kwargs)
    assert session.revision == 0
    assert session.draft == []
