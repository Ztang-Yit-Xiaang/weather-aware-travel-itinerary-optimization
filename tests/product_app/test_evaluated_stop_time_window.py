from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.evaluated_stop_edits import (
    EvaluatedStopTimeWindowCompiler,
)
from itinerary_system.product_app.models import DraftOperationV1
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.service import _attach_time_window_proposal_evidence
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "california-coast-product-demo-v2"
REGISTRY = ROOT / "configs" / "product_app_registry.json"
PARENT_ID = "plan_e1c4f803691e3188"
TARGET = "stearns_wharf"


def _constraint(
    earliest: str | None = "10:00",
    latest: str | None = None,
) -> dict:
    return {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": earliest,
        "latest_departure": latest,
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }


def _operation(
    *,
    earliest: str | None = "10:00",
    latest: str | None = None,
    target: str = TARGET,
) -> DraftOperationV1:
    return DraftOperationV1(
        operation_id=f"operation_window_{target}_{earliest}_{latest}",
        type="set_stop_time_window",
        target=target,
        parameters={
            "earliest_arrival": earliest,
            "latest_departure": latest,
        },
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
    earliest: str | None = "10:00",
    latest: str | None = None,
    target: str = TARGET,
):
    return client.post(
        f"/api/sessions/{session_id}/draft/operations",
        headers=headers,
        json={
            "expected_revision": revision,
            "type": "set_stop_time_window",
            "target": target,
            "parameters": {
                "earliest_arrival": earliest,
                "latest_departure": latest,
            },
            "source": "typed_stop_editor",
            "evidence_refs": [],
        },
    )


def _target_trace(schedule: dict, target: str = TARGET) -> dict:
    return next(
        trace
        for trace in schedule["stop_traces"]
        if trace["stop_id"] == target
    )


def test_earliest_only_window_builds_v3_diff_trace_and_fresh_certificate() -> None:
    package = load_product_demo_package(ROOT, RUN)
    parent_before = deepcopy(package.primary_bundle.parent_plan)
    matrix_before = deepcopy(package.primary_bundle.route_matrix)

    compiled = _compiler().compile(
        [_operation()],
        accepted_plan_id=PARENT_ID,
    )

    assert compiled.state == "eligible"
    assert compiled.execution_mode == "typed_direct_edit_independent_evaluation"
    child = compiled.child_plan
    assert child["plan_id"] != PARENT_ID
    assert child["parent_plan_id"] == PARENT_ID
    assert child["content_hash"] != parent_before["content_hash"]
    assert child["created_at"] == ""
    target = next(row for row in child["selected_stops"] if row["poi_id"] == TARGET)
    parent_target = next(
        row for row in parent_before["selected_stops"] if row["poi_id"] == TARGET
    )
    assert target["time_window_constraint"] == _constraint()
    assert "time_window_constraint" not in parent_target
    for field in ("opening_start", "opening_end", "arrival_time", "departure_time"):
        assert target.get(field) == parent_target.get(field)
    assert child["sequence"] == parent_before["sequence"]
    assert child["ordered_days"] == parent_before["ordered_days"]
    assert child["route_ids_by_day"] == parent_before["route_ids_by_day"]

    diff = compiled.diff
    assert diff["schema_version"] == "plan-diff-v3"
    assert "duration_changes" not in diff
    assert diff["time_shifts"] == []
    assert diff["road_changes"] == []
    assert diff["weighted_edit_cost"] == 0.25
    assert diff["time_window_changes"] == [
        {
            "stop_id": TARGET,
            "day": 4,
            "from_constraint": None,
            "to_constraint": _constraint(),
            "from_earliest_arrival": None,
            "to_earliest_arrival": "10:00",
            "from_latest_departure": None,
            "to_latest_departure": None,
            "owner_strength": "",
            "cost": 0.25,
        }
    ]

    certificate = compiled.certificate
    assert certificate["plan_id"] == child["plan_id"]
    assert certificate["plan_content_hash"] == child["content_hash"]
    assert certificate["comparison_eligibility"] == "eligible"
    assert certificate["evaluation_status"] == "PASSED_WITH_WARNINGS"
    assert certificate["route_validation"]["required_leg_count"] == 16
    assert certificate["route_validation"]["road_validated_leg_count"] == 16
    assert certificate["route_validation"]["missing_leg_count"] == 0
    assert certificate["route_validation"]["fallback_leg_count"] == 0
    assert compiled.parent_route_legs == compiled.route_legs

    schedule = compiled.schedule_impact
    payload = dict(schedule)
    assert payload.pop("content_hash") == stable_content_hash(payload)
    assert schedule["schema_version"] == "evaluated-time-window-schedule-impact-v1"
    assert schedule["configured_default_visit_minutes"] == 45.0
    assert schedule["configured_day_start_minute"] == 540.0
    assert len(schedule["route_leg_evidence"]) == 16
    assert schedule["affected_days"] == [4]
    assert schedule["target_stop_ids"] == [TARGET]
    assert schedule["parent"]["status"] == "incomplete_evidence"
    assert schedule["child"]["status"] == "incomplete_evidence"
    assert len(schedule["child"]["missing_opening_window_stop_ids"]) == 9
    trace = _target_trace(schedule["child"])
    assert trace["time_window_constraint"] == _constraint()
    assert trace["time_window_source"] == "trip_specific_user_constraint"
    assert trace["opening_window_source"] == "unavailable"
    assert trace["incoming_route_query_hash"]
    assert trace["road_arrival_minute"] <= trace["service_start_minute"]
    assert trace["required_window_wait_minutes"] >= 0
    assert trace["departure_minute"] == pytest.approx(
        trace["service_start_minute"] + trace["visit_minutes"]
    )
    assert trace["visit_duration_source"] in {
        "visit_duration_minutes",
        "configured_evaluator_default",
    }
    assert trace["latest_departure_status"] == "not_set"
    assert trace["latest_departure_overrun_minutes"] is None
    assert package.primary_bundle.parent_plan == parent_before
    assert package.primary_bundle.route_matrix == matrix_before


def test_latest_departure_is_checked_after_visit_with_positive_overrun() -> None:
    compiled = _compiler().compile(
        [_operation(earliest=None, latest="00:01")],
        accepted_plan_id=PARENT_ID,
    )

    assert compiled.state == "ineligible"
    assert compiled.reason == "independent_evaluation_failed"
    assert compiled.certificate["evaluation_status"] == "FAILED"
    assert compiled.certificate["comparison_eligibility"] == "ineligible"
    assert [row["code"] for row in compiled.certificate["failures"]] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    trace = _target_trace(compiled.schedule_impact["child"])
    assert trace["road_arrival_minute"] < trace["departure_minute"]
    assert trace["latest_departure_status"] == "violated"
    assert trace["latest_departure_overrun_minutes"] > 0
    assert trace["latest_departure_overrun_minutes"] == pytest.approx(
        trace["departure_minute"] - 1
    )
    assert trace["failure_codes"] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    assert compiled.schedule_impact["child"]["blocking_codes"] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    assert compiled.certificate["route_validation"]["required_leg_count"] == 16


def test_earliest_wait_can_truthfully_trigger_day_overrun() -> None:
    compiled = _compiler().compile(
        [_operation(earliest="23:59", latest=None)],
        accepted_plan_id=PARENT_ID,
    )

    assert compiled.state == "ineligible"
    trace = _target_trace(compiled.schedule_impact["child"])
    assert trace["required_window_wait_minutes"] > 0
    assert trace["service_start_minute"] == 1439.0
    assert "day_time_exceeded" in compiled.schedule_impact["child"]["blocking_codes"]
    assert compiled.schedule_impact["child"]["metrics"]["day_4_overrun_minutes"] > 0


def test_time_window_preview_api_exposes_exact_bound_evidence_and_w5_block(
    tmp_path: Path,
) -> None:
    with _client(tmp_path / "proposal") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        assert _append(
            client,
            session_id,
            headers,
            revision=0,
        ).status_code == 200
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
        accept = client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": 2},
        )

    assert impact.json()["summary"]["can_run_evaluated_preview"] is True
    assert response.status_code == 200
    proposal = response.json()["proposal"]
    assert proposal["decision_eligible"] is True
    assert proposal["ranking_eligible"] is False
    assert proposal["acceptance_eligible"] is False
    assert proposal["acceptance_blocking_code"] == "acceptance_not_enabled_until_w5"
    assert proposal["plan_diff"]["schema_version"] == "plan-diff-v3"
    assert proposal["diff_identity"]["content_hash"] == stable_content_hash(
        proposal["plan_diff"]
    )
    assert proposal["schedule_impact"]["schema_version"] == (
        "evaluated-time-window-schedule-impact-v1"
    )
    assert proposal["schedule_impact_identity"]["content_hash"] == (
        proposal["schedule_impact"]["content_hash"]
    )
    evidence = proposal["certificate_schedule_evidence"]
    evidence_payload = dict(evidence)
    assert evidence_payload.pop("content_hash") == stable_content_hash(evidence_payload)
    assert evidence["schema_version"] == (
        "evaluated-time-window-certificate-schedule-evidence-v1"
    )
    assert evidence["warning_codes"] == ["opening_window_evidence_missing"]
    assert evidence["failure_codes"] == []
    assert proposal["repair"]["tradeoffs"]["utility_retained"] is None
    assert proposal["repair"]["tradeoffs"]["weather_risk_delta"] is None
    assert proposal["route_validation"]["required_leg_count"] == 16
    assert proposal["route_validation"]["road_validated_leg_count"] == 16
    assert accept.status_code == 409
    assert accept.json() == {"detail": "acceptance_not_enabled_until_w5"}


def test_ineligible_time_window_api_preserves_failed_evidence_and_w5_block(
    tmp_path: Path,
) -> None:
    with _client(tmp_path / "ineligible_proposal") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        assert _append(
            client,
            session_id,
            headers,
            revision=0,
            earliest=None,
            latest="00:01",
        ).status_code == 200
        response = client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": 1},
        )
        accept = client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": 2},
        )

    assert response.status_code == 200
    proposal = response.json()["proposal"]
    assert proposal["state"] == "ineligible"
    assert proposal["decision_eligible"] is False
    assert proposal["ranking_eligible"] is False
    assert proposal["acceptance_eligible"] is False
    schedule = proposal["schedule_impact"]
    assert schedule["child"]["status"] == "failed"
    assert schedule["child"]["blocking_codes"] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    trace = _target_trace(schedule["child"])
    assert trace["latest_departure_status"] == "violated"
    assert trace["latest_departure_overrun_minutes"] > 0
    evidence = proposal["certificate_schedule_evidence"]
    assert evidence["evaluation_status"] == "FAILED"
    assert evidence["comparison_eligibility"] == "ineligible"
    assert evidence["failure_codes"] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    assert evidence["warning_codes"] == ["opening_window_evidence_missing"]
    assert proposal["route_validation"]["required_leg_count"] == 16
    assert proposal["route_validation"]["road_validated_leg_count"] == 16
    assert accept.status_code == 409
    assert accept.json() == {"detail": "acceptance_not_enabled_until_w5"}


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"earliest_arrival": None, "latest_departure": None},
        {"earliest_arrival": "9:00", "latest_departure": None},
        {"earliest_arrival": "24:00", "latest_departure": None},
        {"earliest_arrival": True, "latest_departure": None},
        {"earliest_arrival": "18:00", "latest_departure": "17:00"},
        {
            "earliest_arrival": "10:00",
            "latest_departure": None,
            "opening_start": "09:00",
        },
    ],
)
def test_invalid_time_window_append_writes_nothing(
    tmp_path: Path,
    parameters: dict,
) -> None:
    with _client(tmp_path / stable_content_hash(parameters)) as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        response = client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_time_window",
                "target": TARGET,
                "parameters": parameters,
            },
        )
        snapshot = client.get(f"/api/sessions/{session_id}", headers=headers).json()

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_stop_time_window"}
    assert snapshot["session"]["revision"] == 0
    assert snapshot["session"]["draft"] == []


def test_stale_wrong_target_and_duplicate_target_fail_without_extra_write(
    tmp_path: Path,
) -> None:
    with _client(tmp_path / "fail_closed") as client:
        created = client.post("/api/sessions", json={}).json()
        session_id = created["session"]["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        wrong = _append(
            client,
            session_id,
            headers,
            revision=0,
            target="not_a_parent_stop",
        )
        first = _append(client, session_id, headers, revision=0)
        stale = _append(
            client,
            session_id,
            headers,
            revision=0,
            target="surf_n_wear_s_beach_house",
        )
        duplicate_target = _append(
            client,
            session_id,
            headers,
            revision=1,
            earliest="11:00",
        )
        snapshot = client.get(f"/api/sessions/{session_id}", headers=headers).json()

    assert wrong.status_code == 422
    assert wrong.json() == {"detail": "invalid_draft_target"}
    assert first.status_code == 200
    assert stale.status_code == 409
    assert stale.json() == {"detail": "stale_session_revision"}
    assert duplicate_target.status_code == 409
    assert duplicate_target.json() == {"detail": "draft_conflicting_attribute_edits"}
    assert snapshot["session"]["revision"] == 1
    assert len(snapshot["session"]["draft"]) == 1


def test_explicit_same_constraint_is_no_effect_and_malformed_parent_fails() -> None:
    compiler = _compiler()
    parent = compiler._parent_artifact
    route_matrix, day_config, _ = compiler._runtime_inputs()
    stops = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row["time_window_constraint"] = _constraint()
        stops.append(row)
    explicit_parent = replace(parent, selected_stops=tuple(stops))
    direct = EvaluatedStopTimeWindowCompiler(
        parent=explicit_parent,
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
        day_start_time=day_config.day_start_time,
    )
    with pytest.raises(WorkspaceError, match="draft_no_effect"):
        direct.compile((_operation().as_dict(),))

    malformed = []
    for stop in parent.selected_stops:
        row = dict(stop)
        if row.get("poi_id") == TARGET:
            row["time_window_constraint"] = {
                **_constraint(),
                "latest_departure_semantics": "latest_start",
            }
        malformed.append(row)
    malformed_parent = replace(parent, selected_stops=tuple(malformed))
    invalid = EvaluatedStopTimeWindowCompiler(
        parent=malformed_parent,
        route_matrix=route_matrix,
        start_anchor_by_day=day_config.start_anchor_by_day,
        end_anchor_by_day=day_config.end_anchor_by_day,
        max_day_minutes=day_config.max_day_minutes,
        default_visit_minutes=day_config.default_visit_minutes,
        day_start_time=day_config.day_start_time,
    )
    with pytest.raises(WorkspaceError, match="draft_parent_time_window_invalid"):
        invalid.compile((_operation().as_dict(),))


def test_workspace_canonical_no_effect_rejects_before_append(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    session, _ = store.create_session("run", "plan", 4)
    payload = {
        "expected_revision": 0,
        "type": "set_stop_time_window",
        "target": TARGET,
        "parameters": {
            "earliest_arrival": "10:00",
            "latest_departure": None,
        },
        "source": "test",
        "evidence_refs": [],
    }
    kwargs = {
        "valid_stop_ids": {TARGET},
        "day_count": 7,
        "parent_stop_ids": {TARGET},
        "parent_day_by_stop": {TARGET: 4},
        "parent_order_by_day": {4: (TARGET,)},
        "parent_time_window_by_stop": {TARGET: _constraint()},
    }

    with pytest.raises(WorkspaceError, match="draft_no_effect"):
        store.add_operation(session, payload, **kwargs)

    assert session.revision == 0
    assert session.draft == []


def test_time_window_mixed_with_duration_fails_stable() -> None:
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
        _compiler().compile(
            [_operation(), duration],
            accepted_plan_id=PARENT_ID,
        )


def test_service_rejects_coherently_rehashed_trace_or_blocking_forgery() -> None:
    compiler = _compiler()
    compiled = compiler.compile([_operation()], accepted_plan_id=PARENT_ID)
    route_matrix, _, _ = compiler._runtime_inputs()

    forged_trace = deepcopy(compiled.schedule_impact)
    target_trace = _target_trace(forged_trace["child"])
    target_trace["visit_minutes"] += 1
    target_trace["departure_minute"] += 1
    child_payload = dict(forged_trace["child"])
    child_payload.pop("content_hash")
    forged_trace["child"]["content_hash"] = stable_content_hash(child_payload)
    impact_payload = dict(forged_trace)
    impact_payload.pop("content_hash")
    forged_trace["content_hash"] = stable_content_hash(impact_payload)
    with pytest.raises(WorkspaceError, match="draft_schedule_evidence_mismatch"):
        _attach_time_window_proposal_evidence(
            {},
            schedule_impact=forged_trace,
            diff=compiled.diff,
            certificate=compiled.certificate,
            parent_plan=compiler.parent_plan,
            child_plan=compiled.child_plan,
            route_matrix=route_matrix,
            expected_route_legs=compiled.parent_route_legs,
        )

    forged_arrival = deepcopy(compiled.schedule_impact)
    target_trace = _target_trace(forged_arrival["child"])
    target_trace["road_arrival_minute"] += 1
    target_trace["service_start_minute"] += 1
    target_trace["departure_minute"] += 1
    child_payload = dict(forged_arrival["child"])
    child_payload.pop("content_hash")
    forged_arrival["child"]["content_hash"] = stable_content_hash(child_payload)
    impact_payload = dict(forged_arrival)
    impact_payload.pop("content_hash")
    forged_arrival["content_hash"] = stable_content_hash(impact_payload)
    with pytest.raises(WorkspaceError, match="draft_schedule_evidence_mismatch"):
        _attach_time_window_proposal_evidence(
            {},
            schedule_impact=forged_arrival,
            diff=compiled.diff,
            certificate=compiled.certificate,
            parent_plan=compiler.parent_plan,
            child_plan=compiled.child_plan,
            route_matrix=route_matrix,
            expected_route_legs=compiled.parent_route_legs,
        )

    forged_opening = deepcopy(compiled.schedule_impact)
    target_trace = _target_trace(forged_opening["child"])
    target_trace["opening_start_minute"] = 0.0
    target_trace["opening_wait_minutes"] = 0.0
    target_trace["opening_window_source"] = "plan_stop_fields:opening_start:none"
    forged_opening["child"]["missing_opening_window_stop_ids"].remove(TARGET)
    child_payload = dict(forged_opening["child"])
    child_payload.pop("content_hash")
    forged_opening["child"]["content_hash"] = stable_content_hash(child_payload)
    impact_payload = dict(forged_opening)
    impact_payload.pop("content_hash")
    forged_opening["content_hash"] = stable_content_hash(impact_payload)
    forged_certificate = deepcopy(compiled.certificate)
    forged_certificate["metrics"]["schedule_missing_opening_window_count"] = 8.0
    with pytest.raises(WorkspaceError, match="draft_schedule_evidence_mismatch"):
        _attach_time_window_proposal_evidence(
            {},
            schedule_impact=forged_opening,
            diff=compiled.diff,
            certificate=forged_certificate,
            parent_plan=compiler.parent_plan,
            child_plan=compiled.child_plan,
            route_matrix=route_matrix,
            expected_route_legs=compiled.parent_route_legs,
        )

    forged_route = deepcopy(compiled.schedule_impact)
    forged_route["route_leg_evidence"].pop()
    impact_payload = dict(forged_route)
    impact_payload.pop("content_hash")
    forged_route["content_hash"] = stable_content_hash(impact_payload)
    with pytest.raises(WorkspaceError, match="draft_schedule_evidence_mismatch"):
        _attach_time_window_proposal_evidence(
            {},
            schedule_impact=forged_route,
            diff=compiled.diff,
            certificate=compiled.certificate,
            parent_plan=compiler.parent_plan,
            child_plan=compiled.child_plan,
            route_matrix=route_matrix,
            expected_route_legs=compiled.parent_route_legs,
        )

    forged_blocking = deepcopy(compiled.schedule_impact)
    forged_blocking["child"]["blocking_codes"] = ["invented_failure"]
    forged_blocking["child"]["status"] = "failed"
    child_payload = dict(forged_blocking["child"])
    child_payload.pop("content_hash")
    forged_blocking["child"]["content_hash"] = stable_content_hash(child_payload)
    impact_payload = dict(forged_blocking)
    impact_payload.pop("content_hash")
    forged_blocking["content_hash"] = stable_content_hash(impact_payload)
    with pytest.raises(WorkspaceError, match="draft_schedule_evidence_mismatch"):
        _attach_time_window_proposal_evidence(
            {},
            schedule_impact=forged_blocking,
            diff=compiled.diff,
            certificate=compiled.certificate,
            parent_plan=compiler.parent_plan,
            child_plan=compiled.child_plan,
            route_matrix=route_matrix,
            expected_route_legs=compiled.parent_route_legs,
        )
