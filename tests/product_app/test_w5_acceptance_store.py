"""Fault-oriented tests for the W5 append-only acceptance store."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from itinerary_system.product_app.acceptance import AcceptanceError, AcceptanceStore
from itinerary_system.research_artifacts import PlanArtifactV2


def _plan(plan_id: str, *, parent_plan_id: str | None = None) -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id=plan_id,
        parent_plan_id=parent_plan_id,
        source_run_id="run_demo",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"stop_id": "poi_a", "day": 1, "stop_order": 1},
            {"stop_id": "poi_b", "day": 1, "stop_order": 2},
        ),
        day_assignments={"poi_a": 1, "poi_b": 1},
        sequence=("poi_a", "poi_b"),
        ordered_days=({"day": 1, "stop_ids": ("poi_a", "poi_b")},),
        route_ids_by_day={1: "route_day_1"},
        created_at="2026-08-13T00:00:00+00:00",
    )


def _record(plan: PlanArtifactV2) -> dict[str, object]:
    return plan.to_record(include_content_hash=True)


def _commit(
    store: AcceptanceStore,
    *,
    selected_plan: PlanArtifactV2,
    key: str = "acceptance-key-0001",
):
    original = _plan("plan_original")
    pointer = store.ensure_pointer(
        trip_id="trip_demo",
        run_id="run_demo",
        original_plan=_record(original),
    )
    return store.commit(
        trip_id="trip_demo",
        run_id="run_demo",
        session_id="session_demo",
        session_revision_before=3,
        expected_pointer_revision=pointer["revision"],
        expected_plan_id=pointer["accepted_plan_id"],
        expected_plan_content_hash=pointer["accepted_plan_content_hash"],
        decision_type="accept_repair",
        idempotency_key=key,
        selected_plan=_record(selected_plan),
        snapshot={
            "diff": {"diff_id": "diff_demo"},
            "certificate": {"certificate_id": "cert_demo"},
            "route_legs": [],
            "geography_plan": {"plan_id": selected_plan.plan_id},
            "proposal_identity": {"proposal_id": "proposal_demo"},
        },
        proposal_identity={"proposal_id": "proposal_demo"},
    )


def test_commit_is_append_only_and_idempotent(tmp_path):
    store = AcceptanceStore(tmp_path)
    child = replace(_plan("plan_child", parent_plan_id="plan_original"), sequence=("poi_b", "poi_a"))

    first = _commit(store, selected_plan=child)
    second = store.find_idempotent("trip_demo", "acceptance-key-0001")

    assert second is not None

    assert second.decision == first.decision
    assert second.pointer == first.pointer
    assert len(store.history("trip_demo")) == 1
    assert store.load_snapshot(first.decision["decision_id"])["plan"]["plan_id"] == "plan_child"
    assert store.plan_repository.verify_hash("plan_original")
    assert store.plan_repository.verify_hash("plan_child")


def test_idempotency_key_cannot_select_a_different_plan(tmp_path):
    store = AcceptanceStore(tmp_path)
    original = _plan("plan_original")
    pointer = store.ensure_pointer(
        trip_id="trip_demo", run_id="run_demo", original_plan=_record(original)
    )

    def commit(plan_id: str):
        child = replace(
            _plan(plan_id, parent_plan_id="plan_original"),
            sequence=("poi_b", "poi_a"),
        )
        return store.commit(
            trip_id="trip_demo",
            run_id="run_demo",
            session_id="session_demo",
            session_revision_before=0,
            expected_pointer_revision=pointer["revision"],
            expected_plan_id=pointer["accepted_plan_id"],
            expected_plan_content_hash=pointer["accepted_plan_content_hash"],
            decision_type="accept_repair",
            idempotency_key="same-key-different-plan-0001",
            selected_plan=_record(child),
            snapshot=None,
            proposal_identity=None,
        )

    commit("plan_child_one")
    with pytest.raises(AcceptanceError, match="acceptance_idempotency_conflict"):
        commit("plan_child_two")
    assert len(store.history("trip_demo")) == 1

def test_plan_id_collision_fails_with_stable_code_before_ledger_append(tmp_path):
    store = AcceptanceStore(tmp_path)
    original = _plan("plan_original")
    pointer = store.ensure_pointer(
        trip_id="trip_demo", run_id="run_demo", original_plan=_record(original)
    )
    conflicting = replace(original, sequence=("poi_b", "poi_a"))

    with pytest.raises(AcceptanceError, match="acceptance_plan_repository_conflict"):
        store.commit(
            trip_id="trip_demo",
            run_id="run_demo",
            session_id="session_demo",
            session_revision_before=0,
            expected_pointer_revision=pointer["revision"],
            expected_plan_id=pointer["accepted_plan_id"],
            expected_plan_content_hash=pointer["accepted_plan_content_hash"],
            decision_type="accept_repair",
            idempotency_key="plan-id-collision-0001",
            selected_plan=_record(conflicting),
            snapshot=None,
            proposal_identity=None,
        )

    assert store.history("trip_demo") == ()
    assert store.load_pointer("trip_demo") == pointer

def test_failure_before_prepare_leaves_pointer_and_ledger_unchanged(tmp_path):
    def fail_at(name: str) -> None:
        if name == "after_plan_save":
            raise OSError("simulated interruption")

    child = replace(_plan("plan_child", parent_plan_id="plan_original"), sequence=("poi_b", "poi_a"))
    with pytest.raises(AcceptanceError, match="acceptance_transaction_failed"):
        _commit(AcceptanceStore(tmp_path, failure_injector=fail_at), selected_plan=child)

    restored = AcceptanceStore(tmp_path)
    assert restored.load_pointer("trip_demo")["accepted_plan_id"] == "plan_original"
    assert restored.history("trip_demo") == ()


def test_two_concurrent_writers_produce_one_decision(tmp_path):
    store = AcceptanceStore(tmp_path)
    original = _plan("plan_original")
    pointer = store.ensure_pointer(
        trip_id="trip_demo", run_id="run_demo", original_plan=_record(original)
    )

    def commit(index: int):
        child = replace(
            _plan(f"plan_child_{index}", parent_plan_id="plan_original"),
            sequence=("poi_b", "poi_a"),
        )
        return store.commit(
            trip_id="trip_demo",
            run_id="run_demo",
            session_id=f"session_{index}",
            session_revision_before=0,
            expected_pointer_revision=pointer["revision"],
            expected_plan_id=pointer["accepted_plan_id"],
            expected_plan_content_hash=pointer["accepted_plan_content_hash"],
            decision_type="accept_repair",
            idempotency_key=f"concurrent-key-000{index}",
            selected_plan=_record(child),
            snapshot=None,
            proposal_identity=None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit, index) for index in (1, 2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result().decision["decision_id"])
        except AcceptanceError as exc:
            outcomes.append(exc.code)

    assert len([row for row in outcomes if row.startswith("decision_")]) == 1
    assert outcomes.count("acceptance_pointer_conflict") == 1
    assert len(store.history("trip_demo")) == 1

@pytest.mark.parametrize("stage", ["after_prepare", "after_ledger_append", "after_pointer_replace"])
def test_restart_recovers_interrupted_transaction(tmp_path, stage):
    def fail_at(name: str) -> None:
        if name == stage:
            raise OSError("simulated interruption")

    store = AcceptanceStore(tmp_path, failure_injector=fail_at)
    child = replace(_plan("plan_child", parent_plan_id="plan_original"), sequence=("poi_b", "poi_a"))
    with pytest.raises(AcceptanceError, match="acceptance_transaction_failed"):
        _commit(store, selected_plan=child)

    restored = AcceptanceStore(tmp_path)
    pointer = restored.load_pointer("trip_demo")
    assert pointer["accepted_plan_id"] == "plan_child"
    assert pointer["revision"] == 1
    assert len(restored.history("trip_demo")) == 1


def test_stale_pointer_conflict_writes_no_second_decision(tmp_path):
    store = AcceptanceStore(tmp_path)
    child = replace(_plan("plan_child", parent_plan_id="plan_original"), sequence=("poi_b", "poi_a"))
    _commit(store, selected_plan=child)

    with pytest.raises(AcceptanceError, match="acceptance_pointer_conflict"):
        store.commit(
            trip_id="trip_demo",
            run_id="run_demo",
            session_id="session_other",
            session_revision_before=0,
            expected_pointer_revision=0,
            expected_plan_id="plan_original",
            expected_plan_content_hash=_plan("plan_original").content_hash,
            decision_type="keep_original",
            idempotency_key="acceptance-key-0002",
            selected_plan=_record(_plan("plan_original")),
            snapshot=None,
            proposal_identity=None,
        )
    assert len(store.history("trip_demo")) == 1


def test_tampered_pointer_and_ledger_fail_closed_without_rewrite(tmp_path):
    store = AcceptanceStore(tmp_path)
    child = replace(_plan("plan_child", parent_plan_id="plan_original"), sequence=("poi_b", "poi_a"))
    _commit(store, selected_plan=child)
    pointer_path = next((tmp_path / "workspaces").glob("*.json"))
    ledger_path = next((tmp_path / "decisions").glob("*/ledger.jsonl"))

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["accepted_plan_id"] = "plan_forged"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    before_pointer = pointer_path.read_bytes()
    with pytest.raises(AcceptanceError, match="acceptance_pointer_invalid"):
        AcceptanceStore(tmp_path).load_pointer("trip_demo")
    assert pointer_path.read_bytes() == before_pointer

    pointer_path.unlink()
    ledger = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(ledger.replace("plan_child", "plan_evil_"), encoding="utf-8")
    before_ledger = ledger_path.read_bytes()
    with pytest.raises(AcceptanceError, match="acceptance_ledger_invalid"):
        AcceptanceStore(tmp_path).history("trip_demo")
    assert ledger_path.read_bytes() == before_ledger
