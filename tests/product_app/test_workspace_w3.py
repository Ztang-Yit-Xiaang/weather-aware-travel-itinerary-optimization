from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from itinerary_system.product_app.persistence import SessionSnapshotError
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore


def _route_feedback(revision: int) -> dict[str, object]:
    return {
        "expected_revision": revision,
        "type": "route_feedback",
        "target": "selected_route",
        "parameters": {"preference": "reduce_contextual_risk"},
        "source": "map",
        "evidence_refs": ["route-matrix-v1"],
    }


def test_session_snapshot_never_persists_raw_token_and_restores_authenticated_state(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    first = WorkspaceStore(state_root)
    session, token = first.create_session("run", "parent", 3, trip_id="trip")
    first.select(
        session,
        {
            "expected_revision": 0,
            "selected_day": 4,
            "selected_stop_id": "stop-a",
            "selected_alternative_id": "child-a",
        },
    )
    first.add_operation(session, _route_feedback(1), valid_stop_ids=set(), day_count=7)
    first.set_proposal(
        session,
        {"state": "eligible", "certificate_id": "cert-a"},
        expected_revision=2,
    )
    first.append_permission(
        session,
        {
            "expected_revision": 3,
            "permission": "booking_change",
            "decision": "denied",
            "proposal_id": "proposal-a",
            "scope": "trip",
        },
    )

    snapshot_path = state_root / "sessions" / f"{session.session_id}.json"
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    assert token.encode() not in snapshot_bytes
    assert "mutation_token" not in snapshot["session"]
    assert snapshot["schema_version"] == "product-session-snapshot-v1"
    assert snapshot["token_verifier"]["algorithm"] == "sha256-salted-v1"
    assert all(
        token.encode() not in candidate.read_bytes()
        for candidate in state_root.rglob("*")
        if candidate.is_file()
    )

    restarted = WorkspaceStore(state_root)
    restored = restarted.authenticate(session.session_id, token)
    assert not hasattr(restored, "mutation_token")
    assert restored.revision == 4
    assert restored.selected_day == 4
    assert restored.selected_stop_id == "stop-a"
    assert restored.selected_alternative_id == "child-a"
    assert restored.draft[0].evidence_refs == ("route-matrix-v1",)
    assert restored.proposal == {"state": "eligible", "certificate_id": "cert-a"}
    assert restored.permission_decisions[0]["decision"] == "denied"
    with pytest.raises(WorkspaceError, match="invalid_session_token"):
        restarted.authenticate(session.session_id, "wrong-token")


def test_undo_and_new_draft_atomically_invalidate_persisted_proposal(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    store = WorkspaceStore(state_root)
    session, _ = store.create_session("run", "parent", 1)
    store.add_operation(session, _route_feedback(0), valid_stop_ids=set(), day_count=1)
    store.set_proposal(
        session,
        {"state": "eligible", "certificate_id": "obsolete-cert"},
        expected_revision=1,
    )

    undone = store.undo(session, expected_revision=2)
    assert undone.type == "route_feedback"
    assert session.proposal is None
    assert session.draft == []

    store.add_operation(session, _route_feedback(3), valid_stop_ids=set(), day_count=1)
    restored = WorkspaceStore(state_root).get(session.session_id)
    assert restored.revision == 4
    assert restored.proposal is None
    assert len(restored.draft) == 1


def test_revision_compare_and_swap_reloads_disk_across_store_instances(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    first = WorkspaceStore(state_root)
    session, token = first.create_session("run", "parent", 1)
    second = WorkspaceStore(state_root)
    stale = second.authenticate(session.session_id, token)

    first.select(session, {"expected_revision": 0, "selected_day": 2})
    with pytest.raises(WorkspaceError, match="stale_session_revision") as error:
        second.select(stale, {"expected_revision": 0, "selected_day": 3})
    assert error.value.status_code == 409

    restored = WorkspaceStore(state_root).authenticate(session.session_id, token)
    assert restored.revision == 1
    assert restored.selected_day == 2


def test_failed_atomic_save_does_not_publish_or_corrupt_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "state"
    store = WorkspaceStore(state_root)
    session, _ = store.create_session("run", "parent", 1)
    original_snapshot = (state_root / "sessions" / f"{session.session_id}.json").read_bytes()

    def fail_save(_session: object) -> None:
        raise SessionSnapshotError("session_write_failed")

    monkeypatch.setattr(store._snapshots, "save", fail_save)
    with pytest.raises(WorkspaceError, match="session_write_failed") as error:
        store.select(session, {"expected_revision": 0, "selected_day": 2})
    assert error.value.status_code == 503
    assert session.revision == 0
    assert session.selected_day == 1
    assert (state_root / "sessions" / f"{session.session_id}.json").read_bytes() == original_snapshot


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"schema_version": "future-session-v9"}, "session_snapshot_schema_unsupported"),
        ({"schema_version": "product-session-snapshot-v1"}, "session_snapshot_invalid"),
    ],
)
def test_corrupt_or_unsupported_snapshot_fails_closed_without_rewrite(
    tmp_path: Path, payload: dict[str, str], expected_code: str
) -> None:
    state_root = tmp_path / "state"
    sessions_dir = state_root / "sessions"
    sessions_dir.mkdir(parents=True)
    session_id = "session_" + "a" * 32
    path = sessions_dir / f"{session_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    store = WorkspaceStore(state_root)
    assert store.restoration_errors == {session_id: expected_code}
    with pytest.raises(WorkspaceError, match=expected_code):
        store.get(session_id)
    assert path.read_bytes() == before


def test_expired_snapshot_is_not_restored_or_counted_toward_capacity(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    store = WorkspaceStore(state_root)
    expired, _ = store.create_session("run", "parent", 1)
    path = state_root / "sessions" / f"{expired.session_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["session"]["last_accessed_at"] = (
        datetime.now(UTC) - timedelta(hours=25)
    ).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = WorkspaceStore(state_root)
    assert not path.exists()
    with pytest.raises(WorkspaceError, match="unknown_session"):
        restarted.get(expired.session_id)
    for _ in range(restarted.MAX_SESSIONS):
        restarted.create_session("run", "parent", 1)
    with pytest.raises(WorkspaceError, match="session_capacity_reached"):
        restarted.create_session("run", "parent", 1)


def test_non_finite_nested_values_are_rejected_on_write_and_restore(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    store = WorkspaceStore(state_root)
    session, _ = store.create_session("run", "parent", 1)
    path = state_root / "sessions" / f"{session.session_id}.json"
    before = path.read_bytes()

    with pytest.raises(WorkspaceError, match="invalid_route_feedback") as invalid:
        store.add_operation(
            session,
            {
                "expected_revision": 0,
                "type": "route_feedback",
                "target": "selected_route",
                "parameters": {
                    "preference": "reduce_contextual_risk",
                    "weight": float("nan"),
                },
            },
            valid_stop_ids=set(),
            day_count=1,
        )
    assert invalid.value.status_code == 422
    assert session.revision == 0
    assert path.read_bytes() == before

    corrupt_id = "session_" + "b" * 32
    corrupt_path = state_root / "sessions" / f"{corrupt_id}.json"
    corrupt_path.write_text(
        '{"schema_version":"product-session-snapshot-v1","token_verifier":NaN,"session":{}}',
        encoding="utf-8",
    )
    corrupt_before = corrupt_path.read_bytes()
    restarted = WorkspaceStore(state_root)
    assert restarted.restoration_errors[corrupt_id] == "session_snapshot_invalid"
    assert corrupt_path.read_bytes() == corrupt_before


def test_incompatible_replacement_is_rejected_before_entering_draft(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "state")
    session, _ = store.create_session("run", "parent", 1)

    with pytest.raises(WorkspaceError, match="draft_candidate_target_mismatch") as raised:
        store.add_operation(
            session,
            {
                "expected_revision": 0,
                "type": "replace_nearby",
                "target": "parent-a",
                "parameters": {"candidate_id": "candidate-b"},
                "source": "map",
            },
            valid_stop_ids={"parent-a", "parent-b"},
            day_count=2,
            parent_stop_ids={"parent-a", "parent-b"},
            candidate_ids={"candidate-b"},
            candidate_replacements={"candidate-b": "parent-b"},
        )

    assert raised.value.status_code == 422
    assert session.revision == 0
    assert session.draft == []
