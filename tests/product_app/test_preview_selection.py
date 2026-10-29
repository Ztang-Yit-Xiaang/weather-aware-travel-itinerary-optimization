from pathlib import Path

import pytest

from itinerary_system.product_app.workspace import WorkspaceStore


@pytest.mark.parametrize("eligible", [True, False])
def test_preview_selection_is_persisted_with_eligible_replacement(tmp_path: Path, eligible: bool) -> None:
    root = tmp_path / "selection-state"
    store = WorkspaceStore(root)
    session, token = store.create_session("run", "parent", 5)
    store.select(session, {
        "expected_revision": session.revision, "selected_day": 5,
        "selected_stop_id": "bixby", "selected_segment_id": "old-leg",
        "selected_alternative_id": "old-option", "selected_candidate_id": "pfeiffer",
    })
    store.add_operation(session, {
        "expected_revision": session.revision, "type": "replace_nearby",
        "target": "bixby", "parameters": {"candidate_id": "pfeiffer"}, "source": "copilot",
    }, valid_stop_ids={"bixby", "pfeiffer"}, parent_stop_ids={"bixby"},
        candidate_ids={"pfeiffer"}, candidate_replacements={"pfeiffer": "bixby"}, day_count=7)
    proposal = {
        "state": "eligible" if eligible else "ineligible",
        "child_plan_id": "child",
        "geography_plan": {
            "plan_id": "child",
            "stops": {"features": [{"properties": {"stop_id": "pfeiffer", "day": 5}}]},
        },
    }
    store.set_proposal(session, proposal, session.revision)
    restored = WorkspaceStore(root).authenticate(session.session_id, token)
    assert restored.accepted_plan_id == "parent"
    assert restored.proposal == proposal
    assert restored.selected_day == 5
    assert restored.selected_stop_id == ("pfeiffer" if eligible else "bixby")
    assert restored.selected_segment_id == (None if eligible else "old-leg")
    assert restored.selected_alternative_id == (None if eligible else "old-option")
    assert len(restored.draft) == 1


def test_preview_updates_selected_day_without_selecting_a_different_stop(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "state")
    session, _ = store.create_session("run", "parent", 2)
    store.select(session, {"expected_revision": 0, "selected_day": 2, "selected_stop_id": "stanford"})
    store.set_proposal(session, {
        "state": "eligible", "child_plan_id": "child",
        "geography_plan": {"plan_id": "child", "stops": {
            "features": [{"properties": {"stop_id": "stanford", "day": 3}}],
        }},
    }, session.revision)
    assert session.selected_stop_id == "stanford"
    assert session.selected_day == 3


def test_preview_clears_removed_selection_when_no_replacement_is_defined(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "state")
    session, _ = store.create_session("run", "parent", 5)
    store.select(session, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": "bixby"})
    store.set_proposal(session, {
        "state": "eligible", "child_plan_id": "child",
        "geography_plan": {"plan_id": "child", "stops": {"features": []}},
    }, session.revision)
    assert session.selected_stop_id is None
    assert session.selected_day == 5

