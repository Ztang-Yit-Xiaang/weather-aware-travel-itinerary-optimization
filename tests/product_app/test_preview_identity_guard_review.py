"""Independent malformed-identity selection guard checks; no provider or browser."""

import json

import pytest

from itinerary_system.product_app.workspace import WorkspaceStore


@pytest.mark.parametrize("child_id", [None, "", "   ", 0, False, []])
def test_invalid_child_identity_preserves_selected_parent_context(tmp_path, child_id):
    store = WorkspaceStore(tmp_path / "state")
    session, token = store.create_session("run", "parent", 5)
    store.select(
        session,
        {
            "expected_revision": 0,
            "selected_day": 5,
            "selected_stop_id": "bixby",
            "selected_segment_id": "parent-leg",
            "selected_alternative_id": "parent-option",
            "selected_candidate_id": "pfeiffer",
        },
    )
    fields = (
        "selected_day",
        "selected_stop_id",
        "selected_segment_id",
        "selected_alternative_id",
        "selected_candidate_id",
    )
    before = {field: getattr(session, field) for field in fields}
    proposal = {
        "state": "eligible",
        "child_plan_id": child_id,
        "geography_plan": {"plan_id": child_id, "stops": {"features": []}},
    }
    store.set_proposal(session, proposal, session.revision)
    restored = WorkspaceStore(tmp_path / "state").authenticate(session.session_id, token)
    after = {field: getattr(restored, field) for field in fields}
    (tmp_path / "exact_guard_io.json").write_text(
        json.dumps(
            {
                "proposal": proposal,
                "before": before,
                "after": after,
                "accepted_plan_id": restored.accepted_plan_id,
                "provider_calls": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert after == before and restored.accepted_plan_id == "parent"
    assert restored.proposal == proposal
