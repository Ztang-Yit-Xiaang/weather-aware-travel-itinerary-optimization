"""Independent fail-closed remount counterexamples, not browser evidence."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_same_plan_remount as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from test_e2e1_same_plan_remount_verifier import fixture  # noqa: E402


@pytest.mark.parametrize("absent", [True, False])
def test_loading_before_replacement_cannot_leave_acceptance_enabled(absent):
    observation, ack = fixture()
    rows = observation["rows"]
    gap = deepcopy(rows[1])
    gap.update(reason="mutation", detail=None, observed_at="2026-09-08T12:00:02.500+00:00", performance_ms=2.5)
    if absent:
        gap.update(current_map_id=None, current_map=None, current_connected=False,
            previous_connected=False, previous_map={"mapReady": "false"})
    else:
        gap["current_map"]["mapReady"] = "false"
    # The previously rendered acknowledgement and enabled acceptance are stale
    # in this interval, even if a later new-map interval is correctly disabled.
    rows.insert(2, gap)
    for index, row in enumerate(rows, 1):
        row["sequence"] = index
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify_observation(observation, "select_remount", "#map-select-button", ack, "request")


def test_armed_observation_must_reference_the_same_child_as_the_preview():
    observation, ack = fixture()
    observation["rows"][0]["current_map"]["renderedMapPlanId"] = "wrong_plan"
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify_observation(observation, "select_remount", "#map-select-button", ack, "request")
