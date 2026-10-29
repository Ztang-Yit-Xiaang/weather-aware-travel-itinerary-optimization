"""Adversarial observer evidence checks; not a substitute for the browser run."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_same_plan_remount as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def fixture():
    def time(seconds):
        return f"2026-09-08T12:00:{seconds:06.3f}+00:00"

    ack = {
        "active_plan_id": "child",
        "active_plan_content_hash": "child-hash",
        "rendered_map_plan_id": "child",
        "rendered_geometry_hash": "geometry",
        "rendered_marker_ids": ["pfeiffer"],
    }
    rendered = {
        "renderedMapPlanId": "child",
        "renderedGeometryHash": "geometry",
        "renderedMarkerIds": json.dumps(["pfeiffer"]),
    }
    stage = {
        **rendered,
        "activePlanId": "child",
        "activePlanContentHash": "child-hash",
        "previewRequestId": "request",
        "previewState": "PREVIEW_RENDERED",
    }
    data = {
        **rendered,
        "renderedPlanContentHash": "child-hash",
        "mapReady": "true",
        "mapReadyAt": time(0),
        "mapTileCoverage": "1",
        "mapCoverageMethod": "loaded_tile_projected_union_area_v1",
    }
    base = {
        "reason": "armed",
        "detail": None,
        "sequence": 1,
        "observed_at": time(1),
        "performance_ms": 1,
        "path": "/app/itinerary",
        "previous_map_id": 1,
        "previous_connected": True,
        "previous_map": data,
        "current_map_id": 1,
        "current_connected": True,
        "current_map": data,
        "stage": stage,
        "accept": {"present": True, "disabled": False, "visible": True},
    }
    rows = [deepcopy(base) for _ in range(4)]
    for index, row in enumerate(rows, 1):
        row.update(sequence=index, observed_at=time(index), performance_ms=index)
    rows[1].update(reason="trusted_control_event", detail={"trusted": True, "target_id": "map-select-button"})
    for row in rows[2:]:
        row.update(previous_connected=False, previous_map={"mapReady": "false"}, current_map_id=2)
    rows[2].update(
        reason="mutation",
        current_map={"mapReady": "false"},
        stage={
            "activePlanId": "child",
            "activePlanContentHash": "child-hash",
            "previewRequestId": "request",
            "previewState": "PREVIEW_DATA_READY",
        },
        accept={"present": True, "disabled": True, "visible": True},
    )
    rows[3]["reason"] = "stopped"
    rows[3]["current_map"]["mapReadyAt"] = time(3.5)
    return {"label": "select_remount", "rows": rows, "overflow": False}, ack


def test_real_loading_interval_precedes_same_child_readiness():
    observation, ack = fixture()
    assert verifier.verify_observation(observation, "select_remount", "#map-select-button", ack, "request") == 1


@pytest.mark.parametrize(
    "fault",
    [
        "enabled_while_pending",
        "old_ack",
        "new_ack",
        "stage_ack",
        "wrong_state",
        "not_replaced",
        "old_map_connected",
        "synthetic_click",
        "no_loading",
        "wrong_request",
        "wrong_hash",
        "wrong_marker",
        "nan_coverage",
        "low_coverage",
        "stale_ready_time",
        "future_ready_time",
        "overflow",
        "out_of_order",
    ],
)
def test_remount_counterexamples_are_rejected_after_valid_baseline(fault):
    observation, ack = fixture()
    assert verifier.verify_observation(observation, "select_remount", "#map-select-button", ack, "request") == 1
    rows = observation["rows"]
    if fault == "enabled_while_pending":
        rows[2]["accept"]["disabled"] = False
    elif fault == "old_ack":
        rows[2]["previous_map"]["renderedMapPlanId"] = "child"
    elif fault == "new_ack":
        rows[2]["current_map"]["mapReadyAt"] = rows[0]["observed_at"]
    elif fault == "stage_ack":
        rows[2]["stage"]["renderedGeometryHash"] = "geometry"
    elif fault == "wrong_state":
        rows[2]["stage"]["previewState"] = "PREVIEW_RENDERED"
    elif fault == "not_replaced":
        for row in rows:
            row["current_map_id"] = 1
    elif fault == "old_map_connected":
        rows[2]["previous_connected"] = True
    elif fault == "synthetic_click":
        rows[1]["detail"]["trusted"] = False
    elif fault == "no_loading":
        rows[2]["current_map"] = deepcopy(rows[3]["current_map"])
    elif fault == "wrong_request":
        rows[2]["stage"]["previewRequestId"] = "other"
    elif fault == "wrong_hash":
        rows[3]["current_map"]["renderedPlanContentHash"] = "other"
    elif fault == "wrong_marker":
        rows[3]["current_map"]["renderedMarkerIds"] = '["bixby"]'
    elif fault == "nan_coverage":
        rows[3]["current_map"]["mapTileCoverage"] = "nan"
    elif fault == "low_coverage":
        rows[3]["current_map"]["mapTileCoverage"] = "0.94"
    elif fault == "stale_ready_time":
        rows[3]["current_map"]["mapReadyAt"] = rows[0]["observed_at"]
    elif fault == "future_ready_time":
        rows[3]["current_map"]["mapReadyAt"] = "2026-09-08T12:05:00+00:00"
    elif fault == "overflow":
        observation["overflow"] = True
    elif fault == "out_of_order":
        rows[2]["performance_ms"] = 0
    with pytest.raises(core.E2E1VerificationError, match="e2e1_remount_"):
        verifier.verify_observation(observation, "select_remount", "#map-select-button", ack, "request")


def test_missing_browser_evidence_is_not_a_pass(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_remount_evidence_malformed"):
        verifier.verify(tmp_path, "http://127.0.0.1:8127")
