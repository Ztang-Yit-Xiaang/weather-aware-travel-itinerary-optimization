"""Unregistered native marker-release counterexamples; never browser evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_marker_overlay_release as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def fixture():
    start, end = {"x": 20, "y": 40}, {"x": 170, "y": 70}
    events = [{"phase": phase, "type": kind, "trusted": True, "target_is_original_marker": True,
        "hit_is_context": kind == "mouseup" and phase == "capture", "event_id": number + 1,
        "event_time_stamp": 100.5 + number, "hit_panel_id": 1 if phase == "capture" else None,
        "captured_panel_id": 1, "current_panel_id": 1 if phase == "capture" else 2,
        "captured_panel_connected": phase == "capture", "current_context_title": "Unconfirmed relocation",
        **(end if kind == "mouseup" else start)}
        for number, kind in enumerate(("gotpointercapture", "lostpointercapture", "mouseup"))
        for phase in ("capture", "bubble")]
    return {"real_marker_overlay_release": True}, {"action": "real_marker_pointer_drag", "from_pixels": start, "to_pixels": end,
        "real_pointer_aiming": [{"ready": True, "from": start, "to": end}], "native_capture_release_events": events,
        "release_hit_before_mouseup": {**end, "hit_is_context": True, "hit_panel_id": 1, "current_panel_id": 1}}


def test_native_marker_release_bound_to_actual_aimed_panel_endpoint():
    assert verifier.verify(*fixture())["real_marker_overlay_release"] is True


@pytest.mark.parametrize("fault", ["missing_claim", "missing_aim", "failed_aim", "wrong_aim", "foreign_origin",
    "missing_capture", "missing_bubble", "synthetic", "foreign_marker", "uncovered_release", "wrong_release", "nan_release",
    "reversed_capture", "duplicate_release"])
def test_forged_marker_release_or_unbound_aiming_is_rejected(fault):
    assertions, drag = fixture()
    events = drag["native_capture_release_events"]
    if fault == "missing_claim":
        assertions["real_marker_overlay_release"] = False
    elif fault == "missing_aim":
        drag["real_pointer_aiming"] = []
    elif fault == "failed_aim":
        drag["real_pointer_aiming"][-1]["unavailable"] = True
    elif fault == "wrong_aim":
        drag["to_pixels"] = {"x": 180, "y": 70}
    elif fault == "foreign_origin":
        drag["from_pixels"] = {"x": 900, "y": 900}
    elif fault == "missing_capture":
        drag["native_capture_release_events"] = [event for event in events if event["type"] != "gotpointercapture"]
    elif fault == "missing_bubble":
        drag["native_capture_release_events"] = [event for event in events if event["phase"] != "bubble"]
    elif fault == "synthetic":
        events[-1]["trusted"] = False
    elif fault == "foreign_marker":
        events[-1]["target_is_original_marker"] = False
    elif fault == "uncovered_release":
        events[-2]["hit_is_context"] = False
    elif fault == "wrong_release":
        events[-1]["x"] += 10
    elif fault == "nan_release":
        events[-1]["x"] = float("nan")
    elif fault == "reversed_capture":
        events.reverse()
    else:
        events.append(dict(events[-1]))
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify(assertions, drag)


def test_lost_capture_may_retarget_document_after_marker_detaches():
    assertions, drag = fixture()
    for event in drag["native_capture_release_events"]:
        if event["type"] == "lostpointercapture":
            event["target_is_original_marker"] = False
    assert verifier.verify(assertions, drag)["verified"] is True


def test_missing_marker_evidence_has_stable_failure():
    with pytest.raises(core.E2E1VerificationError, match="e2e1_marker_overlay_evidence_malformed"):
        verifier.verify({}, {})


@pytest.mark.parametrize("fault", ["physical_hit_missing", "physical_panel_changed", "physical_point_changed",
    "event_id_changed", "event_timestamp_changed", "nan_timestamp", "boolean_event_id", "capture_foreign_panel",
    "capture_disconnected", "bubble_old_panel_connected", "bubble_foreign_old_panel", "bubble_same_panel",
    "bubble_wrong_title", "bubble_before_capture"])
def test_native_release_and_synchronous_panel_remount_cannot_be_forged(fault):
    assertions, drag = fixture()
    before = drag["release_hit_before_mouseup"]
    capture, bubble = drag["native_capture_release_events"][-2:]
    if fault == "physical_hit_missing":
        before["hit_is_context"] = False
    elif fault == "physical_panel_changed":
        before["current_panel_id"] = 2
    elif fault == "physical_point_changed":
        before["x"] += 5
    elif fault == "event_id_changed":
        bubble["event_id"] += 1
    elif fault == "event_timestamp_changed":
        bubble["event_time_stamp"] += 1
    elif fault == "nan_timestamp":
        bubble["event_time_stamp"] = float("nan")
    elif fault == "boolean_event_id":
        capture["event_id"] = bubble["event_id"] = True
    elif fault == "capture_foreign_panel":
        capture["hit_panel_id"] = 3
    elif fault == "capture_disconnected":
        capture["captured_panel_connected"] = False
    elif fault == "bubble_old_panel_connected":
        bubble["captured_panel_connected"] = True
    elif fault == "bubble_foreign_old_panel":
        bubble["captured_panel_id"] = 3
    elif fault == "bubble_same_panel":
        bubble["current_panel_id"] = 1
    elif fault == "bubble_wrong_title":
        bubble["current_context_title"] = "Unrelated result"
    else:
        drag["native_capture_release_events"][-2:] = [bubble, capture]
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify(assertions, drag)
