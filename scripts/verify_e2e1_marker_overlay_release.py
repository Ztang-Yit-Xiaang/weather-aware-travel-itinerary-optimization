"""Unregistered checks for actual draft-marker release over a context panel.

The caller must also verify canonical snap requests, draft-route sources, and
the generic trusted marker gesture; this module binds the physical release.
"""
from __future__ import annotations

import math

import verify_e2e1_vertical_slice as core


def require(condition, reason):
    core._require(condition, "e2e1_marker_overlay_" + reason)


def point(value):
    return isinstance(value, dict) and all(type(value.get(key)) in (int, float)
        and math.isfinite(value[key]) for key in ("x", "y"))


def _verify(assertions, drag):
    require(assertions["real_marker_overlay_release"] is True
        and drag["action"] == "real_marker_pointer_drag", "claim_or_action_invalid")
    aiming = drag["real_pointer_aiming"]
    require(isinstance(aiming, list) and 1 <= len(aiming) <= 14
        and all(isinstance(row, dict) and point(row["from"]) and point(row["to"])
            and row.get("unavailable") is not True for row in aiming), "aiming_missing_or_unavailable")
    require(aiming[-1].get("ready") is True and aiming[-1]["to"] == drag["to_pixels"]
        and point(drag["from_pixels"])
        and all(abs(aiming[-1]["from"][key] - drag["from_pixels"][key]) <= 1.5 for key in ("x", "y")),
        "aimed_endpoint_not_actual_drag")
    for row in aiming[:-1]:
        require(row.get("ready") is not True, "aiming_continued_after_ready")
        if row.get("zoom") is True:
            require(row["start_visible"] is True and "pan" not in row, "zoom_aiming_hides_marker")
        else:
            require(point(row["pan"]["start"]) and point(row["pan"]["end"]), "invalid_real_pan_aim")
    events = drag["native_capture_release_events"]
    require(isinstance(events, list) and events and all(isinstance(event, dict)
        and event["phase"] in {"capture", "bubble"}
        and event["type"] in {"gotpointercapture", "mouseup", "lostpointercapture"}
        and event["trusted"] is True and (event["type"] == "lostpointercapture"
            or event["target_is_original_marker"] is True) for event in events),
        "untrusted_capture_or_foreign_marker")
    before = drag["release_hit_before_mouseup"]
    require(before["hit_is_context"] is True and type(before["hit_panel_id"]) is int
        and before["hit_panel_id"] > 0 and before["current_panel_id"] == before["hit_panel_id"]
        and point(before) and all(abs(before[key] - drag["to_pixels"][key]) <= 1 for key in ("x", "y")),
        "physical_release_not_over_original_panel")
    releases = {}
    for phase in ("capture", "bubble"):
        selected = [event for event in events if event["phase"] == phase]
        names = [event["type"] for event in selected]
        require(names.count("gotpointercapture") == names.count("mouseup") == 1
            and names.index("gotpointercapture") < names.index("mouseup"), "missing_native_capture_or_release")
        release = selected[names.index("mouseup")]
        require(type(release["event_id"]) is int and release["event_id"] > 0
            and type(release["event_time_stamp"]) in (int, float)
            and math.isfinite(release["event_time_stamp"]) and release["event_time_stamp"] > 0
            and point(release)
            and all(abs(release[key] - drag["to_pixels"][key]) <= 1 for key in ("x", "y")), "release_not_over_panel")
        releases[phase] = release
    capture, bubble = releases["capture"], releases["bubble"]
    require(capture["event_id"] == bubble["event_id"]
        and capture["event_time_stamp"] == bubble["event_time_stamp"]
        and events.index(capture) < events.index(bubble), "release_is_not_same_native_event")
    require(capture["hit_is_context"] is True
        and capture["hit_panel_id"] == capture["captured_panel_id"] == before["hit_panel_id"]
        and capture["current_panel_id"] == before["hit_panel_id"]
        and capture["captured_panel_connected"] is True, "capture_not_over_original_connected_panel")
    # The drag-end handler synchronously remounts the context panel before this
    # same native mouseup reaches document's bubble listener. Bind that exact
    # transition; requiring a second physical hit on the removed panel is false.
    require(bubble["captured_panel_id"] == before["hit_panel_id"]
        and bubble["captured_panel_connected"] is False
        and type(bubble["current_panel_id"]) is int and bubble["current_panel_id"] > 0
        and bubble["current_panel_id"] != before["hit_panel_id"]
        and bubble["current_context_title"] == "Unconfirmed relocation", "panel_remount_not_bound_to_release")
    return {"verified": True, "real_marker_overlay_release": True, "evaluated_repair_claimed": False}


def verify(assertions, drag):
    try:
        return _verify(assertions, drag)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_marker_overlay_evidence_malformed") from error
