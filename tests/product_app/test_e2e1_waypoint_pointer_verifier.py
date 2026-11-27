"""Adversarial unit fixtures for the unregistered real-pointer artifact verifier."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as core  # noqa: E402
import verify_e2e1_waypoint_pointer as waypoint  # noqa: E402


def _overlay_fixture():
    parent = {
        "active_plan_id": "parent",
        "active_plan_content_hash": "hash",
        "rendered_geometry_hash": "geo",
        "rendered_marker_ids": ["accepted_stop"],
    }
    legs = [
        {
            "route_leg_id": "preview_leg_" + str(index) * 16,
            "query_hash": str(index) * 64,
            "geometry": {"type": "LineString", "coordinates": [[-121, 36], [-122, 37]]},
        }
        for index in (1, 2)
    ]
    operation = {
        "parameters": {
            "name": "Unit fixture waypoint",
            "affected_route_legs": legs,
            "snapped_coordinate": {"longitude": -122, "latitude": 37},
        }
    }
    observed = {
        "scope": "read_only_real_map_observation_not_production_overlay_ack",
        "plan_id": "parent",
        "plan_content_hash": "hash",
        "geometry_hash": "geo",
        "accepted_marker_ids": ["accepted_stop"],
        "draft_route_source_loaded": True,
        "draft_routes": {
            "features": [
                {
                    "geometry": leg["geometry"],
                    "properties": {
                        "route_leg_id": leg["route_leg_id"],
                        "road_validated": True,
                        "display_role": "route_checked_draft",
                    },
                }
                for leg in legs
            ]
        },
        "rendered_draft_route_leg_ids": [leg["route_leg_id"] for leg in legs],
        "draft_markers": [
            {
                "visible": True,
                "label": "Unit fixture waypoint",
                "expected_coordinate": {"longitude": -122, "latitude": 37},
                "distance_from_expected_pixels": 0.3,
            }
        ],
    }
    return parent, operation, observed


def test_route_checked_overlay_is_not_an_evaluated_child_acknowledgement():
    parent, operation, observed = _overlay_fixture()
    waypoint._overlay(core, observed, operation, parent)


@pytest.mark.parametrize(
    "fault",
    [
        "plan",
        "hash",
        "geometry",
        "no_visible_marker",
        "no_drawn_leg",
        "invented_leg",
        "wrong_display_role",
        "wrong_road_geometry",
        "duplicate_leg",
        "duplicate_feature",
        "only_one_drawn_leg",
        "wrong_expected_coordinate",
        "marker_wrong_position",
        "marker_nan",
        "marker_negative_distance",
    ],
)
def test_plausible_map_overlay_cannot_pass_with_wrong_artifacts(fault):
    parent, operation, observed = _overlay_fixture()
    if fault in {"plan", "hash", "geometry"}:
        observed[{"plan": "plan_id", "hash": "plan_content_hash", "geometry": "geometry_hash"}[fault]] = "other"
    elif fault == "no_visible_marker":
        observed["draft_markers"][0]["visible"] = False
    elif fault == "no_drawn_leg":
        observed["rendered_draft_route_leg_ids"] = []
    elif fault == "invented_leg":
        observed["rendered_draft_route_leg_ids"] = ["not_in_route_evidence"]
    elif fault == "wrong_display_role":
        observed["draft_routes"]["features"][0]["properties"]["display_role"] = "evaluated"
    elif fault == "wrong_road_geometry":
        observed["draft_routes"]["features"][0]["geometry"] = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    elif fault == "duplicate_leg":
        operation["parameters"]["affected_route_legs"][1] = deepcopy(operation["parameters"]["affected_route_legs"][0])
    elif fault == "duplicate_feature":
        observed["draft_routes"]["features"][1] = deepcopy(observed["draft_routes"]["features"][0])
    elif fault == "only_one_drawn_leg":
        observed["rendered_draft_route_leg_ids"].pop()
    elif fault == "wrong_expected_coordinate":
        observed["draft_markers"][0]["expected_coordinate"] = {"longitude": 0, "latitude": 0}
    else:
        observed["draft_markers"][0]["distance_from_expected_pixels"] = {
            "marker_wrong_position": 2,
            "marker_nan": float("nan"),
            "marker_negative_distance": -0.1,
        }[fault]
    with pytest.raises(core.E2E1VerificationError):
        waypoint._overlay(core, observed, operation, parent)


def test_undo_requires_actual_overlay_and_marker_removal():
    parent, _, observed = _overlay_fixture()
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_overlay_not_removed"):
        waypoint._overlay(core, observed, None, parent)
    observed["draft_routes"]["features"] = []
    observed["draft_markers"] = []
    observed["rendered_draft_route_leg_ids"] = []
    waypoint._overlay(core, observed, None, parent)


@pytest.mark.parametrize("index", [{}, [None], [{"file": "http_001.json"}]])
def test_malformed_http_index_has_stable_rejection(tmp_path, index):
    (tmp_path / "http_index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_http_index_invalid"):
        waypoint._http(core, tmp_path, "http://127.0.0.1:8127")


def test_missing_evidence_is_a_stable_rejection_not_raw_file_exception(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_evidence_malformed"):
        waypoint.verify(tmp_path, "http://127.0.0.1:8127")


def _pointer_fixture():
    _, added, _ = _overlay_fixture()
    added["target"] = "waypoint_unit"
    events = [
        {"type": kind, "trusted": True, "x": x, "y": x, "buttons": buttons}
        for kind, x, buttons in (("mousedown", 10, 1), ("mousemove", 20, 1), ("mouseup", 30, 0))
    ]
    inputs = [
        {"action": "real_canvas_click"},
        {
            "action": "real_marker_pointer_drag",
            "waypoint_id": "waypoint_unit",
            "target_coordinate_from_actual_road_geometry": [-122, 37],
            "trusted_dom_mouse_events": events,
            "from_pixels": {"x": 10, "y": 10},
            "to_pixels": {"x": 30, "y": 30},
        },
    ]
    request = {"longitude": -122, "latitude": 37}
    return inputs, added, request, {"raw_coordinate": deepcopy(request)}


def test_trusted_pointer_correlates_actual_road_coordinate_and_request():
    waypoint._pointer(core, *_pointer_fixture())


@pytest.mark.parametrize(
    "fault", ["synthetic", "no_held_button", "wrong_endpoint", "short_drag", "wrong_coordinate", "wrong_request"]
)
def test_pointer_claims_cannot_replace_trusted_gesture_evidence(fault):
    inputs, added, request, snap = _pointer_fixture()
    events = inputs[1]["trusted_dom_mouse_events"]
    if fault == "synthetic":
        events[1]["trusted"] = False
    elif fault == "no_held_button":
        events[1]["buttons"] = 0
    elif fault == "wrong_endpoint":
        inputs[1]["to_pixels"]["x"] = 31.1
    elif fault == "short_drag":
        inputs[1]["to_pixels"] = {"x": 11, "y": 11}
        events[-1].update(x=11, y=11)
    elif fault == "wrong_coordinate":
        request["longitude"] = snap["raw_coordinate"]["longitude"] = -120
    else:
        request["longitude"] = -120
    with pytest.raises(core.E2E1VerificationError):
        waypoint._pointer(core, inputs, added, request, snap)


def _focus_fixture():
    def stamp(second):
        return f"2026-09-08T12:00:{second:02d}Z"

    def rect(left, top, right, bottom):
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
        }

    def point(x, y, canvas):
        return {
            "x": x,
            "y": y,
            "hit_is_canvas": canvas,
            "hit_is_context": not canvas,
            "hit_tag": "CANVAS" if canvas else "SECTION",
            "hit_id": None,
            "hit_html": "<canvas></canvas>" if canvas else '<section class="map-context-card"></section>',
        }

    def frame(second, opened):
        canvas, panel = rect(100, 100, 900, 730), rect(650, 200, 850, 500)
        starts = [point(120 + index * 10, 650, True) for index in range(36)]
        ends = [point(680 + index * 10, 300, False) for index in range(9)]
        return {
            "schema_version": "pointer-visible-frame-v1",
            "observed_at": stamp(second),
            "viewport": {"left": 0, "top": 0, "right": 1440, "bottom": 900},
            "scroll": {"x": 0, "y": 0},
            "container": canvas,
            "canvas": deepcopy(canvas),
            "panel": panel,
            "visible_canvas": deepcopy(canvas),
            "visible_panel": deepcopy(panel),
            "native_text_route_open": opened,
            "legacy_full_container_samples": [point(200 + index * 10, 400, True) for index in range(16)],
            "visible_canvas_samples": starts,
            "visible_panel_samples": ends,
            "start": {key: starts[0][key] for key in ("x", "y")},
            "end": {key: ends[0][key] for key in ("x", "y")},
            "endpoint_html": ends[0]["hit_html"],
            "endpoint_is_context": True,
        }

    parent = {
        "active_plan_id": "parent",
        "rendered_map_plan_id": "parent",
        "active_plan_content_hash": "hash",
        "rendered_geometry_hash": "geometry",
        "rendered_marker_ids": ["stop"],
        "preview_state": "PARENT_RENDERED",
    }
    after_pan = frame(1, True)
    overlay = {
        "start": deepcopy(after_pan["start"]),
        "end": deepcopy(after_pan["end"]),
        "framing": {
            "action": "locator_scroll_into_view_if_needed",
            "selector": "#geographic-map",
            "map_camera_modified": False,
            "text_route_toggled": False,
            "before": frame(0, True),
            "after": after_pan,
        },
    }
    receipt = {
        "schema_version": "native-canvas-focus-transition-v1",
        "action": "native_text_route_summary_click",
        "selector": ".textual-map-alternative summary",
        "before": frame(4, True),
        "after": frame(6, False),
        "map_ack_before": deepcopy(parent),
        "map_ack_after": deepcopy(parent),
        "click": {"trusted": True, "target_is_summary": True, "at": stamp(5)},
    }
    raw = {
        "action": "real_canvas_click",
        "pixels": {"x": 200, "y": 400},
        "native_click": {"trusted": True, "target_is_canvas": True, "x": 200, "y": 400, "at": stamp(7)},
    }
    ready = {
        "screenshot_started_at": stamp(2),
        "screenshot_completed_at": stamp(3),
        "viewport": {"width": 1440, "height": 900},
        "container": {"width": 800, "height": 630},
    }
    return receipt, overlay, raw, parent, ready


def _focus_io(tmp_path, case, outcome):
    (tmp_path / "exact_focus_verifier_io.json").write_text(
        json.dumps(
            {
                "scope": "offline_independent_pointer_receipt_verification",
                "browser_execution_claimed": False,
                "provider_calls": 0,
                "input": case,
                "output": outcome,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_native_panel_transition_binds_bounds_identity_and_actual_click_order(tmp_path):
    case = _focus_fixture()
    waypoint._canvas_focus_transition(core, *case)
    _focus_io(tmp_path, case, {"accepted": True})


@pytest.mark.parametrize(
    "fault,code",
    [
        ("untrusted", "native_focus_transition_invalid"),
        ("wrong_summary", "native_focus_transition_invalid"),
        ("not_open_before", "native_focus_transition_invalid"),
        ("not_closed", "native_focus_transition_invalid"),
        ("foreign_plan", "accepted_plan_changed"),
        ("foreign_hash", "accepted_plan_changed"),
        ("foreign_geometry", "accepted_plan_changed"),
        ("foreign_markers", "accepted_plan_changed"),
        ("before_release_screenshot", "focus_transition_order_invalid"),
        ("after_raw_click", "focus_transition_order_invalid"),
        ("fake_followup", "focus_followup_click_invalid"),
        ("wrong_click_pixel", "focus_followup_click_invalid"),
        ("camera_setter_claim", "overlay_framing_unbound"),
        ("premature_close", "overlay_framing_unbound"),
        ("different_pan_start", "overlay_framing_unbound"),
        ("negative_rect", "visible_frame_invalid"),
        ("nan_bounds", "visible_frame_invalid"),
        ("false_intersection", "visible_frame_invalid"),
        ("offscreen_canvas_hit", "visible_frame_invalid"),
        ("wrong_hit_tag", "visible_frame_invalid"),
        ("missing_hit_html", "visible_frame_invalid"),
        ("duplicate_samples", "visible_frame_invalid"),
        ("unobserved_start", "visible_frame_invalid"),
        ("empty_inventory", "visible_frame_invalid"),
        ("wrong_viewport", "focus_readiness_bounds_mismatch"),
        ("still_obstructed", "canvas_still_obstructed"),
    ],
)
def test_forged_or_incomplete_native_focus_receipts_fail_closed(tmp_path, fault, code):
    case = _focus_fixture()
    receipt, overlay, raw, _, ready = case
    if fault == "untrusted":
        receipt["click"]["trusted"] = False
    elif fault == "wrong_summary":
        receipt["click"]["target_is_summary"] = False
    elif fault == "not_open_before":
        receipt["before"]["native_text_route_open"] = False
    elif fault == "not_closed":
        receipt["after"]["native_text_route_open"] = True
    elif fault.startswith("foreign_"):
        key = {
            "foreign_plan": "active_plan_id",
            "foreign_hash": "active_plan_content_hash",
            "foreign_geometry": "rendered_geometry_hash",
            "foreign_markers": "rendered_marker_ids",
        }[fault]
        receipt["map_ack_after"][key] = ["foreign"] if fault == "foreign_markers" else "foreign"
    elif fault == "before_release_screenshot":
        receipt["before"]["observed_at"] = ready["screenshot_started_at"]
    elif fault == "after_raw_click":
        receipt["after"]["observed_at"] = "2026-09-08T12:00:08Z"
    elif fault == "fake_followup":
        raw["native_click"]["trusted"] = False
    elif fault == "wrong_click_pixel":
        raw["native_click"]["x"] += 3
    elif fault == "camera_setter_claim":
        overlay["framing"]["map_camera_modified"] = True
    elif fault == "premature_close":
        overlay["framing"]["after"]["native_text_route_open"] = False
    elif fault == "different_pan_start":
        overlay["start"]["x"] += 3
    elif fault == "negative_rect":
        receipt["after"]["canvas"]["width"] = -1
    elif fault == "nan_bounds":
        receipt["after"]["viewport"]["right"] = float("nan")
    elif fault == "false_intersection":
        receipt["after"]["visible_canvas"]["left"] += 1
        receipt["after"]["visible_canvas"]["width"] -= 1
    elif fault == "offscreen_canvas_hit":
        receipt["after"]["visible_canvas_samples"][0]["x"] = -1
    elif fault == "wrong_hit_tag":
        receipt["after"]["visible_canvas_samples"][0]["hit_tag"] = "P"
    elif fault == "missing_hit_html":
        receipt["after"]["visible_canvas_samples"][0]["hit_html"] = ""
    elif fault == "duplicate_samples":
        receipt["after"]["visible_canvas_samples"][1] = deepcopy(receipt["after"]["visible_canvas_samples"][0])
    elif fault == "unobserved_start":
        receipt["after"]["start"]["y"] -= 1
    elif fault == "empty_inventory":
        receipt["after"]["visible_panel_samples"] = []
    elif fault == "wrong_viewport":
        ready["viewport"]["width"] += 1
    elif fault == "still_obstructed":
        for point in receipt["after"]["legacy_full_container_samples"]:
            point.update(hit_is_canvas=False, hit_is_context=False, hit_tag="P", hit_html="<p>Text route</p>")
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_" + code) as caught:
        waypoint._canvas_focus_transition(core, *case)
    _focus_io(tmp_path, case, {"rejected": str(caught.value)})
