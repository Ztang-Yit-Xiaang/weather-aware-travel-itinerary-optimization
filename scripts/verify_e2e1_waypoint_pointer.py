"""Independent map-pointer artifact verification; registration follows source freeze."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

SCENARIO = "waypoint_pointer"
TEST_NAME = "test_map_pointer_add_relocate_and_two_undos_preserve_accepted_plan"
LABELS = ("initial_parent", "canvas_overlay_release", "added_draft", "relocated_draft", "undo_relocation", "undo_add", "parent_after_reload")


def _http(core, base, base_url):
    index = json.loads((base / "http_index.json").read_text(encoding="utf-8"))
    core._require(isinstance(index, list) and all(isinstance(row, dict) and
        {"file", "url", "method", "status"} <= row.keys() for row in index), "e2e1_waypoint_http_index_invalid")
    names = [row["file"] for row in index]
    core._require(all(isinstance(name, str) for name in names) and len(names) == len(set(names))
        and set(names) == {path.name for path in base.glob("http_*.json") if path.name != "http_index.json"},
        "e2e1_waypoint_http_inventory_invalid")
    records = []
    for item in index:
        path = core._safe_path(base, item["file"])
        row = core._load(path)
        core._verify_product_http_capture(row)
        core._require(row.get("schema_version") == "e2e1-product-http-io-v1"
            and row.get("url") == item["url"] and row.get("method") == item["method"]
            and row.get("response_status_code") == item["status"] == 200
            and row["url"].startswith(base_url + "/api/"), "e2e1_waypoint_http_invalid")
        records.append((row, json.loads(row["request_body_utf8"]) if row["request_body_utf8"] else {},
                        json.loads(row["response_body_utf8"])))
    return records


def _same_parent(core, ack, parent):
    fields = ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id",
              "rendered_geometry_hash", "preview_state", "rendered_marker_ids")
    core._require(all(ack.get(field) == parent.get(field) for field in fields)
        and ack["active_plan_id"] == ack["rendered_map_plan_id"]
        and ack["preview_state"] == "PARENT_RENDERED", "e2e1_waypoint_accepted_plan_changed")


def _overlay(core, observed, operation, parent):
    core._require(observed["scope"] == "read_only_real_map_observation_not_production_overlay_ack"
        and observed["plan_id"] == parent["active_plan_id"]
        and observed["plan_content_hash"] == parent["active_plan_content_hash"]
        and observed["geometry_hash"] == parent["rendered_geometry_hash"]
        and set(observed["accepted_marker_ids"]) == set(parent["rendered_marker_ids"]),
        "e2e1_waypoint_overlay_parent_mismatch")
    features = observed["draft_routes"]["features"]
    if operation is None:
        core._require(features == observed["draft_markers"] == observed["rendered_draft_route_leg_ids"] == [],
            "e2e1_waypoint_overlay_not_removed")
        return
    legs = operation["parameters"]["affected_route_legs"]
    core._require(observed["draft_route_source_loaded"] is True and len(features) == len(legs) == 2
        and len({leg["route_leg_id"] for leg in legs}) == 2
        and {feature["properties"]["route_leg_id"] for feature in features}
        == {leg["route_leg_id"] for leg in legs}, "e2e1_waypoint_route_source_invalid")
    for feature in features:
        matches = [leg for leg in legs if leg["route_leg_id"] == feature["properties"]["route_leg_id"]]
        core._require(len(matches) == 1, "e2e1_waypoint_overlay_leg_mismatch")
        leg = matches[0]
        core._require(feature["geometry"] == leg["geometry"]
            and feature["properties"]["road_validated"] is True
            and feature["properties"]["display_role"] == "route_checked_draft"
            and leg["route_leg_id"] == "preview_leg_" + leg["query_hash"][:16],
            "e2e1_waypoint_overlay_geometry_mismatch")
    markers = observed["draft_markers"]
    core._require(len(observed["rendered_draft_route_leg_ids"]) == 2
        and set(observed["rendered_draft_route_leg_ids"]) == {leg["route_leg_id"] for leg in legs}
        and len(markers) == 1 and markers[0]["visible"] is True
        and operation["parameters"]["name"] in markers[0]["label"]
        and markers[0]["expected_coordinate"] == operation["parameters"]["snapped_coordinate"]
        and type(markers[0]["distance_from_expected_pixels"]) in (int, float)
        and math.isfinite(markers[0]["distance_from_expected_pixels"])
        and 0 <= markers[0]["distance_from_expected_pixels"] <= 1.5, "e2e1_waypoint_not_visibly_rendered")


def _pointer(core, inputs, added, snap_request, snap):
    core._require(isinstance(inputs, list) and len(inputs) == 2
        and [row["action"] for row in inputs] == ["real_canvas_click", "real_marker_pointer_drag"]
        and inputs[1]["waypoint_id"] == added["target"], "e2e1_waypoint_pointer_inputs_invalid")
    drag = inputs[1]
    target = drag["target_coordinate_from_actual_road_geometry"]
    coordinates = [point for leg in added["parameters"]["affected_route_legs"]
                   for point in leg["geometry"]["coordinates"]]
    core._require(target in coordinates, "e2e1_waypoint_invented_drag_target")
    gesture = drag["trusted_dom_mouse_events"]
    core._require(isinstance(gesture, list) and len(gesture) >= 3
        and gesture[0]["type"] == "mousedown" and gesture[-1]["type"] == "mouseup"
        and all(row["trusted"] is True and row["type"] in {"mousedown", "mousemove", "mouseup"}
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in ("x", "y")) for row in gesture)
        and gesture[0]["buttons"] == 1 and gesture[-1]["buttons"] == 0
        and all(row["type"] == "mousemove" and row["buttons"] == 1 for row in gesture[1:-1]),
        "e2e1_waypoint_untrusted_pointer_gesture")
    for event, point in ((gesture[0], drag["from_pixels"]), (gesture[-1], drag["to_pixels"])):
        core._require(all(type(point[key]) in (int, float) and math.isfinite(point[key])
            and abs(event[key] - point[key]) <= 1 for key in ("x", "y")), "e2e1_waypoint_pointer_endpoint_mismatch")
    core._require(math.hypot(gesture[-1]["x"] - gesture[0]["x"], gesture[-1]["y"] - gesture[0]["y"]) > 8,
        "e2e1_waypoint_pointer_displacement_insufficient")
    raw = snap["raw_coordinate"]
    core._require(raw == {key: snap_request[key] for key in ("longitude", "latitude")}
        and all(type(raw[key]) in (int, float) and math.isfinite(raw[key])
            and abs(raw[key] - target[index]) <= 0.00002 for index, key in enumerate(("longitude", "latitude"))),
        "e2e1_waypoint_pointer_coordinate_mismatch")


def _visible_frame(core, frame):
    """Recompute viewport intersections; do not trust claimed visible hit flags."""
    def require(value):
        core._require(value, "e2e1_waypoint_visible_frame_invalid")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def within(point, rect):
        return rect["left"] <= point["x"] < rect["right"] and rect["top"] <= point["y"] < rect["bottom"]

    require(frame["schema_version"] == "pointer-visible-frame-v1"
        and type(frame["native_text_route_open"]) is bool)
    viewport = frame["viewport"]
    require(all(finite(viewport[key]) for key in ("left", "top", "right", "bottom"))
        and viewport["left"] == viewport["top"] == 0 and min(viewport["right"], viewport["bottom"]) > 0
        and all(finite(frame["scroll"][key]) for key in ("x", "y")))
    for name in ("container", "canvas", "panel", "visible_canvas", "visible_panel"):
        rect = frame[name]
        require(all(finite(rect[key]) for key in ("left", "top", "right", "bottom", "width", "height"))
            and rect["width"] > 8 and rect["height"] > 8
            and abs(rect["width"] - (rect["right"] - rect["left"])) < .001
            and abs(rect["height"] - (rect["bottom"] - rect["top"])) < .001)
    require(all(abs(frame["container"][key] - frame["canvas"][key]) <= 1
        for key in ("left", "top", "right", "bottom")))
    for name, rectangles in (("visible_canvas", (frame["canvas"], frame["container"], viewport)),
            ("visible_panel", (frame["panel"], viewport))):
        for key in ("left", "top", "right", "bottom"):
            expected = (max if key in ("left", "top") else min)(rect[key] for rect in rectangles)
            require(abs(frame[name][key] - expected) < .001)
    for name, count, bounds in (("legacy_full_container_samples", 16, "container"),
            ("visible_canvas_samples", 36, "visible_canvas"), ("visible_panel_samples", 9, "visible_panel")):
        samples = frame[name]
        require(isinstance(samples, list) and len(samples) == count)
        require(len({(point["x"], point["y"]) for point in samples}) == count)
        for point in samples:
            require(all(finite(point[key]) for key in ("x", "y")) and within(point, frame[bounds])
                and all(type(point[key]) is bool for key in ("hit_is_canvas", "hit_is_context"))
                and not (point["hit_is_canvas"] and point["hit_is_context"]))
            if point["hit_is_canvas"]:
                require(within(point, frame["visible_canvas"]) and point["hit_tag"] == "CANVAS"
                    and isinstance(point["hit_html"], str) and point["hit_html"].startswith("<canvas"))
            if point["hit_is_context"]:
                require(within(point, frame["visible_panel"]) and bool(point["hit_tag"]) and bool(point["hit_html"]))
    require(frame["endpoint_is_context"] is True and bool(frame["endpoint_html"]))
    for name, samples, flag in (("start", "visible_canvas_samples", "hit_is_canvas"),
            ("end", "visible_panel_samples", "hit_is_context")):
        point = frame[name]
        require(isinstance(point, dict) and all(finite(point[key]) for key in ("x", "y")))
        require(any(all(sample[key] == point[key] for key in ("x", "y")) and sample[flag] is True
            for sample in frame[samples]))
    require(math.hypot(frame["start"]["x"] - frame["end"]["x"], frame["start"]["y"] - frame["end"]["y"]) > 8)
    return core._isolation_timestamp(frame["observed_at"])


def _canvas_focus_transition(core, receipt, overlay, raw_click, parent, readiness):
    """Bind the native panel close between the proven release and real map click."""
    core._require(receipt["schema_version"] == "native-canvas-focus-transition-v1"
        and receipt["action"] == "native_text_route_summary_click"
        and receipt["selector"] == ".textual-map-alternative summary"
        and receipt["click"]["trusted"] is True and receipt["click"]["target_is_summary"] is True
        and receipt["before"]["native_text_route_open"] is True
        and receipt["after"]["native_text_route_open"] is False, "e2e1_waypoint_native_focus_transition_invalid")
    for name in ("map_ack_before", "map_ack_after"):
        _same_parent(core, receipt[name], parent)
    framing = overlay["framing"]
    core._require(framing["action"] == "locator_scroll_into_view_if_needed" and framing["selector"] == "#geographic-map"
        and framing["map_camera_modified"] is False and framing["text_route_toggled"] is False
        and framing["before"]["native_text_route_open"] is True and framing["after"]["native_text_route_open"] is True
        and framing["after"]["start"] == overlay["start"] and framing["after"]["end"] == overlay["end"],
        "e2e1_waypoint_overlay_framing_unbound")
    before_pan, after_pan, before_close, after_close = [_visible_frame(core, frame)
        for frame in (framing["before"], framing["after"], receipt["before"], receipt["after"])]
    core._require(any(point["hit_is_canvas"] for point in receipt["after"]["legacy_full_container_samples"]),
        "e2e1_waypoint_canvas_still_obstructed")
    click = raw_click["native_click"]
    core._require(raw_click["action"] == "real_canvas_click" and click["trusted"] is True
        and click["target_is_canvas"] is True and all(type(click[key]) in (int, float) and math.isfinite(click[key])
            and abs(click[key] - raw_click["pixels"][key]) <= 1 for key in ("x", "y")),
        "e2e1_waypoint_focus_followup_click_invalid")
    core._require(before_pan <= after_pan <= core._isolation_timestamp(readiness["screenshot_started_at"])
        <= core._isolation_timestamp(readiness["screenshot_completed_at"]) <= before_close
        <= core._isolation_timestamp(receipt["click"]["at"]) <= after_close <= core._isolation_timestamp(click["at"]),
        "e2e1_waypoint_focus_transition_order_invalid")
    core._require(all(frame["viewport"]["right"] == readiness["viewport"]["width"]
        and frame["viewport"]["bottom"] == readiness["viewport"]["height"]
        and frame["container"]["width"] == readiness["container"]["width"]
        and frame["container"]["height"] == readiness["container"]["height"]
        for frame in (framing["before"], framing["after"], receipt["before"], receipt["after"])),
        "e2e1_waypoint_focus_readiness_bounds_mismatch")


def _verify(root, base_url, core):
    base = root / SCENARIO
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    core._require(len(cases) == 1 and cases[0].get("name") == TEST_NAME
        and cases[0].get("classname") == "tests.product_app.browser.test_waypoint_pointer_live"
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()),
        "e2e1_waypoint_test_inventory_invalid")
    assertions = core._load(base / "browser_assertions.json")
    core._require(assertions["schema_version"] == "waypoint-pointer-browser-v1"
        and all(assertions.get(key) is True for key in ("assertions_passed", "real_canvas_click",
            "real_marker_pointer_drag", "real_canvas_overlay_release", "real_marker_overlay_release", "accepted_plan_unchanged"))
        and assertions.get("setup_api_mutations") is False and assertions.get("evaluated_repair_claimed") is False
        and assertions.get("final_draft_count") == 0, "e2e1_waypoint_assertions_invalid")
    records = _http(core, base, base_url)
    calls = [(http, request, response) for http, request, response in records if http["method"] == "POST"]
    core._require(not any(urlsplit(http["url"]).path.endswith(("/preview", "/accept", "/copilot/messages",
        "/draft/operations")) for http, _, _ in records), "e2e1_waypoint_unexpected_mutation")
    creates = [row for row in calls if urlsplit(row[0]["url"]).path == "/api/sessions"]
    snaps = [row for row in calls if row[0]["url"].endswith("/map/snap-preview")]
    confirms = [row for row in calls if "/map/snap-previews/" in row[0]["url"] and row[0]["url"].endswith("/confirm")]
    undos = [row for row in calls if row[0]["url"].endswith("/draft/undo")]
    core._require(len(creates) == 1 and len(snaps) == len(confirms) == len(undos) == 2,
        "e2e1_waypoint_mutation_counts_invalid")
    operations = []
    for index, expected in enumerate(("add_custom_waypoint", "relocate_custom_waypoint")):
        _, snap_request, snap = snaps[index]
        _, _, confirm = confirms[index]
        operation = confirm["operation"]
        core._require(snap["operation_intent"] == snap_request["operation_intent"] == operation["type"] == expected
            and snap["persisted"] is False and snap["validation_state"] == "route_checked"
            and snap["snap_preview_id"] in confirms[index][0]["url"]
            and confirm["feedback_tier"] == "route_checked" and confirm["evaluated_repair"] is False
            and len(confirm["session"]["draft"]) == index + 1
            and confirm["session"]["draft"][-1] == operation, "e2e1_waypoint_confirm_identity_invalid")
        operations.append(operation)
    added, moved = operations
    core._require(added["target"] == moved["target"] == assertions["waypoint_id"]
        and added["operation_id"] == assertions["added_operation_id"]
        and moved["operation_id"] == assertions["relocated_operation_id"]
        and added["operation_id"] != moved["operation_id"]
        and all(added["parameters"][key] == moved["parameters"][key]
                for key in ("name", "role", "duration", "day", "insertion"))
        and added["parameters"]["snapped_coordinate"] != moved["parameters"]["snapped_coordinate"],
        "e2e1_waypoint_relocation_semantics_invalid")
    for index, expected in enumerate((moved, added)):
        response = undos[index][2]
        core._require(response["undone"] == expected and response["session"]["draft"] == ([added] if index == 0 else []),
            "e2e1_waypoint_undo_identity_invalid")
    checkpoints = core._load(base / "map_checkpoints.json")
    core._require(set(checkpoints) == set(LABELS), "e2e1_waypoint_checkpoint_universe_invalid")
    parent = checkpoints["initial_parent"]["map_ack"]
    core._require(parent["active_plan_id"] == assertions["accepted_plan_id"]
        and parent["active_plan_content_hash"] == assertions["accepted_plan_content_hash"]
        and parent["rendered_geometry_hash"] == assertions["accepted_geometry_hash"],
        "e2e1_waypoint_parent_identity_invalid")
    for label, operation in zip(LABELS, (None, None, added, moved, added, None, None), strict=True):
        row = checkpoints[label]
        _same_parent(core, row["map_ack"], parent)
        _overlay(core, row["overlay"], operation, parent)
    inputs = json.loads((base / "pointer_inputs.json").read_text(encoding="utf-8"))
    core._require(isinstance(inputs, list) and len(inputs) == 3
        and inputs[0]["action"] == "real_canvas_pan_release_over_context", "e2e1_waypoint_overlay_release_missing")
    from verify_e2e1_marker_overlay_release import verify as verify_marker_overlay_release
    from verify_e2e1_overlay_release import verify as verify_overlay_release

    overlay_ready = core._load(base / "canvas_overlay_release.readiness.json")
    verify_overlay_release(inputs[0], parent, overlay_ready)
    _canvas_focus_transition(core, core._load(base / "canvas_focus_transition.json"), inputs[0], inputs[1], parent, overlay_ready)
    verify_marker_overlay_release(assertions, inputs[-1])
    _pointer(core, inputs[1:], added, snaps[1][1], snaps[1][2])
    core._verify_journey_captures(base, LABELS, {parent["active_plan_id"]})
    errors = core._load(base / "browser_errors.json")
    core._require(set(errors) == {"page_errors", "console_errors", "request_failures"}
        and errors["page_errors"] == errors["console_errors"] == [],
        "e2e1_waypoint_browser_errors")
    from verify_e2e1_network_observations import verify as verify_network

    verify_network(base, base_url, errors["request_failures"],
        {"sources":inputs[0]["observation"]["sources_before"],"sprite":inputs[0]["observation"]["sprite_before"]})
    return {"scenario": SCENARIO, "verified": True, "evaluated_repair": False,
        "accepted_plan_id": parent["active_plan_id"], "waypoint_id": added["target"], "provider_calls": 0}


def verify(root, base_url):
    """Scenario checks only; outer runner must also bind source, runtime and guard."""
    import verify_e2e1_vertical_slice as core

    try:
        return _verify(root.resolve(), base_url, core)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_waypoint_evidence_malformed") from error
