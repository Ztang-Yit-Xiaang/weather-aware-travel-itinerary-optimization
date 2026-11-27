"""Independent original/accepted route-checked draft journeys, never evaluation."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _proposal
from verify_e2e1_waypoint_pointer import _overlay, _pointer, _same_parent

SCENARIO = "waypoint_contexts"
TEST = "test_route_checked_waypoint_contexts_preserve_accepted_plan"
CONTEXTS = ("original", "accepted_child")
CONTROL_SELECTORS = {
    "select_draft_waypoint": "[data-draft-waypoint]",
    "relocate_draft_waypoint": "[data-relocate-waypoint]",
    "relocate_custom": "#relocate-custom-waypoint",
    "remove_custom": "#remove-custom-waypoint",
    "exploratory_meaning": "#exploratory-meaning",
    "check_road_access": "#check-road-access",
    "waypoint_duration": "#waypoint-duration-mode",
    "confirm_waypoint": "#confirm-map-waypoint-form",
    "select_draft_marker": ".draft-waypoint-marker",
}
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"


def require(condition, reason):
    core._require(condition, "e2e1_waypoint_contexts_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exploratory_cancel(value, context, parent):
    require(
        value["schema_version"] == "waypoint-exploratory-cancel-v1"
        and value["context"] == context
        and value["assertions_passed"] is True
        and value["draft"] == []
        and value["cancel_control_removed"] is True
        and value["focus_returned_to_edit"] is True,
        "cancel_state_invalid",
    )
    require(value["before_map"] == value["after_map"], "cancel_changed_map")
    for key in (
        "active_plan_id",
        "active_plan_content_hash",
        "rendered_map_plan_id",
        "rendered_geometry_hash",
        "rendered_marker_ids",
    ):
        require(value["after_map"][key] == parent[key], "cancel_parent_identity_invalid")
    require(
        value["raw_coordinate"] == [-121.94165, 36.51710]
        and value["exploratory_click"]["trusted"] is True
        and value["cancel_click"]["trusted"] is True
        and value["cancel_click"]["id"] == "cancel-exploratory"
        and value["exploratory_click"]["target"] == "geographic-map canvas"
        and all(
            finite(value["pixels"][key])
            and finite(value["exploratory_click"][key])
            and abs(value["pixels"][key] - value["exploratory_click"][key]) <= 1
            for key in ("x", "y")
        )
        and all(row["method"] == "GET" for row in value["requests"]),
        "cancel_input_or_mutation_invalid",
    )


def exploratory_cancel_raw(value, requests, inputs):
    start, end = value["request_start"], value["request_end"]
    require(
        type(start) is int
        and type(end) is int
        and 0 <= start <= end <= len(requests)
        and value["requests"] == requests[start:end],
        "cancel_request_slice_unbound",
    )
    require(
        isinstance(inputs["canvas_clicks"], list)
        and inputs["canvas_clicks"]
        and isinstance(inputs["cancel_clicks"], list)
        and len(inputs["cancel_clicks"]) == 1
        and inputs["canvas_clicks"][-1] == value["exploratory_click"]
        and inputs["cancel_clicks"][0] == value["cancel_click"],
        "cancel_raw_inputs_unbound",
    )


def control_proofs(rows, events, requests, records, context, parent):
    """Bind only these nine actual waypoint controls, not all map interactions."""
    expected = {key: 1 for key in CONTROL_SELECTORS}
    expected.update(exploratory_meaning=2, check_road_access=2, waypoint_duration=2, confirm_waypoint=3)
    require(
        isinstance(rows, list)
        and len(rows) == sum(expected.values())
        and {key: sum(row.get("binding_id") == key for row in rows) for key in expected} == expected,
        "control_inventory_invalid",
    )
    used = []
    confirmed = [
        json.loads(row["response_body_utf8"])["operation"]
        for row in records
        if row["method"] == "POST" and "/map/snap-previews/" in row["url"] and row["url"].endswith("/confirm")
    ]
    for row in rows:
        binding = row["binding_id"]
        event = row["event"]
        kind = (
            "change"
            if binding in {"exploratory_meaning", "waypoint_duration"}
            else "submit"
            if binding == "confirm_waypoint"
            else "click"
        )
        index, start, end = row["event_index"], row["request_start"], row["request_end"]
        require(
            row["context"] == context
            and row["assertions_passed"] is True
            and row["evaluated_repair"] is False
            and row["selector"] == CONTROL_SELECTORS[binding]
            and type(index) is int
            and 0 <= index < len(events)
            and events[index] == event
            and event["trusted"] is True
            and event["type"] == kind
            and event["selector"] == row["selector"]
            and finite(event["at"])
            and type(start) is int
            and type(end) is int
            and 0 <= start <= end <= len(requests)
            and row["requests"] == requests[start:end],
            "control_input_unbound",
        )
        used.append(index)
        if row["selector"].startswith("#"):
            require(event["id"] == row["selector"][1:], "control_dom_id_invalid")
        before, after = row["before"], row["after"]
        _same_parent(core, before["map"], parent)
        _same_parent(core, after["map"], parent)
        mutations = [item for item in row["requests"] if item["method"] != "GET"]
        http, operation = row["http"], row["operation"]
        if http is None:
            require(not mutations and before["draft"] == after["draft"], "control_unexpected_mutation")
        else:
            require(
                sum({key: value for key, value in actual.items() if key != "phase"} == http for actual in records) == 1
                and http["response_status_code"] == 200
                and http["method"] == "POST",
                "control_http_unbound",
            )
            suffix = (
                "/map/snap-preview"
                if binding == "check_road_access"
                else "/confirm"
                if binding == "confirm_waypoint"
                else "/draft/undo"
            )
            require(
                binding in {"check_road_access", "confirm_waypoint", "remove_custom"}
                and http["url"].endswith(suffix)
                and sum(item["url"] == http["url"] and item["method"] == "POST" for item in mutations) == 1
                and all(
                    item["method"] == "POST"
                    and (
                        item["url"] == http["url"]
                        or binding != "check_road_access"
                        and item["url"].endswith("/draft/impact-preview")
                    )
                    for item in mutations
                ),
                "control_mutation_window_invalid",
            )
        if binding in {"select_draft_waypoint", "relocate_draft_waypoint", "select_draft_marker", "relocate_custom"}:
            require(
                operation in confirmed
                and operation in before["draft"]
                and operation["type"] == "add_custom_waypoint"
                and after["ui"]["selected_waypoints"] == [operation["target"]]
                and after["ui"]["title"] == operation["parameters"]["name"]
                and len(after["ui"]["selected_markers"]) == 1
                and operation["parameters"]["name"] in after["ui"]["selected_markers"][0],
                "control_selection_result_invalid",
            )
            if binding != "relocate_custom":
                require(before["ui"]["selected_waypoints"] == [], "control_selection_not_changed")
            if binding in {"select_draft_waypoint", "relocate_draft_waypoint"}:
                require(event["waypoint_id"] == operation["target"], "control_waypoint_id_unbound")
            if binding == "select_draft_marker":
                require(operation["parameters"]["name"] in event["label"], "control_marker_label_unbound")
            if binding in {"relocate_draft_waypoint", "relocate_custom"}:
                require(
                    "Drag the selected custom waypoint marker" in (after["ui"]["toast"] or ""),
                    "control_relocation_guidance_missing",
                )
        elif binding == "exploratory_meaning":
            require(
                http is None
                and operation is None
                and event["value"] in {"route_waypoint", "possible_stop"}
                and before["ui"]["meaning"] == "explore_only"
                and after["ui"]["meaning"] == event["value"]
                and after["ui"]["form"] is False
                and "No road-access check has run" in after["ui"]["snap"],
                "control_meaning_result_invalid",
            )
        elif binding == "waypoint_duration":
            fields = after["ui"]["duration_fields"]
            mode = event["value"]
            require(
                http is None
                and operation is None
                and mode in {"range", "exact"}
                and after["ui"]["duration"] == mode
                and before["ui"]["duration"] != mode
                and len(fields) == 3
                and {item["id"] for item in fields}
                == {"waypoint-duration-" + key for key in ("preferred", "minimum", "maximum")},
                "control_duration_inventory_invalid",
            )
            for item in fields:
                enabled = item["id"].endswith("preferred") if mode == "exact" else not item["id"].endswith("preferred")
                require(
                    item["disabled"] is not enabled and item["visible"] is enabled and item["required"] is enabled,
                    "control_duration_result_invalid",
                )
        elif binding == "check_road_access":
            require(
                http is not None and operation is None and before["draft"] == after["draft"] == [],
                "control_snap_staged_draft",
            )
            snap = json.loads(http["response_body_utf8"])
            request = json.loads(http["request_body_utf8"])
            meaning = before["ui"]["meaning"]
            require(
                meaning in {"route_waypoint", "possible_stop"}
                and snap["validation_state"] == "route_checked"
                and snap["persisted"] is False
                and snap["operation_intent"]
                == request["operation_intent"]
                == ("add_route_waypoint" if meaning == "route_waypoint" else "add_custom_waypoint")
                and after["ui"]["form"] is True
                and "route_checked" in after["ui"]["snap"],
                "control_snap_result_invalid",
            )
        else:
            require(http is not None and operation in confirmed, "control_operation_unbound")
            result = json.loads(http["response_body_utf8"])
            require(
                result["session"]["accepted_plan_id"] == parent["active_plan_id"]
                and result["session"]["proposal"] is None
                and after["draft"] == result["session"]["draft"],
                "control_draft_result_invalid",
            )
            if binding == "confirm_waypoint":
                require(
                    result["confirmed"] is True
                    and result["evaluated_repair"] is False
                    and result["feedback_tier"] == "route_checked"
                    and result["operation"] == operation
                    and after["draft"] == [*before["draft"], operation],
                    "control_confirmation_result_invalid",
                )
            else:
                require(
                    result["undone"] == operation and before["draft"] == [operation] and after["draft"] == [],
                    "control_remove_result_invalid",
                )
    require(used == sorted(set(used)), "control_event_order_invalid")
    require(
        {row["event"]["value"] for row in rows if row["binding_id"] == "exploratory_meaning"}
        == {"route_waypoint", "possible_stop"}
        and {row["event"]["value"] for row in rows if row["binding_id"] == "waypoint_duration"} == {"exact", "range"}
        and {row["operation"]["type"] for row in rows if row["binding_id"] == "confirm_waypoint"}
        == {"add_route_waypoint", "add_custom_waypoint", "relocate_custom_waypoint"},
        "control_variant_inventory_invalid",
    )
    return {
        "scope": "Nine exact native waypoint controls with real snap/confirm/Undo; accepted plan unchanged; draft-only, not evaluated",
        "context": context,
        "accepted_plan_id": parent["active_plan_id"],
        "accepted_plan_content_hash": parent["active_plan_content_hash"],
    }


def one(records, suffix, *, method="POST", phase=None):
    found = [
        row
        for row in records
        if row["method"] == method and row["url"].endswith(suffix) and (phase is None or row["phase"] == phase)
    ]
    require(len(found) == 1, "http_call_not_unique")
    row = found[0]
    return (
        json.loads(row["request_body_utf8"]) if row["request_body_utf8"] else {},
        json.loads(row["response_body_utf8"]),
    )


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def coordinate(value):
    return (
        isinstance(value, dict)
        and set(value) == {"latitude", "longitude"}
        and finite(value["latitude"], -90)
        and value["latitude"] <= 90
        and finite(value["longitude"], -180)
        and value["longitude"] <= 180
    )


def confirmation(trace, request, result, snap_request, snap, parent, route_leg, remaining):
    operation, kind = result["operation"], trace["operation"]
    parameters = operation["parameters"]
    require(
        trace["scope"] == "route_checked_draft_only"
        and trace["evaluated_repair"] is False
        and trace["assertions_passed"] is True
        and trace["variant"] == "default"
        and trace["feedback_tier"] == result["feedback_tier"] == "route_checked"
        and result["evaluated_repair"] is False
        and trace["parent"] == parent
        and trace["source"] == operation["source"] == "confirmed_map_interaction"
        and trace["operation_output"] == operation
        and trace["operation_input"] == request
        and trace["selector"] == "#confirm-map-waypoint-form button[type=submit]",
        "draft_scope_or_io_mismatch",
    )
    require(
        result["confirmed"] is True
        and type(snap["session_revision"]) is int
        and snap["session_revision"] >= 0
        and type(request["expected_revision"]) is int
        and type(snap_request["expected_revision"]) is int
        and request["expected_revision"] == snap_request["expected_revision"] == snap["session_revision"]
        and type(result["session"]["revision"]) is int
        and result["session"]["revision"] == snap["session_revision"] + 1,
        "confirmation_revision_unbound",
    )
    require(
        set(parameters)
        == {
            "schema_version",
            "snap_preview_id",
            "waypoint_id",
            "name",
            "day",
            "role",
            "duration",
            "raw_coordinate",
            "snapped_coordinate",
            "selected_access_point",
            "insertion",
            "affected_route_legs",
        }
        and parameters["schema_version"] == "confirmed-map-operation-v1"
        and re.fullmatch(r"snap_[0-9a-f]{32}", parameters["snap_preview_id"])
        and re.fullmatch(r"waypoint_[0-9a-f]{32}", operation["target"])
        and parameters["waypoint_id"] == operation["target"],
        "canonical_operation_invalid",
    )
    require(
        snap["schema_version"] == "map-snap-preview-v1"
        and snap["validation_state"] == "route_checked"
        and snap["persisted"] is False
        and snap["draft_append_allowed"] is True
        and snap_request["operation_intent"] == snap["operation_intent"] == operation["type"] == kind
        and operation["target"] == snap["entity_id"]
        and type(parameters["day"]) is int
        and snap_request["selected_day"] == snap["selected_day"] == parameters["day"] == 5
        and snap_request["travel_mode"] == snap["travel_mode"] == "driving"
        and snap_request["selected_route_segment_id"]
        == snap["selected_route_segment_id"]
        == parameters["insertion"]["route_leg_id"]
        == route_leg["route_leg_id"],
        "snap_context_disconnected",
    )
    require(
        parameters["insertion"]
        == {
            "route_leg_id": route_leg["route_leg_id"],
            "predecessor_id": route_leg["origin_id"],
            "successor_id": route_leg["destination_id"],
            "travel_mode": "driving",
        }
        and snap_request.get("target_waypoint_id")
        == (operation["target"] if kind == "relocate_custom_waypoint" else None),
        "insertion_or_relocation_target_unbound",
    )
    require(
        snap["raw_coordinate"] == {key: snap_request[key] for key in ("longitude", "latitude")}
        and all(
            parameters[key] == snap[key]
            for key in (
                "snap_preview_id",
                "raw_coordinate",
                "snapped_coordinate",
                "selected_access_point",
                "affected_route_legs",
            )
        ),
        "confirmed_geometry_changed",
    )
    require(
        result["session"]["accepted_plan_id"] == parent["active_plan_id"]
        and result["session"]["proposal"] is None
        and result["session"]["draft"] == [*remaining, operation],
        "confirmation_changed_parent_or_draft",
    )
    require(
        type(snap["confirmation_required"]) is bool
        and request["warning_acknowledged"] is snap["confirmation_required"]
        and finite(snap["snap_distance_m"])
        and snap["snap_distance_m"] <= 500
        and snap["confirmation_required"] is (snap["snap_distance_m"] > 100),
        "required_warning_not_confirmed",
    )
    require(
        coordinate(parameters["raw_coordinate"]) and coordinate(parameters["snapped_coordinate"]), "coordinate_invalid"
    )
    access = parameters["selected_access_point"]
    require(
        set(access)
        == {
            "access_point_id",
            "access_type",
            "coordinate",
            "source",
            "road_validated",
            "access_confidence",
            "evidence_refs",
        }
        and re.fullmatch(r"access_[0-9a-f]{32}", access["access_point_id"])
        and access["access_type"] == "road_snap"
        and access["source"] == "runtime_osrm"
        and access["road_validated"] is True
        and access["access_confidence"] == "road_snap_only"
        and access["coordinate"] == parameters["snapped_coordinate"],
        "access_point_unbound",
    )
    legs = parameters["affected_route_legs"]
    require(
        len(legs) == 2
        and len({leg["route_leg_id"] for leg in legs}) == 2
        and [(leg["origin_id"], leg["destination_id"]) for leg in legs]
        == [(route_leg["origin_id"], operation["target"]), (operation["target"], route_leg["destination_id"])],
        "route_chain_not_same_insertion",
    )
    for leg in legs:
        require(
            leg["road_validated"] is True
            and leg["fallback_used"] is False
            and leg["validation_status"] == "road_validated"
            and leg["geometry"]["type"] == "LineString"
            and len(leg["geometry"]["coordinates"]) >= 2
            and all(
                isinstance(point, list)
                and len(point) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in point)
                and -180 <= point[0] <= 180
                and -90 <= point[1] <= 90
                for point in leg["geometry"]["coordinates"]
            )
            and leg["route_leg_id"] == "preview_leg_" + leg["query_hash"][:16]
            and re.fullmatch(r"[0-9a-f]{64}", leg["query_hash"])
            and leg["evidence_refs"] == ["route_query:" + leg["query_hash"]]
            and finite(leg["distance_m"])
            and leg["distance_m"] > 0
            and finite(leg["duration_s"])
            and leg["duration_s"] > 0
            and finite(leg["snap_distance_origin_m"])
            and finite(leg["snap_distance_destination_m"])
            and all(
                leg[key] == value
                for key, value in {
                    "provider": "runtime_osrm",
                    "routing_status": "osrm_route_validated",
                    "travel_mode": "driving",
                    "geometry_source": "runtime_osrm_geojson",
                    "distance_source": "runtime_osrm_route",
                    "duration_source": "runtime_osrm_route",
                }.items()
            ),
            "unvalidated_or_unbound_road_leg",
        )
        core._isolation_timestamp(leg["retrieved_at"])
    point = [parameters["snapped_coordinate"][key] for key in ("longitude", "latitude")]
    refs = ["route_query:" + leg["query_hash"] for leg in legs]
    require(
        legs[0]["geometry"]["coordinates"][-1] == legs[1]["geometry"]["coordinates"][0] == point
        and operation["evidence_refs"] == access["evidence_refs"] == refs,
        "road_geometry_or_provenance_disconnected",
    )
    expected_name = (
        trace["context"] + " real-pointer " + ("via point" if kind == "add_route_waypoint" else "custom stop")
    )
    require(parameters["name"] == expected_name, "name_semantics_changed")
    if kind == "add_route_waypoint":
        require(
            parameters["role"] == "route_waypoint" and parameters["duration"] is None, "route_waypoint_became_visit"
        )
    else:
        require(
            kind in {"add_custom_waypoint", "relocate_custom_waypoint"}
            and parameters["role"] == "scenic_stop"
            and parameters["duration"]
            == {"mode": "exact", "preferred_minutes": 45, "minimum_minutes": 45, "maximum_minutes": 45},
            "custom_visit_semantics_changed",
        )
    expected_request = {
        "expected_revision": snap["session_revision"],
        "warning_acknowledged": snap["confirmation_required"],
    }
    if kind != "relocate_custom_waypoint":
        expected_request.update({key: parameters[key] for key in ("name", "role", "duration")})
    require(request == expected_request, "form_semantics_not_canonical")
    if remaining:
        require(
            kind == "relocate_custom_waypoint"
            and len(remaining) == 1
            and remaining[0]["target"] == operation["target"]
            and all(
                remaining[0]["parameters"][key] == parameters[key]
                for key in ("name", "role", "duration", "day", "insertion")
            ),
            "relocation_changed_visit_metadata",
        )
    return operation


def native_click(row, kind, route_leg, snap_request):
    click, pixels = row["trusted_dom_event"], row["pixels"]
    require(
        row["action"] == "native_canvas_click"
        and row["operation"] == kind
        and row["route_leg"] == route_leg
        and row["raw_coordinate"] == [-121.94165, 36.51710]
        and click["trusted"] is True
        and click["target"] == "geographic-map canvas"
        and all(
            type(click[key]) in (int, float) and math.isfinite(click[key]) and abs(click[key] - pixels[key]) <= 1
            for key in ("x", "y")
        ),
        "native_click_unbound",
    )
    require(
        all(
            type(snap_request[key]) in (int, float)
            and math.isfinite(snap_request[key])
            and abs(snap_request[key] - row["raw_coordinate"][index]) <= 0.00002
            for index, key in enumerate(("longitude", "latitude"))
        ),
        "raw_click_not_snap_request",
    )
    canvas = pixels["canvas"]
    require(
        all(finite(pixels[key]) for key in ("x", "y", "zoom"))
        and all(finite(canvas[key]) for key in ("x", "y", "width", "height"))
        and canvas["width"] > 0
        and canvas["height"] > 0
        and canvas["x"] < pixels["x"] < canvas["x"] + canvas["width"]
        and canvas["y"] < pixels["y"] < canvas["y"] + canvas["height"],
        "native_click_outside_canvas",
    )


def mutation_order(records, session_id, context):
    """Bind the captured native journey to one session and actual HTTP order."""
    prefix = "/api/sessions/" + session_id
    allowed = {
        "/api/sessions",
        *(
            prefix + suffix
            for suffix in (
                "/selection",
                "/draft/impact-preview",
                "/draft/operations",
                "/preview",
                "/accept",
                "/draft/undo",
                "/map/snap-preview",
            )
        ),
    }
    actual, previous_revision = [], -1
    for row in records:
        if row["method"] == "GET":
            continue
        path = urlsplit(row["url"]).path
        is_confirm = re.fullmatch(re.escape(prefix) + r"/map/snap-previews/snap_[0-9a-f]{32}/confirm", path)
        require(row["method"] == "POST" and (path in allowed or is_confirm), "unexpected_mutation_or_session")
        if not (is_confirm or path.endswith(("/map/snap-preview", "/draft/undo"))):
            continue
        request, result = json.loads(row["request_body_utf8"]), json.loads(row["response_body_utf8"])
        revision = request["expected_revision"]
        require(type(revision) is int and revision >= previous_revision, "waypoint_revision_regressed")
        if path.endswith("/map/snap-preview"):
            kind, action = result["operation_intent"], "snap"
            require(result["session_revision"] == revision, "snap_revision_disconnected")
        else:
            operation = result["operation" if is_confirm else "undone"]
            kind, action = operation["type"], "confirm" if is_confirm else "undo"
            session = result["session"]
            require(
                session["session_id"] == session_id
                and type(session["revision"]) is int
                and session["revision"] == revision + 1,
                "mutation_revision_or_session_disconnected",
            )
            if not is_confirm:
                require(set(request) == {"expected_revision"}, "undo_request_invalid")
            revision = session["revision"]
        previous_revision = revision
        if action != "undo":
            require(row["phase"] == kind, "waypoint_phase_disconnected")
        actual.append((kind, action))
    expected = [("add_route_waypoint", action) for action in ("snap", "confirm", "undo")]
    expected += [("add_custom_waypoint", action) for action in ("snap", "confirm")]
    expected += [("relocate_custom_waypoint", action) for action in ("snap", "confirm", "undo")]
    expected += [("add_custom_waypoint", "undo")]
    require(actual == expected, "waypoint_operation_order_invalid")


def capability_binding(snapshot, parent, views):
    require(
        snapshot["schema_version"] == "runtime-capability-snapshot-v1"
        and snapshot["plan_id"] == parent["active_plan_id"]
        and snapshot["plan_content_hash"] == parent["active_plan_content_hash"],
        "capability_context_mismatch",
    )
    require(
        any(
            view["session"]["session_id"] == snapshot["session_id"]
            and view["session"]["accepted_plan_id"] == snapshot["plan_id"]
            and view["session"]["draft"] == []
            and view["session"]["proposal"] is None
            and view["workspace"]["map_edit_capabilities"] == snapshot["capabilities"]["map_edit_capabilities"]
            for view in views
        ),
        "capability_snapshot_not_http_bound",
    )


def accepted_baseline(records, parent, reload):
    relevant = [
        row
        for row in records
        if row["method"] == "POST" and row["url"].endswith(("/draft/operations", "/preview", "/accept"))
    ]
    require(
        [urlsplit(row["url"]).path.rsplit("/", 1)[-1] for row in relevant] == ["operations", "preview", "accept"]
        and all(row["phase"] == "baseline_replace_preview_accept" for row in relevant),
        "baseline_order_invalid",
    )
    stage_request, staged = one(relevant, "/draft/operations")
    preview_request, preview = one(relevant, "/preview")
    accept_request, acceptance = one(relevant, "/accept")
    operation, proposal = staged["operation"], preview["proposal"]
    require(
        stage_request["type"] == operation["type"] == "replace_nearby"
        and stage_request["target"] == operation["target"] == BIXBY
        and stage_request["parameters"] == operation["parameters"] == {"candidate_id": PFEIFFER}
        and stage_request["source"] == operation["source"] == "map"
        and staged["session"]["draft"] == preview["session"]["draft"] == [operation]
        and staged["session"]["accepted_plan_id"] == proposal["parent_plan_id"]
        and preview_request["expected_revision"] == staged["session"]["revision"]
        and preview["session"]["proposal"] == proposal
        and proposal["compiled_request"]["operations"]
        == [{key: value for key, value in operation.items() if key != "created_at"}],
        "baseline_replacement_disconnected",
    )
    _proposal(proposal, parent)
    execution = proposal["runtime_execution"]
    certificate = execution["evaluation_certificate"]
    require(
        execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
        and certificate["failures"] == []
        and [row["stop_id"] for row in execution["repair_request"]["candidate_pois"]] == [PFEIFFER]
        and BIXBY not in parent["rendered_marker_ids"]
        and PFEIFFER in parent["rendered_marker_ids"],
        "baseline_evaluator_or_marker_mismatch",
    )
    decision, pointer = acceptance["result"]["decision"], acceptance["result"]["pointer"]
    for record in (decision, pointer):
        require(
            record["content_hash"]
            == core._canonical_sha256({key: value for key, value in record.items() if key != "content_hash"}),
            "baseline_acceptance_hash_invalid",
        )
    require(
        accept_request["confirmation"] == "accept_evaluated_repair"
        and accept_request["expected_revision"] == preview["session"]["revision"]
        and decision["decision_type"] == "accept_repair"
        and decision["idempotency_key"] == accept_request["idempotency_key"]
        and decision["selected_plan_id"] == pointer["accepted_plan_id"] == parent["active_plan_id"]
        and decision["selected_plan_content_hash"]
        == pointer["accepted_plan_content_hash"]
        == parent["active_plan_content_hash"]
        and decision["previous_plan_id"] == pointer["original_plan_id"] == proposal["parent_plan_id"]
        and pointer["decision_id"] == decision["decision_id"]
        and decision["certificate_id"] == proposal["certificate_id"]
        and decision["certificate_content_hash"] == proposal["certificate_content_hash"]
        and decision["diff_id"] == proposal["diff_id"]
        and decision["diff_content_hash"] == proposal["diff_content_hash"]
        and acceptance["session"]["accepted_plan_id"]
        == reload["session"]["accepted_plan_id"]
        == parent["active_plan_id"]
        and acceptance["session"]["draft"] == reload["session"]["draft"] == []
        and acceptance["session"]["proposal"] is reload["session"]["proposal"] is None
        and reload["workspace"]["accepted_plan_pointer"]
        == {
            key: pointer[key]
            for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
        },
        "accepted_baseline_unbound",
    )
    return preview, acceptance


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require(
        [case.get("name") for case in cases] == [TEST + "[" + context + "]" for context in CONTEXTS]
        and all(case.get("classname") == "tests.product_app.browser.test_waypoint_contexts_live" for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()),
        "browser_universe_invalid",
    )
    require(
        {path.name for path in (root / SCENARIO).iterdir() if path.is_dir()} == set(CONTEXTS),
        "context_directory_universe_invalid",
    )
    for context in CONTEXTS:
        base = root / SCENARIO / context
        assertions = core._load(base / "browser_assertions.json")
        require(
            assertions["schema_version"] == "waypoint-context-browser-v1"
            and assertions["context"] == context
            and assertions["assertions_passed"] is True
            and assertions["route_checked_draft_only"] is True
            and assertions["overlay_regression_claimed"] is False
            and assertions["evaluated_waypoint_repair_claimed"] is False,
            "false_completion_claim",
        )
        parent, checkpoints = assertions["parent"], core._load(base / "checkpoints.json")
        cancel = core._load(base / "exploratory_cancel.json")
        exploratory_cancel(cancel, context, parent)
        exploratory_cancel_raw(
            cancel, load(base / "browser_requests.json"), core._load(base / "exploratory_input_log.json")
        )
        _same_parent(core, parent, parent)
        traces, inputs = load(base / "operation_assertions.json"), load(base / "native_pointer_inputs.json")
        require(
            traces == assertions["operations"]
            and inputs == assertions["native_pointer_inputs"]
            and checkpoints == assertions["checkpoints"],
            "assertion_projection_mismatch",
        )
        expected = ["add_route_waypoint", "add_custom_waypoint", "relocate_custom_waypoint"]
        require(
            [row["operation"] for row in traces] == expected
            and all(row["context"] == context for row in traces)
            and len(inputs) == len(expected),
            "operation_context_universe_invalid",
        )
        records = _http(base, base_url)
        posts = [row for row in records if row["method"] == "POST"]
        baseline_count = int(context == "accepted_child")
        require(not any(row["url"].endswith("/copilot/messages") for row in records), "unexpected_chat")
        for suffix, count in (
            ("/api/sessions", 1),
            ("/draft/operations", baseline_count),
            ("/preview", baseline_count),
            ("/accept", baseline_count),
            ("/map/snap-preview", len(expected)),
            ("/confirm", len(expected)),
            ("/draft/undo", len(expected)),
        ):
            require(sum(row["url"].endswith(suffix) for row in posts) == count, "mutation_inventory_invalid")
        snapshot = core._load(base / "capability_snapshot.json")
        require(
            snapshot["schema_version"] == "runtime-capability-snapshot-v1"
            and snapshot["context"] == context
            and snapshot["plan_id"] == parent["active_plan_id"]
            and snapshot["plan_content_hash"] == parent["active_plan_content_hash"],
            "capability_context_mismatch",
        )
        mutation_order(records, snapshot["session_id"], context)
        views = [
            json.loads(row["response_body_utf8"])
            for row in records
            if row["method"] == "GET" and row["url"].endswith("/api/sessions/" + snapshot["session_id"])
        ]
        _, created = one(records, "/api/sessions")
        require(created["session"]["session_id"] == snapshot["session_id"], "created_session_disconnected")
        capability_binding(snapshot, parent, [created, *views])
        candidates = [
            plan
            for view in views
            for plan in view["workspace"]["geography"]["plans"]
            if plan["plan_id"] == parent["active_plan_id"]
        ]
        if not candidates:
            _, created = one(records, "/api/sessions")
            candidates = [
                plan
                for plan in created["workspace"]["geography"]["plans"]
                if plan["plan_id"] == parent["active_plan_id"]
            ]
        require(candidates and all(plan == candidates[0] for plan in candidates), "canonical_parent_geography_missing")
        geography = candidates[0]
        require(
            geography["content_hash"] == parent["active_plan_content_hash"]
            and core._completed_race_geometry_hash(geography) == parent["rendered_geometry_hash"]
            and {row["properties"]["stop_id"] for row in geography["stops"]["features"]}
            == set(parent["rendered_marker_ids"]),
            "parent_geography_identity_mismatch",
        )
        incoming = [
            row["properties"]
            for row in geography["validated_legs"]["features"]
            if row["properties"]["destination_id"] == (BIXBY if context == "original" else PFEIFFER)
        ]
        require(len(incoming) == 1, "route_context_missing")
        route_leg = incoming[0]
        operations, snaps = {}, {}
        for trace in traces:
            kind = trace["operation"]
            capability = snapshot["capabilities"]["map_edit_capabilities"]["operations"][kind]
            require(
                capability["enabled"] is True
                and capability["preview_executable"] is False
                and capability["evaluated_repair"] is False,
                "capability_overclaims_evaluation",
            )
            snap_request, snap = one(records, "/map/snap-preview", phase=kind)
            request, result = one(records, "/map/snap-previews/" + snap["snap_preview_id"] + "/confirm", phase=kind)
            require(
                core._load(base / (kind + "_confirmation.json")) == {"request": request, "response": result},
                "confirmation_file_unbound",
            )
            remaining = [operations["add_custom_waypoint"]] if kind == "relocate_custom_waypoint" else []
            operations[kind] = confirmation(trace, request, result, snap_request, snap, parent, route_leg, remaining)
            snaps[kind] = (snap_request, snap)
        labels = {
            "context_baseline": None,
            "add_route_waypoint_staged": operations["add_route_waypoint"],
            "undo_add_route_waypoint": None,
            "parent_after_final_reload": None,
        }
        for index, kind in enumerate(expected[:2]):
            native_click(inputs[index], kind, route_leg, snaps[kind][0])
        if "relocate_custom_waypoint" in operations:
            added, moved, drag = operations["add_custom_waypoint"], operations["relocate_custom_waypoint"], inputs[-1]
            require(
                drag["action"] == "native_marker_drag"
                and drag["overlay_release_tested"] is False
                and added["target"] == moved["target"] == drag["waypoint_id"]
                and all(
                    added["parameters"][key] == moved["parameters"][key]
                    for key in ("name", "role", "duration", "day", "insertion")
                )
                and added["parameters"]["snapped_coordinate"] != moved["parameters"]["snapped_coordinate"],
                "relocation_semantics_changed",
            )
            _pointer(
                core,
                [
                    {**inputs[1], "action": "real_canvas_click"},
                    {
                        **drag,
                        "action": "real_marker_pointer_drag",
                        "from_pixels": drag["start"],
                        "to_pixels": drag["destination"],
                        "trusted_dom_mouse_events": drag["trusted_dom_events"],
                    },
                ],
                added,
                *snaps["relocate_custom_waypoint"],
            )
            labels.update(
                add_custom_waypoint_staged=added,
                custom_relocated=moved,
                undo_custom_relocation=added,
                undo_add_custom_waypoint=None,
            )
        require(set(checkpoints) == set(labels), "checkpoint_universe_invalid")
        for label, operation in labels.items():
            row = checkpoints[label]
            _same_parent(core, row["map_ack"], parent)
            _overlay(core, row["overlay"], operation, parent)
            ready = core._load(base / (label + ".readiness.json"))
            require(
                ready["rendered_plan_id"] == parent["active_plan_id"]
                and ready["rendered_geometry_hash"] == parent["rendered_geometry_hash"],
                "capture_checkpoint_disconnected",
            )
            if operation:
                require("evaluated repair is not enabled" in row["draft_status"], "draft_presented_as_evaluated")
        for kind in expected:
            label = "undo_custom_relocation" if kind == "relocate_custom_waypoint" else "undo_" + kind
            result = core._load(base / (label + ".json"))
            matches = [
                json.loads(row["response_body_utf8"])
                for row in posts
                if row["url"].endswith("/draft/undo")
                and json.loads(row["response_body_utf8"])["undone"] == operations[kind]
            ]
            require(
                matches == [result]
                and result["session"]["accepted_plan_id"] == parent["active_plan_id"]
                and result["session"]["proposal"] is None
                and result["session"]["draft"]
                == ([operations["add_custom_waypoint"]] if kind == "relocate_custom_waypoint" else []),
                "undo_disconnected",
            )
        _, reload = one(records, "/api/sessions/" + snapshot["session_id"], method="GET", phase="final_reload")
        require(
            reload["session"]["session_id"] == snapshot["session_id"]
            and reload["session"]["accepted_plan_id"] == parent["active_plan_id"]
            and reload["session"]["draft"] == []
            and reload["session"]["proposal"] is None,
            "final_reload_changed_parent",
        )
        control_proofs(
            load(base / "waypoint_control_receipts.json"),
            load(base / "waypoint_control_input_log.json"),
            load(base / "browser_requests.json"),
            records,
            context,
            parent,
        )
        allowed_plans = {parent["active_plan_id"]}
        captures = ["initial_parent", *labels]
        if baseline_count:
            preview, acceptance = accepted_baseline(records, parent, reload)
            require(
                preview == core._load(base / "baseline_preview_response.json")
                and acceptance == core._load(base / "baseline_acceptance_response.json"),
                "baseline_http_unbound",
            )
            allowed_plans.add(preview["proposal"]["parent_plan_id"])
            captures.append("baseline_preview_ready")
        core._verify_journey_captures(base, captures, allowed_plans)
        require(load(base / "console_errors.json") == [], "browser_errors")
        from verify_e2e1_network_observations import verify as verify_network
        from verify_e2e1_network_observations import verify_asset_observation

        assets = load(base / "map_asset_sources.json")
        verify_asset_observation(assets, parent, context, phase="final_reload")
        verify_network(
            base,
            base_url,
            load(base / "network_failures.json"),
            {"sources": assets["vector_sources"], "sprite": assets["sprite"]},
            asset_observation=assets,
        )
        llm = core._load(base / "llm_io.json")
        require(
            type(llm["provider_call_count"]) is int
            and llm["provider_call_count"] == 0
            and llm["llm_input"] is None
            and llm["llm_output"] is None,
            "llm_scope_invalid",
        )
    return {
        "verified": True,
        "test_count": 2,
        "route_checked_operations": 6,
        "evaluated_waypoint_repair_claimed": False,
        "overlay_regression_claimed": False,
    }


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        StopIteration,
        ET.ParseError,
    ) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_waypoint_contexts_evidence_malformed") from error
