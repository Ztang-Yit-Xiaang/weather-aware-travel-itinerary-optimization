"""Synthetic adversarial checks, not claims that new browser steps have run."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e2e1_runtime_gate_coverage as coverage  # noqa: E402
import verify_e2e1_evaluated_variants as evaluated  # noqa: E402
import verify_e2e1_poi_controls as poi  # noqa: E402
import verify_e2e1_self_managed as conversation  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
import verify_e2e1_waypoint_contexts as waypoint  # noqa: E402


def parent():
    return {
        "active_plan_id": "parent",
        "active_plan_content_hash": "a" * 64,
        "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "b" * 64,
        "rendered_marker_ids": ["bixby_creek_bridge_viewpoint"],
        "preview_state": "PARENT_RENDERED",
    }


def conversation_case(name):
    identity = parent()
    clicks = [
        {
            "trusted": True,
            "id": "",
            "label": "Edit message",
            "action_group": "Message recovery actions"
            if name == "fake_sdk_cancel"
            else "Conversation capacity actions",
        }
    ]
    value = {
        "schema_version": "conversation-control-closure-v1",
        "scenario": name,
        "assertions_passed": True,
        "new_chat_request_count": 0,
        "streams_before": 2,
        "streams_after": 2,
        "raw_clicks": clicks,
        "accepted_map": deepcopy(identity),
        "edit_message_composer": "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park."
        if name == "fake_sdk_cancel"
        else "Review the safer repair.",
        "fake_attempts_before": 1,
        "fake_attempts_after": 1,
        "settings_opened": True,
        "settings_closed": True,
    }
    if name != "fake_sdk_cancel":
        clicks.extend(
            [
                {
                    "trusted": True,
                    "id": "",
                    "label": "Conversation settings",
                    "action_group": "Conversation capacity actions",
                },
                {"trusted": True, "id": "close-transcript-settings", "label": "Close", "action_group": ""},
            ]
        )
    records = []
    if name == "preference_memory":
        clicks[:] = [{"trusted": True, "id": "delete-all-conversations", "label": "Delete all", "action_group": ""}]
        context = {
            "accepted_plan_id": identity["active_plan_id"],
            "accepted_plan_content_hash": identity["active_plan_content_hash"],
            "draft_operations": [],
            "conversation_memory": {"proposed_preferences": []},
        }
        request, response = {"confirmation": "delete_all_conversations"}, {"deleted_count": 1}
        value.update(
            visible_turn_count=0,
            before=deepcopy(context),
            after=deepcopy(context),
            delete_all_request=request,
            delete_all_response=response,
        )
        records = [
            {
                "method": "DELETE",
                "url": "http://127.0.0.1:8127/api/conversations",
                "request_body_utf8": json.dumps(request),
                "response_body_utf8": json.dumps(response),
            }
        ]
    return value, deepcopy(clicks), {"initial": identity}, records


def verify_conversation(tmp_path, name, value, clicks, trace, records):
    files = {"control_closure.json": value, "control_clicks.json": clicks}
    if name.removesuffix("_accepted_child") == "preference_memory":
        files.update(
            {"context_before_delete_all.json": value["before"], "context_after_delete_all.json": value["after"]}
        )
    for filename, content in files.items():
        (tmp_path / filename).write_text(json.dumps(content), encoding="utf-8")
    conversation._conversation_control_closure(tmp_path, name, trace, records)


@pytest.mark.parametrize("name", ["preference_memory", "capacity_turns", "capacity_bytes", "fake_sdk_cancel"])
@pytest.mark.parametrize("context", ["original", "accepted_child"])
def test_conversation_control_exact_inputs_and_outputs_verify(tmp_path, name, context):
    value, clicks, trace, records = conversation_case(name)
    if context == "accepted_child":
        if name == "fake_sdk_cancel":
            value["edit_message_composer"] = "Suggest a safe itinerary change for the accepted plan."
        name += "_accepted_child"
    value["scenario"] = name
    verify_conversation(tmp_path, name, value, clicks, trace, records)


@pytest.mark.parametrize("name", ["preference_memory", "capacity_turns", "capacity_bytes", "fake_sdk_cancel"])
@pytest.mark.parametrize(
    "fault", ["new_request", "stream_delta", "unbound_click", "untrusted_click", "different_plan", "different_geometry"]
)
@pytest.mark.parametrize("context", ["original", "accepted_child"])
def test_conversation_control_receipt_cannot_claim_unexecuted_or_mutating_action(tmp_path, name, fault, context):
    value, clicks, trace, records = conversation_case(name)
    if context == "accepted_child":
        name += "_accepted_child"
    value["scenario"] = name
    if fault == "new_request":
        value["new_chat_request_count"] = 1
    elif fault == "stream_delta":
        value["streams_after"] += 1
    elif fault == "unbound_click":
        clicks.clear()
    elif fault == "untrusted_click":
        value["raw_clicks"][0]["trusted"] = clicks[0]["trusted"] = False
    elif fault == "different_plan":
        value["accepted_map"]["active_plan_id"] = "child"
    elif fault == "different_geometry":
        value["accepted_map"]["rendered_geometry_hash"] = "c" * 64
    with pytest.raises(core.E2E1VerificationError):
        verify_conversation(tmp_path, name, value, clicks, trace, records)


@pytest.mark.parametrize(
    "fault",
    [
        "http_missing",
        "http_duplicate",
        "request_changed",
        "response_changed",
        "nonempty_turns",
        "draft_changed",
        "memory_retained",
    ],
)
def test_delete_all_preserves_exact_confirmed_deletion_and_plan_boundary(tmp_path, fault):
    value, clicks, trace, records = conversation_case("preference_memory")
    if fault == "http_missing":
        records.clear()
    elif fault == "http_duplicate":
        records.append(deepcopy(records[0]))
    elif fault == "request_changed":
        records[0]["request_body_utf8"] = "{}"
    elif fault == "response_changed":
        records[0]["response_body_utf8"] = '{"deleted_count":0}'
    elif fault == "nonempty_turns":
        value["visible_turn_count"] = 1
    elif fault == "draft_changed":
        value["after"]["draft_operations"] = ["unexpected"]
    elif fault == "memory_retained":
        value["after"]["conversation_memory"]["proposed_preferences"] = ["unexpected"]
    with pytest.raises(core.E2E1VerificationError):
        verify_conversation(tmp_path, "preference_memory", value, clicks, trace, records)


@pytest.mark.parametrize("name", ["capacity_turns", "capacity_bytes", "fake_sdk_cancel"])
def test_edit_message_does_not_replace_the_original_user_input(tmp_path, name):
    value, clicks, trace, records = conversation_case(name)
    value["edit_message_composer"] = "Different semantics"
    with pytest.raises(core.E2E1VerificationError):
        verify_conversation(tmp_path, name, value, clicks, trace, records)


def cancel_case(context="original"):
    identity = parent()
    return {
        "schema_version": "waypoint-exploratory-cancel-v1",
        "context": context,
        "assertions_passed": True,
        "draft": [],
        "cancel_control_removed": True,
        "focus_returned_to_edit": True,
        "before_map": deepcopy(identity),
        "after_map": deepcopy(identity),
        "raw_coordinate": [-121.94165, 36.51710],
        "pixels": {"x": 300, "y": 400},
        "exploratory_click": {"trusted": True, "target": "geographic-map canvas", "x": 300, "y": 400},
        "cancel_click": {"trusted": True, "id": "cancel-exploratory"},
        "requests": [],
    }, identity


@pytest.mark.parametrize("context", ["original", "accepted_child"])
def test_cancel_is_only_an_exploratory_draft_free_action(context):
    value, identity = cancel_case(context)
    waypoint.exploratory_cancel(value, context, identity)


@pytest.mark.parametrize(
    "fault",
    [
        "changed_map",
        "stale_parent",
        "new_draft",
        "control_remains",
        "lost_focus",
        "raw_point_changed",
        "untrusted_canvas",
        "untrusted_cancel",
        "wrong_cancel",
        "wrong_canvas",
        "pixel_disconnected",
        "pixel_nan",
        "api_mutation",
    ],
)
def test_cancel_receipt_rejects_false_input_or_changed_plan(fault):
    value, identity = cancel_case()
    if fault == "changed_map":
        value["after_map"]["rendered_geometry_hash"] = "other"
    elif fault == "stale_parent":
        identity["active_plan_id"] = "other"
    elif fault == "new_draft":
        value["draft"] = ["unexpected"]
    elif fault == "control_remains":
        value["cancel_control_removed"] = False
    elif fault == "lost_focus":
        value["focus_returned_to_edit"] = False
    elif fault == "raw_point_changed":
        value["raw_coordinate"][0] = 0
    elif fault == "untrusted_canvas":
        value["exploratory_click"]["trusted"] = False
    elif fault == "untrusted_cancel":
        value["cancel_click"]["trusted"] = False
    elif fault == "wrong_cancel":
        value["cancel_click"]["id"] = "other"
    elif fault == "wrong_canvas":
        value["exploratory_click"]["target"] = "other"
    elif fault == "pixel_disconnected":
        value["pixels"]["x"] += 20
    elif fault == "pixel_nan":
        value["pixels"]["x"] = float("nan")
    elif fault == "api_mutation":
        value["requests"] = [{"method": "POST", "url": "http://127.0.0.1:8127/api/sessions/s/draft/operations"}]
    with pytest.raises(core.E2E1VerificationError):
        waypoint.exploratory_cancel(value, "original", identity)


@pytest.mark.parametrize(
    "fault", [None, "hidden_request", "slice_start", "bool_index", "wrong_raw_canvas", "missing_cancel"]
)
def test_cancel_raw_log_is_exact_not_a_summary(fault):
    value, _ = cancel_case()
    requests = [{"method": "GET", "url": "http://127.0.0.1:8127/api/map/config"}]
    value.update(request_start=0, request_end=1, requests=deepcopy(requests))
    raw = {"canvas_clicks": [deepcopy(value["exploratory_click"])], "cancel_clicks": [deepcopy(value["cancel_click"])]}
    if fault == "hidden_request":
        requests[0]["method"] = "POST"
    elif fault == "slice_start":
        value["request_start"] = 1
    elif fault == "bool_index":
        value["request_end"] = True
    elif fault == "wrong_raw_canvas":
        raw["canvas_clicks"][0]["x"] = 0
    elif fault == "missing_cancel":
        raw["cancel_clicks"] = []
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            waypoint.exploratory_cancel_raw(value, requests, raw)
    else:
        waypoint.exploratory_cancel_raw(value, requests, raw)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("fault", [None, "alias", "disabled", "untrusted", "path", "request", "operation", "raw"])
def test_preview_and_undo_aliases_require_exact_real_control_and_canonical_ids(position, fault):
    value = {
        "schema_version": "evaluated-action-aliases-v1",
        "context": "original",
        "assertions_passed": True,
        "request_id": "request",
        "operation_id": "operation",
    }
    for action in ("preview", "undo"):
        selector = (
            ("#inline-", "#draft-", "#context-")[position] + action
            if action == "preview"
            else ("#inline-", "#context-")[position % 2] + action
        )
        repairs = (action == "preview" and position == 2) or (action == "undo" and position % 2 == 1)
        event = {"id": selector[1:], "trusted": True, "path": "/app/repairs" if repairs else "/app/itinerary"}
        value[action] = {
            "selector": selector,
            "visible": True,
            "disabled": False,
            "path": event["path"],
            "event": event,
            "raw_clicks": [event],
        }
    if fault == "alias":
        value["undo"]["selector"] = "#wrong"
    elif fault == "disabled":
        value["undo"]["disabled"] = True
    elif fault == "untrusted":
        value["undo"]["event"]["trusted"] = False
    elif fault == "path":
        value["undo"]["path"] = "/app/compare"
    elif fault == "request":
        value["request_id"] = "other"
    elif fault == "operation":
        value["operation_id"] = "other"
    elif fault == "raw":
        value["undo"]["raw_clicks"] = []
    args = (
        value,
        {"control": value["preview"]},
        "original",
        position,
        {"compiled_request": {"source_request_id": "request"}},
        {"operation_id": "operation"},
    )
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            evaluated.control_aliases(*args)
    else:
        evaluated.control_aliases(*args)


@pytest.mark.parametrize(
    "fault", [None, "duplicate_id", "wrong_display", "wrong_access", "name", "extra_source", "hidden_not_empty"]
)
def test_public_candidate_sources_must_match_exact_http_candidate_geography(fault):
    candidate = {
        "candidate_id": "candidate",
        "place": {"name": "Park", "display_coordinate": {"longitude": -121, "latitude": 36}},
        "selected_access_point": {"coordinate": {"longitude": -120.9, "latitude": 36.1}},
    }
    sources = {
        source: {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"candidate_id": "candidate", "name": "Park"},
                    "geometry": {"type": "Point", "coordinates": point},
                }
            ],
        }
        for source, point in zip(poi.SOURCES, [[-121, 36], [-120.9, 36.1]], strict=True)
    }
    expected = [candidate]
    if fault == "duplicate_id":
        sources[poi.SOURCES[0]]["features"] *= 2
    elif fault == "wrong_display":
        sources[poi.SOURCES[0]]["features"][0]["geometry"]["coordinates"][0] = 0
    elif fault == "wrong_access":
        sources[poi.SOURCES[1]]["features"][0]["geometry"]["coordinates"][0] = 0
    elif fault == "name":
        sources[poi.SOURCES[0]]["features"][0]["properties"]["name"] = "Other"
    elif fault == "extra_source":
        sources["unknown"] = {}
    elif fault == "hidden_not_empty":
        expected = []
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            poi.features(sources, expected)
    else:
        poi.features(sources, expected)


def responsive_case():
    plan = {
        "plan_id": "parent",
        "content_hash": "a" * 64,
        "validated_legs": {"type": "FeatureCollection", "features": []},
        "stops": {"type": "FeatureCollection", "features": [{"properties": {"stop_id": "stop"}}]},
    }
    dataset = {
        "renderedMapPlanId": "parent",
        "renderedPlanContentHash": "a" * 64,
        "renderedGeometryHash": core._completed_race_geometry_hash(plan),
        "renderedMarkerIds": '["stop"]',
        "mapReady": "true",
        "mapTileCoverage": "1",
    }
    views = [
        {
            "width": width,
            "height": height,
            "mobile_toggle_visible": index == 1,
            "maps": [
                {"pane": pane, "visible": index != 1 or pane == "option", "dataset": deepcopy(dataset)}
                for pane in ("baseline", "option")
            ],
        }
        for index, (width, height) in enumerate(((1440, 900), (390, 844), (1440, 900)))
    ]
    rows = [
        {
            "context": "original",
            "assertions_passed": True,
            "accepted_plan_id": "parent",
            "accepted_plan_content_hash": "a" * 64,
            "views": views,
        }
    ]
    return rows, {"active_plan_id": "parent", "active_plan_content_hash": "a" * 64}, [plan]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_width",
        "missing_view",
        "visible_baseline_mobile",
        "no_toggle",
        "bad_hash",
        "low_coverage",
        "not_ready",
        "stale_plan",
        "wrong_markers",
    ],
)
def test_active_compare_resize_preserves_both_exact_route_identities(fault):
    rows, identity, plans = responsive_case()
    views = rows[0]["views"]
    if fault == "wrong_width":
        views[1]["width"] = 800
    elif fault == "missing_view":
        views.pop()
    elif fault == "visible_baseline_mobile":
        views[1]["maps"][0]["visible"] = True
    elif fault == "no_toggle":
        views[1]["mobile_toggle_visible"] = False
    elif fault == "bad_hash":
        views[1]["maps"][1]["dataset"]["renderedGeometryHash"] = "wrong"
    elif fault == "low_coverage":
        views[1]["maps"][1]["dataset"]["mapTileCoverage"] = ".94"
    elif fault == "not_ready":
        views[1]["maps"][1]["dataset"]["mapReady"] = "false"
    elif fault == "stale_plan":
        views[1]["maps"][1]["dataset"]["renderedMapPlanId"] = "stale"
    elif fault == "wrong_markers":
        views[1]["maps"][1]["dataset"]["renderedMarkerIds"] = "[]"
    if fault:
        with pytest.raises(coverage.ReceiptError):
            coverage.verify_compare_responsive(rows, "original", identity, plans)
    else:
        coverage.verify_compare_responsive(rows, "original", identity, plans)


def accepted_poi_fixture():
    identity = parent()
    identity["rendered_marker_ids"] = ["pfeiffer_big_sur_state_park"]
    leg = {"route_leg_id": "incoming_pfeiffer", "day": 5, "destination_id": "pfeiffer_big_sur_state_park"}
    candidate = {
        "candidate_id": "point_lobos",
        "place": {"name": "Lobos", "display_coordinate": {"longitude": -121, "latitude": 36}},
        "selected_access_point": {"coordinate": {"longitude": -121.1, "latitude": 36.1}},
    }
    candidate_sources = {
        source: {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"candidate_id": "point_lobos", "name": "Lobos"},
                    "geometry": {"type": "Point", "coordinates": point},
                }
            ],
        }
        for source, point in zip(poi.SOURCES, [[-121, 36], [-121.1, 36.1]], strict=True)
    }
    empty = {source: {"type": "FeatureCollection", "features": []} for source in poi.SOURCES}
    route = {
        "kind": "replacement",
        "day": 5,
        "replacement_target_id": "pfeiffer_big_sur_state_park",
        "route_leg_id": "incoming_pfeiffer",
    }

    def http(query, insertion=False):
        context = {**route, "kind": "insertion", "replacement_target_id": None} if insertion else route
        query += "&expected_revision=2&day=5&route_leg_id=incoming_pfeiffer"
        if not insertion:
            query += "&replacement_target_id=pfeiffer_big_sur_state_park"
        body = json.dumps(
            {
                "session_id": "s",
                "session_revision": 2,
                "context": context,
                "candidates": [] if insertion else [candidate],
            }
        )
        digest = hashlib.sha256(body.encode()).hexdigest()
        empty_hash = hashlib.sha256(b"").hexdigest()
        return {
            "schema_version": "e2e1-product-http-io-v1",
            "method": "GET",
            "url": "http://127.0.0.1:8127/api/sessions/s/map/poi-candidates?" + query,
            "request_body_utf8": "",
            "request_body_sha256": empty_hash,
            "request_original_body_sha256": empty_hash,
            "response_status_code": 200,
            "response_body_utf8": body,
            "response_body_sha256": digest,
            "response_original_body_sha256": digest,
            "credential_redactions": {"request": [], "response": []},
            "body_capture": "exact_utf8",
            "request_headers_persisted": False,
            "response_headers_persisted": False,
            "authorization_persisted": False,
            "session_token_persisted": False,
        }

    events = [
        {"type": "click", "trusted": True, "id": "prepare-replacement-candidates"},
        {"type": "click", "trusted": True, "id": "find-route-candidates"},
    ]
    discovered = http("limit=5")
    raw = {
        "context": "accepted_child",
        "target": "pfeiffer_big_sur_state_park",
        "selected_leg": leg,
        "preparation_events": deepcopy(events[:1]),
        "discovery_events": deepcopy(events),
        "discovery_http": discovered,
        "initial_candidates": [candidate],
        "parent": identity,
        "provider_calls": 0,
        "request_start": 1,
    }
    requests = [{"method": "GET", "url": discovered["url"]}]
    rows = []
    for index, selector in enumerate(poi.SELECTORS):
        kind = "submit" if index < 2 else "change" if index < 4 else "click"
        events.append({"type": kind, "trusted": True, "id": selector[1:]})
        fields = (
            (
                {"role": "meal", "max_detour_minutes": "120", "limit": "3"},
                {"role": "", "max_detour_minutes": "60", "limit": "5"},
            )[index]
            if index < 2
            else None
        )
        record = (
            http("&".join(key + "=" + value for key, value in fields.items()))
            if fields
            else http("limit=5", insertion=True)
            if index == 7
            else None
        )
        if record:
            requests.append({"method": "GET", "url": record["url"]})
        rows.append(
            {
                "selector": selector,
                "visible": True,
                "disabled": False,
                "events": deepcopy(events),
                "map": deepcopy(identity),
                "http": record,
                "fields": fields,
                "assertions_passed": True,
                "drawer_visible": index != 6,
                "before_layers": deepcopy(empty if index in {3, 5} else candidate_sources),
                "after_layers": deepcopy(empty if index in {2, 4, 6, 7} else candidate_sources),
            }
        )
    raw.update(events=events, requests=requests, request_end=len(requests))
    snapshot = {"map_ack": identity, "geography": {"validated_legs": {"features": [{"properties": leg}]}}}
    records = [
        {"phase": "accepted_poi_controls", **record}
        for record in [discovered, *[row["http"] for row in rows if row["http"]]]
    ]
    raw["replacement_reentry"] = poi_reentry_fixture(
        identity, raw["target"], leg["route_leg_id"], [candidate], rows[-1]["http"], events, candidate_sources
    )
    records.extend(
        {"phase": "accepted_poi_reentry", **raw["replacement_reentry"][key]}
        for key in ("stop_http", "leg_http", "candidate_http")
    )
    return rows, raw, snapshot, records


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "original_target",
        "foreign_leg",
        "untrusted_prepare",
        "untrusted_discover",
        "missing_record",
        "wrong_phase",
        "paid",
        "missing_control",
        "parent_changed",
        "hidden_mutation",
        "wrong_route",
        "wrong_layer_point",
    ],
)
def test_accepted_poi_controls_bind_actual_target_leg_raw_actions_and_http(fault):
    rows, raw, snapshot, records = accepted_poi_fixture()
    if fault == "original_target":
        raw["target"] = "bixby_creek_bridge_viewpoint"
    elif fault == "foreign_leg":
        raw["selected_leg"] = {**raw["selected_leg"], "route_leg_id": "foreign"}
    elif fault == "untrusted_prepare":
        raw["preparation_events"][0]["trusted"] = False
    elif fault == "untrusted_discover":
        raw["discovery_events"][-1]["trusted"] = False
    elif fault == "missing_record":
        records.pop(1)
    elif fault == "wrong_phase":
        records[0]["phase"] = "original_poi_controls"
    elif fault == "paid":
        raw["provider_calls"] = 1
    elif fault == "missing_control":
        rows.pop(3)
    elif fault == "parent_changed":
        rows[1]["map"]["active_plan_id"] = "other"
    elif fault == "hidden_mutation":
        raw["requests"][1]["method"] = "POST"
    elif fault == "wrong_route":
        raw["selected_leg"]["day"] = 7
    elif fault == "wrong_layer_point":
        rows[1]["after_layers"][poi.SOURCES[0]]["features"][0]["geometry"]["coordinates"][0] = 0
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            poi.accepted_rows(rows, raw, snapshot, records)
    else:
        assert poi.accepted_rows(rows, raw, snapshot, records)["context"] == "accepted_child"


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_candidate",
        "missing_candidate",
        "untrusted",
        "wrong_selected",
        "wrong_title",
        "changed_parent",
        "hidden_control",
        "enabled_control",
        "forced",
        "mutation",
        "foreign_capability",
        "missing_http",
        "false_reason",
    ],
)
def test_accepted_candidate_inspection_and_disabled_replacement_need_exact_evidence(fault):
    _, raw, snapshot, records = accepted_poi_fixture()
    candidate = raw["initial_candidates"][0]
    events = [{"type": "click", "trusted": True, "candidate": candidate["candidate_id"]}]
    capability = {"enabled": False}
    selection = poi_selection_wire(snapshot["map_ack"], 4, {"selected_candidate_id": candidate["candidate_id"]})
    initial_http = raw["replacement_reentry"]["candidate_http"]
    payload = json.loads(initial_http["response_body_utf8"])
    payload["session_revision"] = 5
    refreshed = poi_wire(payload, "GET", initial_http["url"].replace("expected_revision=4", "expected_revision=5"))
    inspection_records = [{"phase": "accepted_poi_inspection", **record} for record in (selection, refreshed)]
    records.extend(inspection_records)
    value = {
        "context": "accepted_child",
        "parent": snapshot["map_ack"],
        "candidate_http": initial_http,
        "capability": capability,
        "events": events,
        "requests": [{"method": record["method"], "url": record["url"]} for record in inspection_records],
        "http_records": inspection_records,
        "accepted_plan_changed": False,
        "assertions_passed": True,
        "rows": [
            {
                "candidate": candidate,
                "selected": "true",
                "title": candidate["place"]["name"],
                "events": deepcopy(events),
                "selection_http": selection,
                "refresh_http": refreshed,
                "map": snapshot["map_ack"],
                "replace_control": {
                    "visible": True,
                    "disabled": True,
                    "title": "Only an exact server-registered replacement for the selected accepted-plan stop can become a draft",
                },
                "forced_interaction": False,
                "assertions_passed": True,
            }
        ],
    }
    records.append(
        {
            "response_body_utf8": json.dumps(
                {
                    "session": {"accepted_plan_id": snapshot["map_ack"]["active_plan_id"]},
                    "workspace": {"draft_capabilities": {"operations": {"replace_nearby": capability}}},
                }
            )
        }
    )
    row = value["rows"][0]
    if fault == "wrong_candidate":
        row["candidate"] = {**candidate, "candidate_id": "foreign"}
    elif fault == "missing_candidate":
        value["rows"] = []
    elif fault == "untrusted":
        row["events"][0]["trusted"] = events[0]["trusted"] = False
    elif fault == "wrong_selected":
        row["selected"] = "false"
    elif fault == "wrong_title":
        row["title"] = "Foreign"
    elif fault == "changed_parent":
        row["map"] = {"active_plan_id": "foreign"}
    elif fault == "hidden_control":
        row["replace_control"]["visible"] = False
    elif fault == "enabled_control":
        row["replace_control"]["disabled"] = False
    elif fault == "forced":
        row["forced_interaction"] = True
    elif fault == "mutation":
        value["requests"] = [{"method": "POST"}]
    elif fault == "foreign_capability":
        value["capability"] = {"enabled": True}
    elif fault == "missing_http":
        records = []
    elif fault == "false_reason":
        row["replace_control"]["title"] = "not supported maybe"
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            poi.accepted_inspection(value, snapshot, records)
    else:
        assert poi.accepted_inspection(value, snapshot, records)["candidate_count"] == 1


def poi_wire(payload, method, url, request=None):
    request_body = json.dumps(request) if request is not None else ""
    response_body = json.dumps(payload)
    row = {
        "schema_version": "e2e1-product-http-io-v1",
        "method": method,
        "url": url,
        "response_status_code": 200,
        "body_capture": "exact_utf8",
        "credential_redactions": {"request": [], "response": []},
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
    }
    for side, body in (("request", request_body), ("response", response_body)):
        row[side + "_body_utf8"] = body
        row[side + "_body_sha256"] = row[side + "_original_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return row


def poi_selection_wire(identity, revision, expected):
    session = {
        "session_id": "s",
        "trip_id": "trip",
        "run_id": "run",
        "revision": revision + 1,
        "draft": [],
        "accepted_plan_id": identity["active_plan_id"],
        **expected,
    }
    return poi_wire(
        {"session": session},
        "POST",
        "http://127.0.0.1:8127/api/sessions/s/selection",
        {"expected_revision": revision, **expected},
    )


def poi_reentry_fixture(identity, target, leg, candidates, generic, prior_events, sources):
    events = deepcopy(prior_events)
    result = {
        "schema_version": "poi-replacement-reentry-v1",
        "assertions_passed": True,
        "target": target,
        "route_leg_id": leg,
        "generic_http": generic,
        "map": deepcopy(identity),
        "layers": deepcopy(sources),
        "visible_candidate_ids": [row["candidate_id"] for row in candidates],
    }
    for key, field, value in (
        ("stop_events", "stop", target),
        ("preparation_events", "id", "prepare-replacement-candidates"),
        ("leg_events", "leg", leg),
        ("events", "id", "find-route-candidates"),
    ):
        events.append({"type": "click", "trusted": True, field: value})
        result[key] = deepcopy(events)
    result["stop_http"] = poi_selection_wire(
        identity, 2, {"selected_day": 5, "selected_stop_id": target, "selected_segment_id": None}
    )
    result["leg_http"] = poi_selection_wire(
        identity, 3, {"selected_day": 5, "selected_stop_id": None, "selected_segment_id": leg}
    )
    payload = {
        "session_id": "s",
        "session_revision": 4,
        "context": {"kind": "replacement", "day": 5, "replacement_target_id": target, "route_leg_id": leg},
        "candidates": candidates,
    }
    result["candidate_http"] = poi_wire(
        payload,
        "GET",
        "http://127.0.0.1:8127/api/sessions/s/map/poi-candidates?"
        + "expected_revision=4&day=5&route_leg_id="
        + leg
        + "&replacement_target_id="
        + target,
    )
    return result


@pytest.mark.parametrize("context", ["original", "accepted_child"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "stale_reopen",
        "missing_target",
        "stale_revision",
        "foreign_session",
        "wrong_stop",
        "wrong_leg",
        "foreign_origin",
        "untrusted_preparation",
        "missing_click",
        "changed_map",
        "cached_candidate",
        "wrong_layer",
        "hidden_candidate",
    ],
)
def test_replacement_reentry_requires_actual_new_context_selection_and_candidates(context, fault):
    rows, raw, snapshot, _ = accepted_poi_fixture()
    identity = snapshot["map_ack"]
    target = raw["target"]
    if context == "original":
        target = "bixby_creek_bridge_viewpoint"
        identity = parent()
    value = poi_reentry_fixture(
        identity,
        target,
        raw["selected_leg"]["route_leg_id"],
        raw["initial_candidates"],
        rows[-1]["http"],
        raw["events"],
        raw["replacement_reentry"]["layers"],
    )
    if fault == "stale_reopen":
        value["generic_http"] = raw["discovery_http"]
    elif fault in {"missing_target", "stale_revision", "foreign_session"}:
        payload = json.loads(value["candidate_http"]["response_body_utf8"])
        if fault == "missing_target":
            payload["context"]["replacement_target_id"] = None
        elif fault == "stale_revision":
            payload["session_revision"] = 2
        else:
            payload["session_id"] = "foreign"
        value["candidate_http"] = poi_wire(payload, "GET", value["candidate_http"]["url"])
    elif fault == "wrong_stop":
        value["stop_http"] = poi_selection_wire(
            identity, 2, {"selected_day": 5, "selected_stop_id": "foreign", "selected_segment_id": None}
        )
    elif fault == "wrong_leg":
        value["leg_http"] = poi_selection_wire(
            identity, 3, {"selected_day": 5, "selected_stop_id": None, "selected_segment_id": "foreign"}
        )
    elif fault == "foreign_origin":
        value["stop_http"]["url"] = value["stop_http"]["url"].replace("127.0.0.1:8127", "127.0.0.1:8128")
    elif fault == "untrusted_preparation":
        value["preparation_events"][-1]["trusted"] = False
    elif fault == "missing_click":
        value["leg_events"] = value["preparation_events"]
    elif fault == "changed_map":
        value["map"]["active_plan_id"] = "foreign"
    elif fault == "cached_candidate":
        value["candidate_http"] = raw["discovery_http"]
    elif fault == "wrong_layer":
        value["layers"][poi.SOURCES[0]]["features"] = []
    elif fault == "hidden_candidate":
        value["visible_candidate_ids"] = []
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            poi.verify_reentry(value, rows[-1]["http"], raw["events"], identity, target, raw["initial_candidates"])
    else:
        assert (
            poi.verify_reentry(value, rows[-1]["http"], raw["events"], identity, target, raw["initial_candidates"])[
                "session_revision"
            ]
            == 4
        )


def test_generic_reopen_cannot_be_credited_as_replacement_context():
    rows, raw, snapshot, records = accepted_poi_fixture()
    rows[-1]["http"] = raw["discovery_http"]
    raw["requests"][-1]["url"] = raw["discovery_http"]["url"]
    with pytest.raises(core.E2E1VerificationError):
        poi.accepted_rows(rows, raw, snapshot, records)
