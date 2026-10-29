"""Offline adversarial verifier fixtures, never browser or provider evidence.

The oracle imports no UI assertion helper and sends no network requests. Exact
synthetic input/output is retained under pytest's disposable evidence directory.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as core  # noqa: E402
import verify_e2e1_waypoint_contexts as verifier  # noqa: E402

BASE = "http://127.0.0.1:8127"
STAMP = "2026-09-08T12:00:00+00:00"
KINDS = ("add_route_waypoint", "add_custom_waypoint", "relocate_custom_waypoint")


def _case(kind=KINDS[0], context="original"):
    target, snap_id = "waypoint_" + "1" * 32, "snap_" + "2" * 32
    point = {"longitude": -121.94165, "latitude": 36.51710}
    route = {"route_leg_id": "canonical_leg", "origin_id": "predecessor", "destination_id": verifier.BIXBY}
    insertion = {
        "route_leg_id": "canonical_leg",
        "predecessor_id": "predecessor",
        "successor_id": verifier.BIXBY,
        "travel_mode": "driving",
    }
    legs = []
    for index, (origin, destination, coordinates) in enumerate(
        (
            ("predecessor", target, [[-121.95, 36.52], list(point.values())]),
            (target, verifier.BIXBY, [list(point.values()), [-121.93, 36.51]]),
        ),
        3,
    ):
        query = str(index) * 64
        legs.append(
            {
                "route_leg_id": "preview_leg_" + query[:16],
                "origin_id": origin,
                "destination_id": destination,
                "query_hash": query,
                "evidence_refs": ["route_query:" + query],
                "travel_mode": "driving",
                "validation_status": "road_validated",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "distance_m": 1800,
                "duration_s": 160,
                "provider": "runtime_osrm",
                "routing_status": "osrm_route_validated",
                "geometry_source": "runtime_osrm_geojson",
                "distance_source": "runtime_osrm_route",
                "duration_source": "runtime_osrm_route",
                "road_validated": True,
                "fallback_used": False,
                "retrieved_at": STAMP,
                "snap_distance_origin_m": 2,
                "snap_distance_destination_m": 3,
            }
        )
    refs = ["route_query:" + leg["query_hash"] for leg in legs]
    access = {
        "access_point_id": "access_" + "5" * 32,
        "access_type": "road_snap",
        "coordinate": point,
        "source": "runtime_osrm",
        "road_validated": True,
        "access_confidence": "road_snap_only",
        "evidence_refs": refs,
    }
    duration = (
        None
        if kind == KINDS[0]
        else {"mode": "exact", "preferred_minutes": 45, "minimum_minutes": 45, "maximum_minutes": 45}
    )
    name = context + " real-pointer " + ("via point" if kind == KINDS[0] else "custom stop")
    parameters = {
        "schema_version": "confirmed-map-operation-v1",
        "snap_preview_id": snap_id,
        "waypoint_id": target,
        "name": name,
        "day": 5,
        "role": "route_waypoint" if kind == KINDS[0] else "scenic_stop",
        "duration": duration,
        "raw_coordinate": deepcopy(point),
        "snapped_coordinate": point,
        "selected_access_point": access,
        "insertion": insertion,
        "affected_route_legs": legs,
    }
    operation = {
        "operation_id": "draft_" + kind,
        "type": kind,
        "target": target,
        "parameters": parameters,
        "source": "confirmed_map_interaction",
        "evidence_refs": refs,
        "created_at": STAMP,
    }
    parent = {
        "active_plan_id": "parent",
        "active_plan_content_hash": "parent_hash",
        "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "geometry_hash",
        "rendered_marker_ids": [verifier.BIXBY],
        "preview_state": "PARENT_RENDERED",
    }
    snap_request = {
        "expected_revision": 7,
        **parameters["raw_coordinate"],
        "selected_day": 5,
        "selected_route_segment_id": "canonical_leg",
        "travel_mode": "driving",
        "operation_intent": kind,
    }
    if kind == KINDS[2]:
        snap_request["target_waypoint_id"] = target
    snap = {
        key: parameters[key]
        for key in (
            "snap_preview_id",
            "raw_coordinate",
            "snapped_coordinate",
            "selected_access_point",
            "affected_route_legs",
        )
    }
    snap.update(
        schema_version="map-snap-preview-v1",
        entity_id=target,
        operation_intent=kind,
        validation_state="route_checked",
        persisted=False,
        draft_append_allowed=True,
        session_revision=7,
        selected_day=5,
        selected_route_segment_id="canonical_leg",
        travel_mode="driving",
        confirmation_required=False,
        snap_distance_m=30,
        created_at=STAMP,
        expires_at="2026-09-08T12:10:00+00:00",
    )
    request = {"expected_revision": 7, "warning_acknowledged": False}
    if kind != KINDS[2]:
        request.update(name=name, role=parameters["role"], duration=deepcopy(duration))
    remaining = []
    if kind == KINDS[2]:
        previous = deepcopy(operation)
        previous.update(type=KINDS[1], operation_id="previous_add")
        remaining.append(previous)
    result = {
        "confirmed": True,
        "feedback_tier": "route_checked",
        "evaluated_repair": False,
        "operation": operation,
        "session": {
            "session_id": "session",
            "revision": 8,
            "accepted_plan_id": "parent",
            "proposal": None,
            "draft": [*remaining, operation],
        },
    }
    trace = {
        "context": context,
        "operation": kind,
        "variant": "default",
        "source": "confirmed_map_interaction",
        "selector": "#confirm-map-waypoint-form button[type=submit]",
        "feedback_tier": "route_checked",
        "scope": "route_checked_draft_only",
        "evaluated_repair": False,
        "operation_input": request,
        "operation_output": operation,
        "parent": parent,
        "assertions_passed": True,
    }
    return [trace, request, result, snap_request, snap, parent, route, remaining]


def _write_io(tmp_path, inputs, output):
    (tmp_path / "verifier_exact_io.json").write_text(
        json.dumps(
            {
                "scope": "offline_synthetic_verifier_fixture",
                "browser_execution_claimed": False,
                "provider_call_count": 0,
                "input": inputs,
                "output": output,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _controls(context):
    """Synthetic alias-verifier fixture; canonical operation checks have separate tests."""
    route, custom, moved = (_case(kind, context) for kind in KINDS)
    parent = route[5]
    rows, events, requests, records = [], [], [], []
    def state(draft=(), **ui):
        return {"map": deepcopy(parent), "draft": deepcopy(list(draft)), "ui": {
            "meaning":None,"form":False,"duration":None,"duration_fields":[],"title":"",
            "selected_waypoints":[],"selected_markers":[],"toast":"",**ui}}
    def add(binding,before,after,*,value=None,operation=None,http=None):
        selector = verifier.CONTROL_SELECTORS[binding]
        event = {"type":"change" if binding in {"exploratory_meaning","waypoint_duration"} else "submit"
            if binding=="confirm_waypoint" else "click", "trusted":True,"selector":selector,
            "id":selector[1:] if selector.startswith("#") else "", "value":value,
            "waypoint_id":operation["target"] if operation and selector.startswith("[data-") else None,
            "label":operation["parameters"]["name"] if operation else "Unit native label", "at":1770000000000+len(rows)}
        start = len(requests)
        if http:
            requests.append({"method":http["method"],"url":http["url"]})
            records.append(deepcopy(http))
        rows.append({"binding_id":binding,"context":context,"selector":selector,"event_index":len(events),"event":deepcopy(event),
            "before":before,"after":after,"request_start":start,"request_end":len(requests),"requests":deepcopy(requests[start:]),
            "http":http,"operation":deepcopy(operation),"assertions_passed":True,"evaluated_repair":False})
        events.append(event)
    def http(suffix,request,response):
        return {"method":"POST","url":BASE+"/api/sessions/session"+suffix,"response_status_code":200,
            "request_body_utf8":json.dumps(request),"response_body_utf8":json.dumps(response)}
    def confirm(case,before):
        operation = case[2]["operation"]
        result = deepcopy(case[2])
        result["session"]["draft"] = [*before,operation]
        add("confirm_waypoint",state(before,form=True),state(result["session"]["draft"]),operation=operation,
            http=http("/map/snap-previews/"+case[4]["snap_preview_id"]+"/confirm",case[1],result))
    for case,meaning in ((route,"route_waypoint"),(custom,"possible_stop")):
        add("exploratory_meaning",state(meaning="explore_only"),state(meaning=meaning,snap="No road-access check has run."),value=meaning)
        add("check_road_access",state(meaning=meaning),state(meaning=meaning,form=True,snap="Road-check result: route_checked."),
            http=http("/map/snap-preview",case[3],case[4]))
        if case is custom:
            previous = "exact"
            for mode in ("range","exact"):
                fields = []
                for field in ("preferred","minimum","maximum"):
                    enabled = field=="preferred" if mode=="exact" else field!="preferred"
                    fields.append({"id":"waypoint-duration-"+field,"disabled":not enabled,"visible":enabled,"required":enabled})
                add("waypoint_duration",state(duration=previous),state(duration=mode,duration_fields=fields),value=mode)
                previous=mode
        confirm(case,[])
    operation = custom[2]["operation"]
    selected = {"selected_waypoints":[operation["target"]],"selected_markers":[operation["parameters"]["name"]+". Custom draft stop."],
        "title":operation["parameters"]["name"],"toast":"Drag the selected custom waypoint marker to propose a new raw location."}
    for binding in ("select_draft_waypoint","relocate_draft_waypoint","select_draft_marker","relocate_custom"):
        add(binding,state([operation],**(selected if binding=="relocate_custom" else {})),state([operation],**selected),operation=operation)
    confirm(moved,[operation])
    result = {"undone":operation,"session":{"accepted_plan_id":parent["active_plan_id"],"draft":[],"proposal":None}}
    add("remove_custom",state([operation],**selected),state(),operation=operation,http=http("/draft/undo",{"expected_revision":10},result))
    return rows,events,requests,records,context,parent


@pytest.mark.parametrize("context",verifier.CONTEXTS)
def test_all_nine_waypoint_aliases_have_exact_native_control_results(context,tmp_path):
    arguments = _controls(context)
    output = verifier.control_proofs(*arguments)
    assert output["context"]==context and output["accepted_plan_id"]==arguments[-1]["active_plan_id"]
    _write_io(tmp_path,arguments,output)


@pytest.mark.parametrize("context",verifier.CONTEXTS)
def test_waypoint_projection_credits_only_the_nine_proven_bindings_in_the_recorded_context(context,tmp_path):
    import e2e1_runtime_gate_coverage as gates
    from test_e2e1_runtime_gate_coverage import bundle

    rows,events,requests,records,_,parent = _controls(context)
    prefix = "waypoint_contexts/"+context+"/"
    artifacts = {"waypoint_control_receipts.json":rows,"waypoint_control_input_log.json":events,
        "browser_requests.json":requests,"browser_assertions.json":{"parent":parent},
        "http_index.json":[{"file":f"http_{i}.json"} for i in range(len(records))]}
    artifacts.update({f"http_{i}.json":record for i,record in enumerate(records)})
    receipt = bundle(tmp_path,"waypoint-contexts",{prefix+name:value for name,value in artifacts.items()})
    proofs,missing = gates.existing_control_proofs(gates.Receipts(tmp_path,[receipt]))
    assert set(proofs)=={(binding,context) for binding in verifier.CONTROL_SELECTORS}
    assert all(proofs[(binding,context)]["selector"]==selector for binding,selector in verifier.CONTROL_SELECTORS.items())
    assert any(row["mode"]=="waypoint-contexts" and row["context"]!=context for row in missing)


@pytest.mark.parametrize("binding",verifier.CONTROL_SELECTORS)
@pytest.mark.parametrize("fault",["untrusted","wrong_context","changed_map","unbound_input"])
def test_waypoint_control_projection_rejects_unbound_or_false_native_action(binding,fault,tmp_path):
    args = _controls("accepted_child")
    row = next(item for item in args[0] if item["binding_id"]==binding)
    if fault=="untrusted":
        row["event"]["trusted"]=args[1][row["event_index"]]["trusted"]=False
    elif fault=="wrong_context":
        row["context"]="original"
    elif fault=="changed_map":
        row["after"]["map"]["rendered_geometry_hash"]="stale"
    else:
        row["event_index"]=len(args[1])
    with pytest.raises(core.E2E1VerificationError) as error:
        verifier.control_proofs(*args)
    _write_io(tmp_path,args,{"rejected":str(error.value)})


@pytest.mark.parametrize("fault",["meaning_unchanged","duration_fields_unchanged","wrong_selected_id","wrong_marker_name",
    "missing_relocation_hint","road_check_staged","confirm_parent_only","remove_kept_draft","extra_api_call","missing_binding"])
def test_waypoint_control_visible_result_is_not_a_click_only_claim(fault,tmp_path):
    args = _controls("original")
    rows,events,requests,records,_,_ = args
    def first(binding):
        return next(row for row in rows if row["binding_id"]==binding)
    if fault=="meaning_unchanged":
        first("exploratory_meaning")["after"]["ui"]["meaning"]="explore_only"
    elif fault=="duration_fields_unchanged":
        first("waypoint_duration")["after"]["ui"]["duration_fields"][0]["disabled"]=False
    elif fault=="wrong_selected_id":
        first("select_draft_waypoint")["after"]["ui"]["selected_waypoints"]=["other"]
    elif fault=="wrong_marker_name":
        first("select_draft_marker")["after"]["ui"]["selected_markers"]=["other"]
    elif fault=="missing_relocation_hint":
        first("relocate_custom")["after"]["ui"]["toast"]=""
    elif fault=="road_check_staged":
        first("check_road_access")["after"]["draft"]=["unexpected"]
    elif fault=="confirm_parent_only":
        first("confirm_waypoint")["after"]["draft"]=[]
    elif fault=="remove_kept_draft":
        row=first("remove_custom")
        row["after"]["draft"]=deepcopy(row["before"]["draft"])
    elif fault=="extra_api_call":
        row=first("select_draft_marker")
        requests.append({"method":"POST","url":BASE+"/api/sessions/session/accept"})
        row.update(request_start=len(requests)-1,request_end=len(requests),requests=deepcopy(requests[-1:]))
    else:
        rows.pop()
    with pytest.raises(core.E2E1VerificationError) as error:
        verifier.control_proofs(*args)
    _write_io(tmp_path,args,{"rejected":str(error.value)})


@pytest.mark.parametrize("kind,context", [(kind, context) for context in verifier.CONTEXTS for kind in KINDS])
def test_road_checked_confirmations_keep_their_exact_draft_only_semantics(tmp_path, kind, context):
    case = _case(kind, context)
    output = verifier.confirmation(*case)
    assert output == case[2]["operation"]
    _write_io(tmp_path, case, {"verified": True, "operation": output, "evaluated_repair": False})


@pytest.mark.parametrize(
    "fault,code",
    [
        ("request_name", "form_semantics_not_canonical"),
        ("request_role", "form_semantics_not_canonical"),
        ("request_duration", "form_semantics_not_canonical"),
        ("request_revision", "confirmation_revision_unbound"),
        ("snap_revision", "confirmation_revision_unbound"),
        ("result_revision", "confirmation_revision_unbound"),
        ("boolean_revision", "confirmation_revision_unbound"),
        ("not_confirmed", "confirmation_revision_unbound"),
        ("schema", "canonical_operation_invalid"),
        ("extra_parameter", "canonical_operation_invalid"),
        ("waypoint_id", "canonical_operation_invalid"),
        ("malformed_snap_id", "canonical_operation_invalid"),
        ("insertion_origin", "insertion_or_relocation_target_unbound"),
        ("insertion_mode", "insertion_or_relocation_target_unbound"),
        ("extra_add_target", "insertion_or_relocation_target_unbound"),
        ("unacknowledged_warning", "required_warning_not_confirmed"),
        ("oversize_snap", "required_warning_not_confirmed"),
        ("negative_snap", "required_warning_not_confirmed"),
        ("boolean_coordinate", "coordinate_invalid"),
        ("invalid_latitude", "coordinate_invalid"),
        ("access_coordinate", "access_point_unbound"),
        ("access_source", "access_point_unbound"),
        ("fallback", "unvalidated_or_unbound_road_leg"),
        ("provider", "unvalidated_or_unbound_road_leg"),
        ("geometry_source", "unvalidated_or_unbound_road_leg"),
        ("nan_distance", "unvalidated_or_unbound_road_leg"),
        ("boolean_duration", "unvalidated_or_unbound_road_leg"),
        ("zero_duration", "unvalidated_or_unbound_road_leg"),
        ("short_query_hash", "unvalidated_or_unbound_road_leg"),
        ("extra_leg_ref", "unvalidated_or_unbound_road_leg"),
        ("wrong_junction", "road_geometry_or_provenance_disconnected"),
        ("wrong_operation_refs", "road_geometry_or_provenance_disconnected"),
        ("reversed_access_refs", "road_geometry_or_provenance_disconnected"),
        ("reverse_chain", "route_chain_not_same_insertion"),
        ("map_parent_changed", "confirmation_changed_parent_or_draft"),
        ("proposal_created", "confirmation_changed_parent_or_draft"),
        ("empty_draft", "confirmation_changed_parent_or_draft"),
        ("false_evaluation", "draft_scope_or_io_mismatch"),
        ("wrong_selector", "draft_scope_or_io_mismatch"),
        ("wrong_name", "name_semantics_changed"),
    ],
)
def test_coherently_changed_road_and_form_evidence_is_rejected(tmp_path, fault, code):
    case = _case()
    verifier.confirmation(*case)
    trace, request, result, snap_request, snap, _, _, _ = case
    params = result["operation"]["parameters"]
    leg = params["affected_route_legs"][0]
    if fault.startswith("request_"):
        key = fault.removeprefix("request_")
        request["expected_revision" if key == "revision" else key] = 6 if key == "revision" else "wrong"
    elif fault == "snap_revision":
        snap["session_revision"] = 6
    elif fault == "result_revision":
        result["session"]["revision"] = 9
    elif fault == "boolean_revision":
        snap_request["expected_revision"] = True
    elif fault == "not_confirmed":
        result["confirmed"] = False
    elif fault == "schema":
        params["schema_version"] = "different"
    elif fault == "extra_parameter":
        params["evaluated"] = True
    elif fault == "waypoint_id":
        params["waypoint_id"] = "waypoint_" + "9" * 32
    elif fault == "malformed_snap_id":
        params["snap_preview_id"] = "snap_wrong"
    elif fault == "insertion_origin":
        params["insertion"]["predecessor_id"] = "other"
    elif fault == "insertion_mode":
        params["insertion"]["travel_mode"] = "walking"
    elif fault == "extra_add_target":
        snap_request["target_waypoint_id"] = result["operation"]["target"]
    elif fault == "unacknowledged_warning":
        snap.update(confirmation_required=True, snap_distance_m=150)
    elif fault == "oversize_snap":
        snap["snap_distance_m"] = 501
    elif fault == "negative_snap":
        snap["snap_distance_m"] = -1
    elif fault == "boolean_coordinate":
        params["snapped_coordinate"]["latitude"] = True
    elif fault == "invalid_latitude":
        params["snapped_coordinate"]["latitude"] = 100
    elif fault == "access_coordinate":
        params["selected_access_point"]["coordinate"] = {"longitude": 0, "latitude": 0}
    elif fault == "access_source":
        params["selected_access_point"]["source"] = "guessed"
    elif fault == "fallback":
        leg["fallback_used"] = True
    elif fault == "provider":
        leg["provider"] = "invented"
    elif fault == "geometry_source":
        leg["geometry_source"] = "straight_line"
    elif fault == "nan_distance":
        leg["distance_m"] = float("nan")
    elif fault == "boolean_duration":
        leg["duration_s"] = True
    elif fault == "zero_duration":
        leg["duration_s"] = 0
    elif fault == "short_query_hash":
        leg["query_hash"] = "3" * 16
    elif fault == "extra_leg_ref":
        leg["evidence_refs"].append("unbound")
    elif fault == "wrong_junction":
        leg["geometry"]["coordinates"][-1] = [-120, 35]
    elif fault == "wrong_operation_refs":
        result["operation"]["evidence_refs"] = ["wrong"]
    elif fault == "reversed_access_refs":
        params["selected_access_point"]["evidence_refs"] = list(
            reversed(params["selected_access_point"]["evidence_refs"])
        )
    elif fault == "reverse_chain":
        leg["origin_id"], leg["destination_id"] = leg["destination_id"], leg["origin_id"]
    elif fault == "map_parent_changed":
        result["session"]["accepted_plan_id"] = "child"
    elif fault == "proposal_created":
        result["session"]["proposal"] = {"state": "eligible"}
    elif fault == "empty_draft":
        result["session"]["draft"] = []
    elif fault == "false_evaluation":
        result["evaluated_repair"] = True
    elif fault == "wrong_selector":
        trace["selector"] = "fabricated"
    elif fault == "wrong_name":
        params["name"] = "other"
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_contexts_" + code) as caught:
        verifier.confirmation(*case)
    _write_io(tmp_path, {"fault": fault, "case": case}, {"rejected": str(caught.value)})


def test_warning_requires_explicit_matching_confirmation():
    case = _case()
    case[4].update(confirmation_required=True, snap_distance_m=150)
    case[1]["warning_acknowledged"] = True
    verifier.confirmation(*case)


@pytest.mark.parametrize("fault", ["target", "duration", "request_metadata"])
def test_relocation_never_changes_visit_metadata_or_targets_another_waypoint(fault):
    case = _case(KINDS[2], "accepted_child")
    verifier.confirmation(*case)
    if fault == "target":
        case[3]["target_waypoint_id"] = "other"
    elif fault == "duration":
        case[-1][0]["parameters"]["duration"]["preferred_minutes"] = 60
    else:
        case[1]["name"] = case[2]["operation"]["parameters"]["name"]
    with pytest.raises(
        core.E2E1VerificationError,
        match="insertion_or_relocation_target_unbound|relocation_changed_visit_metadata|form_semantics_not_canonical",
    ):
        verifier.confirmation(*case)


def _record(path, request, response, phase="add_route_waypoint"):
    return {
        "url": BASE + "/api/sessions/session" + path,
        "method": "POST",
        "phase": phase,
        "request_body_utf8": json.dumps(request),
        "response_body_utf8": json.dumps(response),
    }


def _mutations(context):
    result, revision = [], 2
    actions = [(KINDS[0], action) for action in ("snap", "confirm", "undo")]
    actions += [
        (KINDS[1], "snap"),
        (KINDS[1], "confirm"),
        (KINDS[2], "snap"),
        (KINDS[2], "confirm"),
        (KINDS[2], "undo"),
        (KINDS[1], "undo"),
    ]
    for kind, action in actions:
        request = {"expected_revision": revision}
        if action == "snap":
            path, response = "/map/snap-preview", {"operation_intent": kind, "session_revision": revision}
        else:
            path = "/map/snap-previews/snap_" + "2" * 32 + "/confirm" if action == "confirm" else "/draft/undo"
            revision += 1
            response = {
                "operation" if action == "confirm" else "undone": {"type": kind},
                "session": {"session_id": "session", "revision": revision},
            }
        result.append(_record(path, request, response, kind))
    return result


@pytest.mark.parametrize("context", verifier.CONTEXTS)
def test_http_mutations_preserve_real_add_relocate_reverse_undo_order(context):
    verifier.mutation_order(_mutations(context), "session", context)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("reorder", "waypoint_operation_order_invalid"),
        ("duplicate", "waypoint_operation_order_invalid"),
        ("missing", "waypoint_operation_order_invalid"),
        ("other_endpoint_session", "unexpected_mutation_or_session"),
        ("other_response_session", "mutation_revision_or_session_disconnected"),
        ("stale_revision", "waypoint_revision_regressed"),
        ("wrong_snap_revision", "snap_revision_disconnected"),
        ("wrong_undo_revision", "mutation_revision_or_session_disconnected"),
        ("extra_undo_input", "undo_request_invalid"),
        ("wrong_phase", "waypoint_phase_disconnected"),
        ("chat", "unexpected_mutation_or_session"),
        ("delete", "unexpected_mutation_or_session"),
    ],
)
def test_http_counterexamples_rejected_before_claiming_waypoint_coverage(fault, code):
    rows = _mutations("accepted_child")
    verifier.mutation_order(rows, "session", "accepted_child")
    if fault in {"reorder", "duplicate", "missing"}:
        if fault == "missing":
            rows.pop()
        elif fault == "reorder":
            for index, kind in ((-2, KINDS[1]), (-1, KINDS[2])):
                response = json.loads(rows[index]["response_body_utf8"])
                response["undone"]["type"] = kind
                rows[index]["response_body_utf8"] = json.dumps(response)
        else:
            duplicate = deepcopy(rows[-1])
            response = json.loads(rows[-1]["response_body_utf8"])
            duplicate["request_body_utf8"] = json.dumps({"expected_revision": response["session"]["revision"]})
            response["session"]["revision"] += 1
            duplicate["response_body_utf8"] = json.dumps(response)
            rows.append(duplicate)
    elif fault == "other_endpoint_session":
        rows[0]["url"] = rows[0]["url"].replace("/session/", "/other/")
    elif fault in {"other_response_session", "wrong_undo_revision"}:
        response = json.loads(rows[-1]["response_body_utf8"])
        response["session"]["session_id" if fault == "other_response_session" else "revision"] = (
            "other" if fault == "other_response_session" else 99
        )
        rows[-1]["response_body_utf8"] = json.dumps(response)
    elif fault == "stale_revision":
        rows[-1]["request_body_utf8"] = json.dumps({"expected_revision": 1})
    elif fault == "wrong_snap_revision":
        rows[0]["response_body_utf8"] = json.dumps({"operation_intent": KINDS[0], "session_revision": 0})
    elif fault == "extra_undo_input":
        request = json.loads(rows[-1]["request_body_utf8"])
        rows[-1]["request_body_utf8"] = json.dumps({**request, "target": "other"})
    elif fault == "wrong_phase":
        rows[0]["phase"] = "wrong"
    elif fault == "chat":
        rows.append(_record("/copilot/messages", {}, {}))
    elif fault == "delete":
        rows.append({**_record("/conversation", {}, {}), "method": "DELETE"})
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_contexts_" + code):
        verifier.mutation_order(rows, "session", "accepted_child")


def _click():
    case = _case()
    return {
        "action": "native_canvas_click",
        "operation": KINDS[0],
        "route_leg": case[6],
        "raw_coordinate": [-121.94165, 36.51710],
        "pixels": {"x": 200, "y": 300, "zoom": 13, "canvas": {"x": 100, "y": 200, "width": 400, "height": 400}},
        "trusted_dom_event": {"trusted": True, "x": 200, "y": 300, "target": "geographic-map canvas"},
    }, case


def test_real_canvas_pixel_input_matches_exact_snap_request():
    row, case = _click()
    verifier.native_click(row, KINDS[0], case[6], case[3])


@pytest.mark.parametrize(
    "fault", ["synthetic", "mismatched_pixel", "outside_canvas", "wrong_coordinate", "nonfinite_pixel", "wrong_route"]
)
def test_native_canvas_click_cannot_be_replaced_by_claimed_success(fault):
    row, case = _click()
    if fault == "synthetic":
        row["trusted_dom_event"]["trusted"] = False
    elif fault == "mismatched_pixel":
        row["trusted_dom_event"]["x"] += 2
    elif fault == "outside_canvas":
        row["pixels"]["x"] = row["trusted_dom_event"]["x"] = 600
    elif fault == "wrong_coordinate":
        case[3]["longitude"] = -120
    elif fault == "nonfinite_pixel":
        row["pixels"]["zoom"] = float("nan")
    else:
        row["route_leg"] = {**case[6], "route_leg_id": "other"}
    with pytest.raises(core.E2E1VerificationError, match="native_click_|raw_click_not_snap_request"):
        verifier.native_click(row, KINDS[0], case[6], case[3])


@pytest.mark.parametrize("fault", [None, "plan", "hash", "session", "capabilities", "draft", "proposal"])
def test_runtime_capabilities_must_match_a_real_parent_workspace(fault):
    parent = _case()[5]
    snapshot = {
        "schema_version": "runtime-capability-snapshot-v1",
        "session_id": "session",
        "plan_id": "parent",
        "plan_content_hash": "parent_hash",
        "capabilities": {"map_edit_capabilities": {"operations": {KINDS[0]: {"enabled": True}}}},
    }
    view = {
        "session": {"session_id": "session", "accepted_plan_id": "parent", "draft": [], "proposal": None},
        "workspace": {"map_edit_capabilities": deepcopy(snapshot["capabilities"]["map_edit_capabilities"])},
    }
    verifier.capability_binding(snapshot, parent, [view])
    if fault is None:
        return
    if fault in {"plan", "hash", "session"}:
        snapshot[{"plan": "plan_id", "hash": "plan_content_hash", "session": "session_id"}[fault]] = "wrong"
    elif fault == "capabilities":
        snapshot["capabilities"]["map_edit_capabilities"]["operations"][KINDS[0]]["enabled"] = False
    else:
        view["session"][fault] = [{"operation": "unstaged"}] if fault == "draft" else {"state": "eligible"}
    with pytest.raises(
        core.E2E1VerificationError, match="capability_context_mismatch|capability_snapshot_not_http_bound"
    ):
        verifier.capability_binding(snapshot, parent, [view])


def test_missing_full_browser_bundle_is_not_a_pass(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_waypoint_contexts_evidence_malformed"):
        verifier.verify(tmp_path, BASE)


def _baseline():
    """Minimal canonical artifacts; not a replacement for a browser journey."""
    child = {
        "plan_id": "child",
        "parent_plan_id": "parent",
        "planning_request_id": "request",
        "sequence": [verifier.PFEIFFER],
    }
    child["content_hash"] = core._artifact_content_hash(child)
    geography = {
        "plan_id": "child",
        "content_hash": child["content_hash"],
        "stops": {"features": [{"properties": {"stop_id": verifier.PFEIFFER}}]},
        "validated_legs": {"type": "FeatureCollection", "features": []},
    }
    parent = {
        "active_plan_id": "child",
        "active_plan_content_hash": child["content_hash"],
        "rendered_map_plan_id": "child",
        "rendered_geometry_hash": core._completed_race_geometry_hash(geography),
        "rendered_marker_ids": [verifier.PFEIFFER],
    }
    certificate = {
        "plan_id": "child",
        "plan_content_hash": child["content_hash"],
        "certificate_id": "certificate",
        "comparison_eligibility": "eligible",
        "failures": [],
    }
    certificate["content_hash"] = core._artifact_content_hash(certificate)
    diff = {"parent_plan_id": "parent", "child_plan_id": "child", "diff_id": "diff"}
    operation = {
        "type": "replace_nearby",
        "target": verifier.BIXBY,
        "parameters": {"candidate_id": verifier.PFEIFFER},
        "source": "map",
        "created_at": STAMP,
        "operation_id": "replacement",
    }
    request = {key: operation[key] for key in ("type", "target", "parameters", "source")}
    request["expected_revision"] = 1
    staged = {"operation": operation, "session": {"revision": 2, "draft": [operation], "accepted_plan_id": "parent"}}
    proposal = {
        "state": "eligible",
        "execution_mode": "deterministic_repair_pipeline",
        "child_plan_id": "child",
        "child_plan_content_hash": child["content_hash"],
        "parent_plan_id": "parent",
        "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"],
        "diff_id": "diff",
        "diff_content_hash": core._artifact_content_hash(diff),
        "geography_plan": geography,
        "compiled_request": {"operations": [{key: value for key, value in operation.items() if key != "created_at"}]},
        "runtime_execution": {
            "evaluation_executed": True,
            "evaluator": "PlanEvaluator.evaluate_final_plan",
            "request_id": "request",
            "repair_request": {"request_id": "request", "candidate_pois": [{"stop_id": verifier.PFEIFFER}]},
            "child_plan": child,
            "evaluation_certificate": certificate,
            "plan_diff": diff,
        },
    }
    preview = {"proposal": proposal, "session": {"revision": 3, "draft": [operation], "proposal": proposal}}
    accept_request = {
        "confirmation": "accept_evaluated_repair",
        "expected_revision": 3,
        "idempotency_key": "accept-once",
    }
    decision = {
        "decision_id": "decision",
        "decision_type": "accept_repair",
        "idempotency_key": "accept-once",
        "previous_plan_id": "parent",
        "selected_plan_id": "child",
        "selected_plan_content_hash": child["content_hash"],
        "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"],
        "diff_id": "diff",
        "diff_content_hash": proposal["diff_content_hash"],
    }
    pointer = {
        "schema_version": "workspace-pointer-v1",
        "accepted_plan_id": "child",
        "accepted_plan_content_hash": child["content_hash"],
        "original_plan_id": "parent",
        "decision_id": "decision",
        "revision": 1,
    }
    for record in (decision, pointer):
        record["content_hash"] = core._canonical_sha256(record)
    session = {"accepted_plan_id": "child", "draft": [], "proposal": None}
    accepted = {"result": {"decision": decision, "pointer": pointer}, "session": deepcopy(session)}
    reload = {
        "session": session,
        "workspace": {
            "accepted_plan_pointer": {
                key: pointer[key]
                for key in (
                    "schema_version",
                    "accepted_plan_id",
                    "accepted_plan_content_hash",
                    "decision_id",
                    "revision",
                )
            }
        },
    }
    records = [
        _record("/draft/operations", request, staged),
        _record("/preview", {"expected_revision": 2}, preview),
        _record("/accept", accept_request, accepted),
    ]
    for row in records:
        row["phase"] = "baseline_replace_preview_accept"
    return records, parent, reload


def test_accepted_context_starts_with_exact_real_replacement_artifact_lineage(tmp_path):
    case = _baseline()
    result = verifier.accepted_baseline(*case)
    _write_io(tmp_path, case, {"verified": True, "preview": result[0], "acceptance": result[1]})


@pytest.mark.parametrize(
    "fault,code",
    [
        ("wrong_operation", "baseline_replacement_disconnected"),
        ("wrong_compiled_operation", "baseline_replacement_disconnected"),
        ("not_evaluated", "e2e1_self_preview_not_eligible"),
        ("wrong_evaluator", "baseline_evaluator_or_marker_mismatch"),
        ("wrong_marker", "e2e1_journey_render_identity_invalid"),
        ("no_confirmation", "accepted_baseline_unbound"),
        ("decision_hash", "baseline_acceptance_hash_invalid"),
        ("rehash_wrong_child", "accepted_baseline_unbound"),
        ("pointer_hash", "baseline_acceptance_hash_invalid"),
        ("wrong_reload", "accepted_baseline_unbound"),
        ("wrong_reload_pointer", "accepted_baseline_unbound"),
        ("duplicate_preview", "baseline_order_invalid"),
        ("missing_acceptance", "baseline_order_invalid"),
        ("out_of_order", "baseline_order_invalid"),
    ],
)
def test_accepted_context_cannot_be_claimed_from_staging_or_unrelated_artifacts(tmp_path, fault, code):
    case = _baseline()
    verifier.accepted_baseline(*case)
    records, parent, reload = case
    index = (
        0
        if fault == "wrong_operation"
        else 1
        if fault in {"wrong_compiled_operation", "not_evaluated", "wrong_evaluator"}
        else 2
    )
    payload = json.loads(records[index]["response_body_utf8"])
    if fault == "wrong_operation":
        payload["operation"]["type"] = "add_candidate"
    elif fault == "wrong_compiled_operation":
        payload["proposal"]["compiled_request"]["operations"] = []
    elif fault == "not_evaluated":
        payload["proposal"]["state"] = "ineligible"
    elif fault == "wrong_evaluator":
        payload["proposal"]["runtime_execution"]["evaluator"] = "planner_self_check"
    elif fault == "wrong_marker":
        parent["rendered_marker_ids"] = [verifier.BIXBY]
    elif fault == "no_confirmation":
        request = json.loads(records[2]["request_body_utf8"])
        records[2]["request_body_utf8"] = json.dumps({**request, "confirmation": "auto"})
    elif fault in {"decision_hash", "pointer_hash", "rehash_wrong_child"}:
        record = payload["result"]["pointer" if fault == "pointer_hash" else "decision"]
        if fault == "rehash_wrong_child":
            record["selected_plan_id"] = "other_child"
            record["content_hash"] = core._canonical_sha256(
                {key: value for key, value in record.items() if key != "content_hash"}
            )
        else:
            record["content_hash"] = "incorrect"
    elif fault == "wrong_reload":
        reload["session"]["accepted_plan_id"] = "parent"
    elif fault == "wrong_reload_pointer":
        reload["workspace"]["accepted_plan_pointer"]["decision_id"] = "older_decision"
    if index == 1:
        # Keep both public projections coherent so the intended semantic gate,
        # not an unrelated stale projection, rejects this counterexample.
        payload["session"]["proposal"] = deepcopy(payload["proposal"])
    records[index]["response_body_utf8"] = json.dumps(payload)
    if fault == "duplicate_preview":
        records.insert(2, deepcopy(records[1]))
    elif fault == "missing_acceptance":
        records.pop()
    elif fault == "out_of_order":
        records.reverse()
    with pytest.raises(core.E2E1VerificationError, match=code) as caught:
        verifier.accepted_baseline(*case)
    _write_io(tmp_path, {"fault": fault, "case": case}, {"rejected": str(caught.value)})
