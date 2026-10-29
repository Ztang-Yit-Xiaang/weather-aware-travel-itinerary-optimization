"""Prospective synthetic counterexamples; not browser acceptance evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as verifier  # noqa: E402


def visibility_case():
    rect = {"left": 20, "top": 20, "right": 120, "bottom": 60, "width": 100, "height": 40}
    ancestor = {"rect": {"left": 0, "top": 0, "right": 390, "bottom": 844, "width": 390, "height": 844},
        "overflow_x": "hidden", "overflow_y": "auto", "client_left": 0, "client_top": 0,
        "client_width": 390, "client_height": 844, "scroll_top": 0, "scroll_left": 0,
        "display": "block", "visibility": "visible", "opacity": 1}
    texts = {"#inline-undo": "Undo latest draft", "#inline-preview": "Preview repair",
             ".textual-map-alternative > summary": "Text route description",
             ".text-route-summary > h3:first-child": "Day 5 route",
             ".text-route-summary dd:first-of-type": "1. Bixby Creek Bridge"}
    rows = []
    for selector, text in texts.items():
        rows.append({"selector": selector, "present": True, "text": text, "disabled": True if selector.startswith("#") else None,
            "rect": deepcopy(rect), "clip": {key: rect[key] for key in ("left", "top", "right", "bottom")}, "visible_ratio": 1,
            "display": "block", "visibility": "visible", "opacity": 1, "ancestors": [deepcopy(ancestor)],
            "points": [{"x": 20 + 100 * dx, "y": 20 + 40 * dy, "unoccluded": True,
                        "hit": {"tag": "BUTTON" if selector.startswith("#") else "H3", "id": selector[1:] if selector.startswith("#") else ""}}
                       for dx, dy in ((.5, .5), (.1, .1), (.9, .1), (.1, .9), (.9, .9))]})
    value = {"at": "2026-09-08T10:00:03+00:00", "viewport": {"width": 390, "height": 844}, "elements": rows,
        "scroll_owners": [{"tag": "HTML", "class_name": "", "scroll_top": 0, "scroll_left": 0, "client_height": 844, "scroll_height": 844}]}
    checkpoint = {"captured_at": "2026-09-08T10:00:02+00:00", "readiness": {"viewport": value["viewport"]}}
    return value, checkpoint, "2026-09-08T10:00:04+00:00", texts


@pytest.mark.parametrize("fault", [None, "missing", "hidden", "transparent", "ancestor_hidden", "clipped", "fake_ratio",
    "offscreen", "occluded", "missing_hit", "wrong_point", "nan", "wrong_heading", "wrong_order", "enabled",
    "different_control", "stale_capture", "wrong_viewport", "missing_scroll"])
def test_workspace_controls_require_visible_canonical_text_and_real_hit_testing(fault):
    value, checkpoint, completed, texts = visibility_case()
    row = value["elements"][0]
    if fault == "missing":
        value["elements"].pop()
    elif fault == "hidden":
        row["visibility"] = "hidden"
    elif fault == "transparent":
        row["opacity"] = 0
    elif fault == "ancestor_hidden":
        row["ancestors"][0]["display"] = "none"
    elif fault == "clipped":
        row["ancestors"][0]["client_height"] = 25
    elif fault == "fake_ratio":
        row["visible_ratio"] = .99
    elif fault == "offscreen":
        row["rect"]["left"] = -100
    elif fault == "occluded":
        row["points"][0]["unoccluded"] = False
    elif fault == "missing_hit":
        row["points"][0]["hit"] = None
    elif fault == "wrong_point":
        row["points"][0]["x"] = 999
    elif fault == "nan":
        row["rect"]["width"] = float("nan")
    elif fault == "wrong_heading":
        value["elements"][3]["text"] = "Day 1 route"
    elif fault == "wrong_order":
        value["elements"][4]["text"] = "Pfeiffer"
    elif fault == "enabled":
        row["disabled"] = False
    elif fault == "different_control":
        row["points"][0]["hit"]["id"] = "close-copilot"
    elif fault == "stale_capture":
        value["at"] = "2026-09-08T09:00:00+00:00"
    elif fault == "wrong_viewport":
        value["viewport"] = {"width": 1440, "height": 900}
    elif fault == "missing_scroll":
        value["scroll_owners"] = []
    if fault:
        with pytest.raises(verifier.E2E1VerificationError):
            verifier._verify_workspace_visibility(value, checkpoint, completed, texts)
    else:
        verifier._verify_workspace_visibility(value, checkpoint, completed, texts)


def wire(payload, url, request):
    row = {"schema_version": "e2e1-product-http-io-v1", "method": "POST", "url": url, "response_status_code": 200,
        "body_capture": "exact_utf8", "credential_redactions": {"request": [], "response": []},
        "request_headers_persisted": False, "response_headers_persisted": False, "authorization_persisted": False,
        "session_token_persisted": False}
    for side, data in (("request", request), ("response", payload)):
        body = json.dumps(data)
        row[side + "_body_utf8"] = body
        row[side + "_body_sha256"] = row[side + "_original_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return row


def selection_case():
    initial = {"session_id": "s", "trip_id": "trip", "run_id": "run", "revision": 0, "accepted_plan_id": "parent", "draft": []}
    created, plan = {"session": initial}, {"plan_id": "parent", "content_hash": "a" * 16}
    fields = {"selected_stop_id": "bixby_creek_bridge_viewpoint", "selected_day": 5, "selected_segment_id": None}
    selection = wire({"session": {**initial, **fields, "revision": 1}}, "http://127.0.0.1:8127/api/sessions/s/selection",
                     {"expected_revision": 0, **fields})
    impact = wire({"schema_version": "product-draft-impact-preview-v1", "session_id": "s", "session_revision": 1,
        "parent_plan_id": "parent", "parent_plan_content_hash": plan["content_hash"], "operations": [], "is_certified": False,
        "summary": {"operation_count": 0, "can_run_evaluated_preview": False}},
        "http://127.0.0.1:8127/api/sessions/s/draft/impact-preview", {"expected_revision": 1})
    records = [{"url": "http://127.0.0.1:8127/api/sessions", "response_body_utf8": json.dumps(created)},
        {**selection, "observed_at": "2026-09-08T10:00:01+00:00"}, {**impact, "observed_at": "2026-09-08T10:00:02+00:00"}]
    checkpoint = {"selected_stop": fields["selected_stop_id"], "selected_day": "5",
        "text_scope": {"planId": "parent", "planContentHash": plan["content_hash"]},
        "readiness": {"screenshot_started_at": "2026-09-08T10:00:03+00:00"}}
    return selection, impact, records, created, plan, checkpoint, "2026-09-08T10:00:00+00:00"


@pytest.mark.parametrize("fault", [None, "not_captured", "duplicate", "out_of_order", "late_impact", "wrong_revision",
    "foreign_owner", "changed_plan", "staged_draft", "wrong_stop", "wrong_day", "wrong_hash", "certified",
    "executable", "stale_dom", "stale_text", "selection_between"])
def test_text_route_capture_requires_exact_completed_selection_and_empty_impact(fault):
    selection, impact, records, created, plan, checkpoint, before = selection_case()
    if fault in {"wrong_revision", "foreign_owner", "changed_plan", "staged_draft", "wrong_stop"}:
        payload = json.loads(selection["response_body_utf8"])
        key, value = {"wrong_revision": ("revision", 2), "foreign_owner": ("trip_id", "foreign"),
            "changed_plan": ("accepted_plan_id", "child"), "staged_draft": ("draft", [{}]), "wrong_stop": ("selected_stop_id", "pfeiffer")}[fault]
        payload["session"][key] = value
        selection = wire(payload, selection["url"], json.loads(selection["request_body_utf8"]))
        records[1].update(selection)
    elif fault in {"wrong_hash", "certified", "executable"}:
        payload = json.loads(impact["response_body_utf8"])
        if fault == "wrong_hash":
            payload["parent_plan_content_hash"] = "foreign"
        elif fault == "certified":
            payload["is_certified"] = True
        else:
            payload["summary"]["can_run_evaluated_preview"] = True
        impact = wire(payload, impact["url"], json.loads(impact["request_body_utf8"]))
        records[2].update(impact)
    elif fault == "not_captured":
        records.pop()
    elif fault == "duplicate":
        records.append(deepcopy(records[1]))
    elif fault == "out_of_order":
        records[1:] = reversed(records[1:])
    elif fault == "late_impact":
        records[2]["observed_at"] = "2026-09-08T11:00:00+00:00"
    elif fault == "wrong_day":
        checkpoint["selected_day"] = "1"
    elif fault == "stale_dom":
        checkpoint["selected_stop"] = "pfeiffer"
    elif fault == "stale_text":
        checkpoint["text_scope"]["planContentHash"] = "old"
    elif fault == "selection_between":
        records.insert(2, {"url": selection["url"]})
    if fault:
        with pytest.raises(verifier.E2E1VerificationError):
            verifier._verify_workspace_text_selection(selection, impact, records, created, plan, checkpoint, before)
    else:
        verifier._verify_workspace_text_selection(selection, impact, records, created, plan, checkpoint, before)


def metric_scroll_case(axis="y"):
    visible, checkpoint, _, _ = visibility_case()
    plan = {"plan_id":"parent","content_hash":"a" * 16}
    scope = {"planId":plan["plan_id"],"planContentHash":plan["content_hash"],"evaluationId":""}
    metrics = [{"scope":{key:scope[key] for key in ("planId","planContentHash")},
        "label":label,"value":"Unavailable","note":"Exact displayed-plan evidence"}
        for label in ("Route time","Contextual risk","Preservation","Edit cost","Certificate")]
    def at(second):
        return f"2026-09-08T10:00:{second:02d}+00:00"
    snapshot = {"at":at(3),"scope":scope,"metrics":metrics,"scroll_top":0,"scroll_left":0,
        "client_height":60,"client_width":200,"scroll_height":100 if "y" in axis else 60,
        "scroll_width":600 if "x" in axis else 200,"overflow_x":"auto","overflow_y":"auto",
        "page_scroll":{"top":0,"left":0},"workspace_scroll":{"top":0,"left":0}}
    initial = {"rendered_map_plan_id":"parent","rendered_geometry_hash":"geometry"}
    record = {"schema_version":"workspace-metric-scroll-v1","setup":"locator_scroll_into_view_if_needed",
        "restoration":"native_reverse_wheel_then_Control+Home","map_before":initial,"map_after":deepcopy(initial),
        "before":deepcopy(snapshot),"onscreen":{**deepcopy(snapshot),"at":at(4)},
        "end":{**deepcopy(snapshot),"at":at(6),"scroll_top":snapshot["scroll_height"]-60,
            "scroll_left":snapshot["scroll_width"]-200},
        "restored":{**deepcopy(snapshot),"at":at(9)},"completed_at":at(10),
        "overflow_axes":{"x":"x" in axis,"y":"y" in axis},"commands":[],"events":[]}
    if axis:
        dx, dy = snapshot["scroll_width"] if "x" in axis else 0, snapshot["scroll_height"] if "y" in axis else 0
        for action, sign, time in (("wheel_to_end",1,5),("wheel_restore",-1,8)):
            record["commands"].append({"action":action,"delta_x":sign*dx,"delta_y":sign*dy,"at":at(time)})
            record["events"].append({"type":"wheel","delta_x":sign*dx,"delta_y":sign*dy,"trusted":True,"at":at(time)})
            record["events"].append({"type":"scroll","trusted":True,"at":at(time),
                "scroll_top":record["end"]["scroll_top"] if sign==1 else 0,
                "scroll_left":record["end"]["scroll_left"] if sign==1 else 0})
    record["commands"].append({"action":"keyboard_home","key":"Control+Home","at":at(8)})
    record["events"].append({"type":"keydown","trusted":True,"key":"Home","ctrl_key":True,"at":at(8)})
    detail = deepcopy(visible["elements"][0])
    detail.update(selector="#metric-strip .metric:last-child small",text=metrics[-1]["note"],disabled=None)
    for point in detail["points"]:
        point["hit"] = {"tag":"SMALL","id":"","class_name":""}
    visible.update(at=at(7),elements=[detail])
    record["detail_visibility"] = visible
    checkpoint.update(metric_scope=scope,scroll_owners=[{"class_name":"metric-strip",
        **{key:snapshot[key] for key in ("client_height","client_width","scroll_height","scroll_width")}}])
    return record, checkpoint, initial, plan, at(11)


@pytest.mark.parametrize("axis", ["", "x", "y", "xy"])
def test_footer_reachability_accepts_native_scroll_or_truthful_no_overflow(axis):
    verifier._verify_workspace_metric_scroll(*metric_scroll_case(axis))


@pytest.mark.parametrize("fault", ["missing_owner","changed_owner_extent","fake_no_overflow","not_at_end",
    "not_restored","page_not_restored","wrong_parent","changed_metrics","missing_metric","no_native_wheel",
    "synthetic_wheel","no_scroll_event","no_home","wrong_wheel_delta","late_event","late_capture",
    "covered_detail","wrong_detail","nan_extent","stale_map","only_programmatic_scroll"])
def test_footer_reachability_rejects_fabricated_or_incomplete_proof(fault):
    record, checkpoint, initial, plan, completed = metric_scroll_case()
    if fault == "missing_owner":
        checkpoint["scroll_owners"] = []
    elif fault == "changed_owner_extent":
        checkpoint["scroll_owners"][0]["client_height"] = 200
    elif fault == "fake_no_overflow":
        record["overflow_axes"]["y"] = False
    elif fault == "not_at_end":
        record["end"]["scroll_top"] = 0
    elif fault == "not_restored":
        record["restored"]["scroll_top"] = 10
    elif fault == "page_not_restored":
        record["restored"]["page_scroll"]["top"] = 200
    elif fault == "wrong_parent":
        record["end"]["scope"]["planId"] = "foreign"
    elif fault == "changed_metrics":
        record["end"]["metrics"][0]["value"] = "invented"
    elif fault == "missing_metric":
        record["before"]["metrics"].pop()
    elif fault == "no_native_wheel":
        record["events"] = [row for row in record["events"] if row["type"] != "wheel"]
    elif fault == "synthetic_wheel":
        record["events"][0]["trusted"] = False
    elif fault == "no_scroll_event":
        record["events"] = [row for row in record["events"] if row["type"] != "scroll"]
    elif fault == "no_home":
        record["events"] = [row for row in record["events"] if row["type"] != "keydown"]
    elif fault == "wrong_wheel_delta":
        record["events"][0]["delta_y"] = 1
    elif fault == "late_event":
        record["events"][0]["at"] = "2026-09-08T10:00:30+00:00"
    elif fault == "late_capture":
        record["detail_visibility"]["at"] = "2026-09-08T10:00:30+00:00"
    elif fault == "covered_detail":
        record["detail_visibility"]["elements"][0]["points"][0]["unoccluded"] = False
    elif fault == "wrong_detail":
        record["detail_visibility"]["elements"][0]["text"] = "other detail"
    elif fault == "nan_extent":
        record["end"]["client_height"] = float("nan")
    elif fault == "stale_map":
        record["map_after"]["rendered_map_plan_id"] = "foreign"
    elif fault == "only_programmatic_scroll":
        record["restoration"] = "assign_scrollTop"
    with pytest.raises(verifier.E2E1VerificationError):
        verifier._verify_workspace_metric_scroll(record, checkpoint, initial, plan, completed)
