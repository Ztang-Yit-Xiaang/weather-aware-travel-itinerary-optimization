"""Bind actual POI filters/layers/close to real HTTP and public renderer sources."""

from __future__ import annotations

import json
from collections import Counter
from urllib.parse import parse_qs, urlsplit

import verify_e2e1_vertical_slice as core

SOURCES = ("v2-poi-candidates", "v2-poi-access-points")
SELECTORS = (
    "#poi-candidate-filter",
    "#poi-candidate-filter",
    "#poi-layer-toggle",
    "#poi-layer-toggle",
    "#layers-button",
    "#layers-button",
    "#close-poi-candidates",
    "#find-route-candidates",
)


def require(condition, reason):
    core._require(condition, "e2e1_poi_controls_" + reason)


def features(sources, candidates):
    require(set(sources) == set(SOURCES), "source_inventory_invalid")
    for source in SOURCES:
        data = sources[source]
        require(
            data["type"] == "FeatureCollection" and len(data["features"]) == len(candidates), "source_count_invalid"
        )
        expected = {row["candidate_id"]: row for row in candidates}
        require(
            {row["properties"]["candidate_id"] for row in data["features"]} == set(expected),
            "candidate_identity_invalid",
        )
        for feature in data["features"]:
            candidate = expected[feature["properties"]["candidate_id"]]
            point = (
                candidate["place"]["display_coordinate"]
                if source == SOURCES[0]
                else candidate["selected_access_point"]["coordinate"]
            )
            require(
                feature["geometry"] == {"type": "Point", "coordinates": [point["longitude"], point["latitude"]]}
                and feature["properties"]["name"] == candidate["place"]["name"],
                "source_coordinate_disconnected",
            )


def verify_rows(
    rows, raw, parent, initial_candidates, *, context="original", target="bixby_creek_bridge_viewpoint", day=5
):
    require(
        [row["selector"] for row in rows] == list(SELECTORS)
        and raw["parent"] == parent
        and type(raw["provider_calls"]) is int
        and raw["provider_calls"] == 0,
        "journey_universe_invalid",
    )
    start, end = raw["request_start"], raw["request_end"]
    require(
        type(start) is int
        and type(end) is int
        and 0 <= start <= end == len(raw["requests"])
        and all(row["method"] == "GET" for row in raw["requests"][start:end]),
        "discovery_mutation_detected",
    )
    candidates = initial_candidates
    for index, row in enumerate(rows):
        require(
            row["assertions_passed"] is True
            and row["visible"] is True
            and row["disabled"] is False
            and row["events"]
            and row["events"] == raw["events"][: len(row["events"])]
            and row["map"] == parent,
            "input_or_parent_changed",
        )
        kind = "submit" if index < 2 else "change" if index < 4 else "click"
        events = [event for event in row["events"] if event["type"] == kind]
        require(
            events and events[-1]["trusted"] is True and events[-1]["id"] == row["selector"][1:],
            "real_control_input_missing",
        )
        if index in {0, 1, 7}:
            http = row["http"]
            core._verify_product_http_capture(http)
            require(
                http["method"] == "GET"
                and http["response_status_code"] == 200
                and "/map/poi-candidates?" in http["url"]
                and {"method": "GET", "url": http["url"]} in raw["requests"][start:end],
                "http_unbound",
            )
            response = json.loads(http["response_body_utf8"])
            expected_target = None if index == 7 else target
            expected_kind = "insertion" if index == 7 else "replacement"
            query = parse_qs(urlsplit(http["url"]).query)
            require(
                response["context"]["day"] == day
                and response["context"]["replacement_target_id"] == expected_target
                and response["context"]["kind"] == expected_kind
                and query.get("replacement_target_id", []) == ([] if index == 7 else [target]),
                "route_context_changed",
            )
            if index == 7:
                require(response["candidates"] == [], "generic_reopen_falsely_reused_replacements")
            candidates = response["candidates"]
            if index < 2:
                expected = (
                    {"role": "meal", "max_detour_minutes": "120", "limit": "3"},
                    {"role": "", "max_detour_minutes": "60", "limit": "5"},
                )[index]
                query = parse_qs(urlsplit(http["url"]).query)
                require(
                    row["fields"] == expected
                    and all(query.get(key, [""]) == [value] for key, value in expected.items()),
                    "filter_input_not_sent",
                )
            features(row["after_layers"], candidates)
        elif index in {2, 4}:
            features(row["before_layers"], candidates)
            features(row["after_layers"], [])
        elif index in {3, 5}:
            features(row["before_layers"], [])
            features(row["after_layers"], candidates)
        else:
            features(row["after_layers"], [])
        require(row["drawer_visible"] is (index != 6), "drawer_result_invalid")
    reentry = verify_reentry(
        raw["replacement_reentry"], rows[-1]["http"], raw["events"], parent, target, initial_candidates
    )
    return {
        "verified": True,
        "selectors": sorted(set(SELECTORS)),
        "context": context,
        "provider_calls": 0,
        "replacement_reentry": reentry,
    }


def selection_http(record, parent, revision, expected):
    core._verify_product_http_capture(record)
    request = json.loads(record["request_body_utf8"])
    session = json.loads(record["response_body_utf8"])["session"]
    require(
        record["method"] == "POST"
        and record["response_status_code"] == 200
        and urlsplit(record["url"]).path == "/api/sessions/" + session["session_id"] + "/selection"
        and request["expected_revision"] == revision
        and session["revision"] == revision + 1
        and session["accepted_plan_id"] == parent["active_plan_id"]
        and session["draft"] == []
        and all(request.get(key) == value and session.get(key) == value for key, value in expected.items()),
        "selection_not_exact_or_changed_parent",
    )
    return session


def verify_reentry(value, generic, prior_events, parent, target, initial_candidates):
    require(
        value["schema_version"] == "poi-replacement-reentry-v1"
        and value["assertions_passed"] is True
        and value["target"] == target
        and value["generic_http"] == generic
        and value["map"] == parent
        and prior_events == value["events"][: len(prior_events)],
        "reentry_scope_unbound",
    )
    core._verify_product_http_capture(generic)
    insertion = json.loads(generic["response_body_utf8"])
    leg = insertion["context"]["route_leg_id"]
    require(
        insertion["context"]["kind"] == "insertion"
        and insertion["context"]["replacement_target_id"] is None
        and insertion["candidates"] == []
        and value["route_leg_id"] == leg,
        "reentry_requires_real_empty_insertion",
    )
    stop = selection_http(
        value["stop_http"],
        parent,
        insertion["session_revision"],
        {"selected_day": insertion["context"]["day"], "selected_stop_id": target, "selected_segment_id": None},
    )
    segment = selection_http(
        value["leg_http"],
        parent,
        stop["revision"],
        {"selected_day": insertion["context"]["day"], "selected_stop_id": None, "selected_segment_id": leg},
    )
    require(
        stop["session_id"] == segment["session_id"] == insertion["session_id"]
        and all(stop[key] == segment[key] for key in ("trip_id", "run_id"))
        and all(urlsplit(value[key]["url"])[:2] == urlsplit(generic["url"])[:2] for key in ("stop_http", "leg_http")),
        "reentry_session_owner_changed",
    )
    candidate_http = value["candidate_http"]
    core._verify_product_http_capture(candidate_http)
    payload = json.loads(candidate_http["response_body_utf8"])
    query = parse_qs(urlsplit(candidate_http["url"]).query)
    require(
        candidate_http["method"] == "GET"
        and candidate_http["response_status_code"] == 200
        and urlsplit(candidate_http["url"]).path == "/api/sessions/" + stop["session_id"] + "/map/poi-candidates"
        and urlsplit(candidate_http["url"])[:2] == urlsplit(generic["url"])[:2]
        and query.get("replacement_target_id") == [target]
        and query.get("route_leg_id") == [leg]
        and query.get("expected_revision") == [str(segment["revision"])]
        and payload["session_id"] == segment["session_id"]
        and payload["session_revision"] == segment["revision"]
        and payload["context"]["kind"] == "replacement"
        and payload["context"]["replacement_target_id"] == target
        and payload["context"]["day"] == insertion["context"]["day"]
        and payload["context"]["route_leg_id"] == leg,
        "reentry_candidate_http_not_fresh",
    )
    candidates = payload["candidates"]
    ids = [row["candidate_id"] for row in candidates]
    require(
        ids
        and len(ids) == len(set(ids))
        and set(ids) == {row["candidate_id"] for row in initial_candidates}
        and value["visible_candidate_ids"] == ids,
        "reentry_visible_candidates_not_current",
    )
    previous = len(prior_events)
    for key, field, expected in (
        ("stop_events", "stop", target),
        ("preparation_events", "id", "prepare-replacement-candidates"),
        ("leg_events", "leg", leg),
        ("events", "id", "find-route-candidates"),
    ):
        events = value[key]
        require(
            events == value["events"][: len(events)]
            and len(events) > previous
            and events[-1]["trusted"] is True
            and events[-1]["type"] == "click"
            and events[-1].get(field) == expected,
            "reentry_native_action_missing",
        )
        previous = len(events)
    features(value["layers"], candidates)
    return {"target": target, "route_leg_id": leg, "candidate_ids": ids, "session_revision": segment["revision"]}


def accepted_rows(rows, raw, snapshot, records):
    parent = snapshot["map_ack"]
    target = "pfeiffer_big_sur_state_park"
    leg = raw["selected_leg"]
    require(
        raw["context"] == "accepted_child"
        and raw["target"] == target
        and target in parent["rendered_marker_ids"]
        and leg in [feature["properties"] for feature in snapshot["geography"]["validated_legs"]["features"]]
        and leg["day"] == 5
        and leg["destination_id"] == target,
        "accepted_route_not_canonical",
    )
    discovery = raw["discovery_http"]
    core._verify_product_http_capture(discovery)
    payload = json.loads(discovery["response_body_utf8"])
    require(
        discovery["method"] == "GET"
        and discovery["response_status_code"] == 200
        and payload["context"]["replacement_target_id"] == target
        and payload["context"]["day"] == 5
        and payload["context"]["route_leg_id"] == leg["route_leg_id"]
        and payload["candidates"] == raw["initial_candidates"],
        "accepted_discovery_unbound",
    )
    for key, selector in (
        ("preparation_events", "prepare-replacement-candidates"),
        ("discovery_events", "find-route-candidates"),
    ):
        events = raw[key]
        require(
            events
            and events == raw["events"][: len(events)]
            and events[-1]["type"] == "click"
            and events[-1]["trusted"] is True
            and events[-1]["id"] == selector,
            "accepted_discovery_click_unbound",
        )
    expected_http = [discovery, *[row["http"] for row in rows if row["http"]]]
    actual_http = [
        record
        for record in records
        if record["phase"] == "accepted_poi_controls" and "/map/poi-candidates?" in record["url"]
    ]
    require(len(actual_http) == len(expected_http), "accepted_http_count_unbound")
    for index, (http, record) in enumerate(zip(expected_http, actual_http, strict=True)):
        if index < len(expected_http) - 1:
            require(
                json.loads(http["response_body_utf8"])["context"] == payload["context"], "accepted_filter_route_changed"
            )
        require(all(record.get(key) == value for key, value in http.items()), "accepted_http_not_captured")
    for key in ("stop_http", "leg_http", "candidate_http"):
        http = raw["replacement_reentry"][key]
        require(
            any(
                record.get("phase") == "accepted_poi_reentry"
                and all(record.get(name) == value for name, value in http.items())
                for record in records
            ),
            "accepted_reentry_http_unbound",
        )
    return verify_rows(rows, raw, parent, raw["initial_candidates"], context="accepted_child", target=target)


def verify(base, parent):
    return verify_rows(
        json.loads((base / "poi_control_journeys.json").read_text(encoding="utf-8")),
        core._load(base / "poi_control_input_log.json"),
        parent,
        core._load(base / "browser_poi_candidate_response.json")["candidates"],
    )


def accepted_inspection(value, snapshot, records):
    require(
        value["context"] == "accepted_child"
        and value["parent"] == snapshot["map_ack"]
        and value["assertions_passed"] is True
        and value["accepted_plan_changed"] is False,
        "inspection_scope_invalid",
    )
    http = value["candidate_http"]
    core._verify_product_http_capture(http)
    require(
        any(all(record.get(key) == item for key, item in http.items()) for record in records),
        "inspection_candidates_http_unbound",
    )
    response = json.loads(http["response_body_utf8"])
    require(
        http["method"] == "GET"
        and http["response_status_code"] == 200
        and "/map/poi-candidates?" in http["url"]
        and response["context"]["replacement_target_id"] == "pfeiffer_big_sur_state_park"
        and value["rows"]
        and [row["candidate"] for row in value["rows"]] == response["candidates"],
        "inspection_candidate_universe_invalid",
    )
    payloads = [json.loads(record["response_body_utf8"]) for record in records]
    require(
        any(
            isinstance(payload, dict)
            and payload.get("session", {}).get("accepted_plan_id") == value["parent"]["active_plan_id"]
            and payload.get("workspace", {}).get("draft_capabilities", {}).get("operations", {}).get("replace_nearby")
            == value["capability"]
            for payload in payloads
        ),
        "inspection_capability_unbound",
    )
    observed = value["http_records"]
    for record in observed:
        core._verify_product_http_capture(record)
        require(
            record in records
            and record.get("phase") == "accepted_poi_inspection"
            and record["response_status_code"] == 200,
            "inspection_http_unbound",
        )
    prefix = urlsplit(http["url"]).path.removesuffix("/map/poi-candidates")
    require(
        all(
            row.get("method") == "GET"
            or (
                row.get("method") == "POST"
                and urlsplit(row.get("url", "")).path in {prefix + "/selection", prefix + "/draft/impact-preview"}
            )
            for row in value["requests"]
        ),
        "inspection_mutated",
    )
    require(
        Counter((row["method"], row["url"]) for row in value["requests"] if "/api/" in row.get("url", ""))
        == Counter((row["method"], row["url"]) for row in observed),
        "inspection_request_universe_unbound",
    )
    require(
        sum(row["method"] == "POST" and row["url"].endswith("/selection") for row in observed) == len(value["rows"]),
        "inspection_selection_count_invalid",
    )
    revision = response["session_revision"]
    previous = 0
    for row in value["rows"]:
        candidate = row["candidate"]
        events = row["events"]
        require(
            events == value["events"][: len(events)]
            and len(events) > previous
            and events[-1]["type"] == "click"
            and events[-1]["trusted"] is True
            and events[-1]["candidate"] == candidate["candidate_id"]
            and row["selected"] == "true"
            and row["title"] == candidate["place"]["name"]
            and row["map"] == value["parent"]
            and row["assertions_passed"] is True
            and row["forced_interaction"] is False,
            "inspection_input_unbound",
        )
        registered = candidate.get("registered_replacement") or {}
        selection = row["selection_http"]
        refreshed = row["refresh_http"]
        require(
            all(
                any(all(record.get(key) == item for key, item in receipt.items()) for record in observed)
                for receipt in (selection, refreshed)
            ),
            "inspection_click_http_missing",
        )
        session = selection_http(
            selection, value["parent"], revision, {"selected_candidate_id": candidate["candidate_id"]}
        )
        require(urlsplit(selection["url"])[:2] == urlsplit(http["url"])[:2], "inspection_selection_origin_changed")
        revision = session["revision"]
        core._verify_product_http_capture(refreshed)
        fresh = json.loads(refreshed["response_body_utf8"])
        require(
            refreshed["method"] == "GET"
            and refreshed["response_status_code"] == 200
            and urlsplit(refreshed["url"]).path == prefix + "/map/poi-candidates"
            and urlsplit(refreshed["url"])[:2] == urlsplit(http["url"])[:2]
            and parse_qs(urlsplit(refreshed["url"]).query).get("expected_revision") == [str(revision)]
            and fresh["session_id"] == session["session_id"] == response["session_id"]
            and fresh["session_revision"] == revision
            and fresh["context"] == response["context"]
            and fresh["candidates"] == response["candidates"],
            "inspection_refetch_not_exact",
        )
        enabled = (
            value["capability"]["enabled"]
            and registered.get("draft_type") == "replace_nearby"
            and registered.get("candidate_id") == candidate["candidate_id"] == candidate["place"]["place_id"]
            and registered.get("target_stop_id") == response["context"]["replacement_target_id"]
        )
        control = row["replace_control"]
        require(
            not enabled
            and control["visible"] is True
            and control["disabled"] is True
            and control["title"]
            == "Only an exact server-registered replacement for the selected accepted-plan stop can become a draft",
            "inspection_disabled_guard_unbound",
        )
        previous = len(events)
    for record in observed:
        if record["method"] == "POST" and record["url"].endswith("/draft/impact-preview"):
            request = json.loads(record["request_body_utf8"])
            impact = json.loads(record["response_body_utf8"])
            require(
                impact["schema_version"] == "product-draft-impact-preview-v1"
                and impact["session_id"] == response["session_id"]
                and impact["session_revision"] == request["expected_revision"]
                and response["session_revision"] < impact["session_revision"] <= revision
                and impact["parent_plan_id"] == value["parent"]["active_plan_id"]
                and impact["parent_plan_content_hash"] == value["parent"]["active_plan_content_hash"]
                and impact["operations"] == []
                and impact["is_certified"] is False
                and impact["summary"]["operation_count"] == 0
                and impact["summary"]["can_run_evaluated_preview"] is False,
                "inspection_impact_changed_parent_or_draft",
            )
    return {
        "scope": "Every returned accepted-target candidate inspected through its actual button; replacement independently disabled by exact registration guard",
        "candidate_count": len(value["rows"]),
        "unsupported_claimed": False,
    }
