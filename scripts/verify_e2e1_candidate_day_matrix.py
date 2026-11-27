"""Independent candidate/day route readiness and real staging/Undo evidence gate.

Never calls the compiler or the browser test's readiness helper. The route oracle
is the pinned manifest's anchors plus independently checked directed road cells.
Immutable package/delta loaders only verify source files; they do not plan routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _proposal

SCENARIO = "candidate_day_matrix"
TEST = "test_browser_candidate_day_readiness_in_original_and_accepted_child"
CONTEXTS = ("original", "accepted_child")
BASELINE_PHASE = "establish_accepted_child_through_real_ui"
CONTROL_IDS = {"registered-candidate", "destination-day", "draft-add", "draft-preview"}


def require(condition, reason):
    core._require(condition, "e2e1_candidate_days_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(first, second):
    return all(
        first.get(key) == second.get(key)
        for key in (
            "active_plan_id",
            "active_plan_content_hash",
            "rendered_map_plan_id",
            "rendered_geometry_hash",
            "rendered_marker_ids",
        )
    )


def browser_health(base, base_url, parent):
    """Retain cancellations only with independently bound ready-map evidence."""
    from verify_e2e1_network_observations import verify as verify_network
    from verify_e2e1_network_observations import verify_asset_observation

    errors, llm = load(base / "browser_errors.json"), load(base / "llm_io.json")
    require(
        set(errors) == {"page_errors", "console_errors", "request_failures", "http_errors", "capture_errors"}
        and all(values == [] for key, values in errors.items() if key != "request_failures")
        and llm["provider_call_count"] == 0
        and llm["llm_input"] is llm["llm_output"] is None,
        "browser_or_provider_errors",
    )
    assets = load(base / "map_asset_sources.json")
    verify_asset_observation(assets, parent, "accepted_child", phase="final_candidate_matrix")
    return verify_network(
        base,
        base_url,
        errors["request_failures"],
        {"sources": assets["vector_sources"], "sprite": assets["sprite"]},
        asset_observation=assets,
    )


def immutable_sources(record):
    from itinerary_system.product_app.product_demo import load_product_demo_package
    from itinerary_system.product_app.registry import ProductRunRegistry
    from itinerary_system.product_app.route_delta import configured_route_matrix

    registry = ProductRunRegistry(core.ROOT, core.ROOT / "configs/product_app_registry.json")
    selected = registry.default
    package = load_product_demo_package(core.ROOT, registry.run_dir(selected.run_id))
    bundle = package.primary_bundle
    matrix = configured_route_matrix(core.ROOT, selected.run_id, bundle.route_matrix)
    require(
        record["run_id"] == selected.run_id
        and record["run_directory"] == selected.relative_path
        and record["manifest_sha256"] == selected.manifest_hash
        and record["original_parent"] == bundle.parent_plan
        and record["configured_route_matrix"] == matrix,
        "immutable_source_mismatch",
    )
    manifest = bundle.run_manifest
    anchors = {row["day"]: (row["start_anchor_id"], row["end_anchor_id"]) for row in manifest["scenario"]["days"]}
    require(
        len(anchors) == len(manifest["scenario"]["days"])
        and set(anchors) == {row["day"] for row in bundle.parent_plan["ordered_days"]},
        "declared_anchor_universe_invalid",
    )
    candidates = {row["replacement_stop_id"]: row["plan_id"] for row in manifest["alternatives"]}
    require(len(candidates) == len(manifest["alternatives"]), "immutable_candidate_duplicates")
    return matrix, anchors, candidates, manifest["scenario"]["affected_stop_id"]


def directed_cells(matrix):
    cells = {(row["origin_id"], row["destination_id"]): row for row in matrix["cells"]}
    require(len(cells) == len(matrix["cells"]), "duplicate_directed_road_cell")
    return cells


def road_failure(cell, pair, context_id):
    if cell is None:
        return "missing_route_matrix_cell"
    positive = all(
        type(cell.get(key)) in (int, float)
        and math.isfinite(cell[key])
        and (cell[key] >= 0 if pair[0] == pair[1] else cell[key] > 0)
        for key in ("distance_m", "duration_s")
    )
    if cell.get("road_validated") is not True or cell.get("fallback_used") is not False or not positive:
        return "route_leg_not_road_validated"
    if cell.get("context_snapshot_id") != context_id:
        return "route_context_mismatch"
    points = cell.get("geometry")
    if not isinstance(points, list) or (pair[0] != pair[1] and len(points) < 2):
        return "route_geometry_missing"
    if any(
        not isinstance(point, list)
        or len(point) != 2
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
        or not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180)
        for point in points
    ):
        return "route_geometry_invalid"
    return None


def readiness(parent, matrix, anchors, candidate, day):
    require(
        parent["content_hash"] == core._artifact_content_hash(parent)
        and parent["context_snapshot_id"] == matrix["context_snapshot_id"],
        "parent_hash_or_context_invalid",
    )
    require(candidate not in parent["sequence"] and type(day) is int and day in anchors, "candidate_or_day_invalid")
    days = {row["day"]: row["stop_ids"] for row in parent["ordered_days"]}
    require(
        len(days) == len(parent["ordered_days"])
        and set(days) == set(anchors)
        and [stop for stops in days.values() for stop in stops] == parent["sequence"],
        "parent_day_sequence_invalid",
    )
    start, end = anchors[day]
    sequence = [start, *days[day], candidate, end]
    cells, legs, findings, evidence = directed_cells(matrix), [], [], []
    for origin, destination in zip(sequence[:-1], sequence[1:], strict=True):
        pair = (origin, destination)
        cell = cells.get(pair)
        code = road_failure(cell, pair, parent["context_snapshot_id"])
        leg = {"origin_id": origin, "destination_id": destination, "ready": code is None}
        if cell is not None and code != "route_leg_not_road_validated":
            require(isinstance(cell.get("route_leg_id"), str) and cell["route_leg_id"], "road_leg_identity_missing")
            leg["route_leg_id"] = cell["route_leg_id"]
        legs.append(leg)
        if code:
            findings.append(
                {
                    "code": code,
                    "origin_id": origin,
                    "destination_id": destination,
                    "matrix_id": matrix["matrix_id"],
                    "context_snapshot_id": matrix["context_snapshot_id"],
                }
            )
        evidence.append(
            {"origin_id": origin, "destination_id": destination, "actual_cell": cell, "blocking_code": code}
        )
    status = (
        "route_evidence_ready"
        if not findings
        else (
            "blocked_missing_route_evidence"
            if all(row["code"] == "missing_route_matrix_cell" for row in findings)
            else "blocked_invalid_route_evidence"
        )
    )
    return {
        "candidate_id": candidate,
        "day": day,
        "parent_plan_id": parent["plan_id"],
        "parent_plan_content_hash": parent["content_hash"],
        "context_snapshot_id": parent["context_snapshot_id"],
        "matrix_id": matrix["matrix_id"],
        "insertion_mode": "append_after_day_stops",
        "route_sequence": sequence,
        "required_legs": legs,
        "preview_executable": not findings,
        "status": status,
        "comparison_eligibility": "not_evaluated",
        "blocking_findings": findings,
    }, evidence


def check_readiness(actual, expected):
    fields = set(expected) - {"blocking_findings"}
    require(
        set(actual) == set(expected)
        and all(actual[key] == expected[key] for key in fields)
        and type(actual["preview_executable"]) is bool,
        "pair_readiness_mismatch",
    )
    findings = actual["blocking_findings"]
    require(len(findings) == len(expected["blocking_findings"]), "blocking_findings_missing")
    for actual_finding, expected_finding in zip(findings, expected["blocking_findings"], strict=True):
        require(
            set(actual_finding) == set(expected_finding) | {"corrective_action"}
            and all(actual_finding[key] == value for key, value in expected_finding.items())
            and isinstance(actual_finding["corrective_action"], str)
            and actual_finding["corrective_action"].strip(),
            "blocking_findings_mismatch",
        )


def control_set(rows, candidates, days):
    controls = {row["id"]: row for row in rows}
    require(set(controls) == CONTROL_IDS and len(rows) == len(CONTROL_IDS), "control_inventory_incomplete")
    for name, row in controls.items():
        require(
            row["visible"] is True
            and type(row["disabled"]) is bool
            and row["tag"] == ("select" if name in {"registered-candidate", "destination-day"} else "button"),
            "control_not_visible_or_invalid",
        )
    for name, values in (("registered-candidate", set(candidates)), ("destination-day", {str(day) for day in days})):
        options = controls[name]["options"]
        placeholders = [row for row in options if name == "registered-candidate" and row["value"] == ""]
        option_values = {row["value"] for row in options if row["value"] != ""}
        require(
            option_values == values
            and len(placeholders) <= 1
            and len(options) == len(values) + len(placeholders)
            and all(type(row["disabled"]) is type(row["selected"]) is bool for row in options)
            and [row["value"] for row in options if row["selected"]] == [controls[name]["value"]],
            "control_options_mismatch",
        )
    return controls


def pair_controls(case, candidates, days, capability):
    before = control_set(case["controls_before"], candidates, days)
    selected = control_set(case["controls_selected"], candidates, days)
    candidate = next(row for row in before["registered-candidate"]["options"] if row["value"] == case["candidate_id"])
    day = next(row for row in before["destination-day"]["options"] if row["value"] == str(case["day"]))
    can_select = not (
        before["registered-candidate"]["disabled"]
        or candidate["disabled"]
        or before["destination-day"]["disabled"]
        or day["disabled"]
    )
    require(
        case["forced_interaction"] is False
        and case["pair_selected_through_enabled_controls"] is can_select
        and selected["draft-preview"]["disabled"] is True,
        "forced_or_false_pair_selection",
    )
    expected_add_enabled = (
        capability["enabled"] is True and case["readiness"]["preview_executable"] is True and can_select
    )
    require(selected["draft-add"]["disabled"] is (not expected_add_enabled), "add_enablement_not_capability_bound")
    if can_select:
        require(
            selected["registered-candidate"]["value"] == case["candidate_id"]
            and selected["destination-day"]["value"] == str(case["day"]),
            "selected_pair_dom_mismatch",
        )
    staged = can_select and not selected["draft-add"]["disabled"]
    require(
        case["scope"] == ("real_staging_and_impact_only" if staged else "disabled_pair_not_interacted"),
        "pair_scope_mismatch",
    )
    if not staged:
        require(
            case["readiness"]["preview_executable"] is False
            and case["readiness"]["blocking_findings"]
            and case["draft_request_delta"] == 0
            and isinstance(case["visible_reason"], str)
            and case["visible_reason"].strip()
            and set(case["disabled_selectors"]) == {name for name, row in selected.items() if row["disabled"]}
            and case["candidate_option_disabled"] is candidate["disabled"]
            and case["day_option_disabled"] is day["disabled"]
            and not {"operation", "request", "impact", "undo_response"} & set(case),
            "disabled_pair_mutated_or_misreported",
        )
    return staged


def one_http(records, suffix, *, request=None, response=None, method="POST"):
    rows = [
        row
        for row in records
        if row["method"] == method
        and row["url"].endswith(suffix)
        and (request is None or json.loads(row["request_body_utf8"]) == request)
        and (response is None or json.loads(row["response_body_utf8"]) == response)
    ]
    require(len(rows) == 1, "exact_http_binding_missing_or_duplicate")
    return rows[0]


def legacy_add_control_receipt(case, records):
    """Bind the native Add click to this exact staged candidate/day operation."""
    receipt = case["control_action_receipt"]
    control = receipt["control"]
    events, start, end = receipt["raw_events"], receipt["event_start"], receipt["event_end"]
    staged_http = one_http(records, "/draft/operations", request=case["request"])
    staged = json.loads(staged_http["response_body_utf8"])
    require(
        receipt["assertions_passed"] is True
        and receipt["context"] == case["context"]
        and receipt["phase"] == staged_http["phase"]
        and receipt["request"] == case["request"]
        and receipt["response"] == staged
        and control["id"] == "draft-add"
        and control["visible"] is True
        and control["disabled"] is False
        and type(start) is type(end) is int
        and 0 <= start < end == len(events),
        "legacy_add_control_receipt_invalid",
    )
    clicks = [
        event
        for event in events[start:end]
        if event["type"] == "click"
        and event["trusted"] is True
        and event["id"] == "draft-add"
        and event["path"] == control["path"]
    ]
    require(len(clicks) == 1, "legacy_add_native_click_missing")
    return {
        "scope": "Exact trusted Add click produced the bound candidate/day staged operation",
        "phase": receipt["phase"],
        "operation_id": staged["operation"]["operation_id"],
        "parent_plan_id": case["accepted_plan_id"],
        "parent_plan_content_hash": case["accepted_plan_content_hash"],
    }


def staged_pair(case, records, snapshot, candidates, days):
    parent, operation, impact = snapshot["parent"], case["operation"], case["impact"]
    fields = {
        "type": "add_candidate",
        "target": case["candidate_id"],
        "parameters": {"day": case["day"]},
        "source": "map",
    }
    require(
        all(operation[key] == case["request"][key] == value for key, value in fields.items())
        and set(case["request"]) == set(fields) | {"expected_revision"}
        and case["draft_request_delta"] == 1,
        "staged_pair_semantics_mismatch",
    )
    staged_http = one_http(records, "/draft/operations", request=case["request"])
    staged = json.loads(staged_http["response_body_utf8"])
    legacy_add_control_receipt(case, records)
    session = staged["session"]
    require(
        staged["operation"] == operation
        and session["draft"] == [operation]
        and session["proposal"] is None
        and session["accepted_plan_id"] == parent["plan_id"]
        and session["session_id"] == snapshot["session_id"]
        and session["revision"] == case["request"]["expected_revision"] + 1,
        "stage_session_mismatch",
    )
    impact_http = one_http(
        records, "/draft/impact-preview", request={"expected_revision": session["revision"]}, response=impact
    )
    ready = case["readiness"]["preview_executable"]
    blockers = [] if ready else [case["readiness"]["status"]]
    require(
        impact["session_id"] == session["session_id"]
        and impact["session_revision"] == session["revision"]
        and impact["parent_plan_id"] == parent["plan_id"]
        and impact["parent_plan_content_hash"] == parent["content_hash"]
        and impact["is_certified"] is False
        and impact["certification_status"] == "not_certified"
        and len(impact["operations"]) == impact["summary"]["operation_count"] == 1,
        "impact_identity_or_certification_mismatch",
    )
    item, summary = impact["operations"][0], impact["summary"]
    require(
        item["operation_id"] == operation["operation_id"]
        and item["type"] == "add_candidate"
        and item["target"] == case["candidate_id"]
        and item["preview_executable"] is item["evaluated_repair"] is ready
        and item["feedback_tier"] == ("evaluated" if ready else "draft_only")
        and item["blocking_codes"] == summary["blocking_codes"] == blockers
        and summary["can_run_evaluated_preview"] is ready
        and summary["evaluated_executable_count"] == int(ready)
        and summary["draft_only_count"] == int(not ready),
        "impact_readiness_misrepresented",
    )
    controls = control_set(case["controls_after_stage"], candidates, days)
    require(
        controls["draft-preview"]["disabled"] is (not ready)
        and "Not certified." in case["visible_impact"]
        and ("1 evaluated-preview capable" if ready else "1 draft only") in case["visible_impact"]
        and (ready or "Preview blocked:" in case["visible_impact"]),
        "visible_impact_mismatch",
    )
    undo_http = one_http(records, "/draft/undo", response=case["undo_response"])
    undone = case["undo_response"]
    require(
        json.loads(undo_http["request_body_utf8"]) == {"expected_revision": session["revision"]}
        and undone["undone"] == operation
        and undone["session"]["draft"] == []
        and undone["session"]["proposal"] is None
        and undone["session"]["accepted_plan_id"] == parent["plan_id"]
        and undone["session"]["session_id"] == session["session_id"]
        and undone["session"]["revision"] == session["revision"] + 1,
        "undo_not_bound_or_parent_changed",
    )
    require(
        records.index(staged_http) < records.index(impact_http) < records.index(undo_http),
        "stage_impact_undo_order_invalid",
    )
    require(
        identity(case["map_ack"], snapshot["map_ack"]) and case["map_ack"]["preview_state"] == "PARENT_RENDERED",
        "staging_changed_accepted_map",
    )


def parent_geography(snapshot, matrix, anchors):
    parent, geography, ack = snapshot["parent"], snapshot["geography"], snapshot["map_ack"]
    require(
        parent["content_hash"] == core._artifact_content_hash(parent)
        and geography["plan_id"] == ack["active_plan_id"] == ack["rendered_map_plan_id"] == parent["plan_id"]
        and geography["content_hash"] == ack["active_plan_content_hash"] == parent["content_hash"]
        and ack["rendered_geometry_hash"] == core._completed_race_geometry_hash(geography)
        and ack["preview_state"] == "PARENT_RENDERED"
        and set(ack["rendered_marker_ids"]) == set(parent["sequence"]),
        "parent_geography_or_ack_mismatch",
    )
    stops = sorted(geography["stops"]["features"], key=lambda row: row["properties"]["sequence_index"])
    require([row["properties"]["stop_id"] for row in stops] == parent["sequence"], "parent_marker_order_mismatch")
    cells = directed_cells(matrix)
    for day in parent["ordered_days"]:
        start, end = anchors[day["day"]]
        sequence = [start, *day["stop_ids"], end]
        features = sorted(
            (
                row
                for row in geography["validated_legs"]["features"]
                if row["properties"]["day"] == day["day"] and row["properties"]["cross_day"] is False
            ),
            key=lambda row: row["properties"]["leg_index"],
        )
        require(
            [(row["properties"]["origin_id"], row["properties"]["destination_id"]) for row in features]
            == list(zip(sequence[:-1], sequence[1:], strict=True)),
            "parent_route_not_declared_day_sequence",
        )
        for feature in features:
            prop = feature["properties"]
            pair = (prop["origin_id"], prop["destination_id"])
            cell = cells.get(pair)
            require(
                road_failure(cell, pair, parent["context_snapshot_id"]) is None
                and prop["plan_id"] == parent["plan_id"]
                and prop["content_hash"] == parent["content_hash"]
                and prop["route_leg_id"] == cell["route_leg_id"]
                and prop["distance_m"] == cell["distance_m"]
                and prop["duration_s"] == cell["duration_s"]
                and prop["road_validated"] is True
                and prop["fallback_used"] is False
                and feature["geometry"]
                == {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in cell["geometry"]]},
                "parent_route_not_immutable_road_evidence",
            )


def accepted_baseline(artifacts, snapshots, records, checkpoints, affected, candidates):
    initial, accepted = snapshots["original"], snapshots["accepted_child"]
    payload = artifacts["preview_response"]
    proposal = payload["proposal"]
    phase_records = [row for row in records if row["phase"] == BASELINE_PHASE]
    stage_http = one_http(phase_records, "/draft/operations")
    staged = json.loads(stage_http["response_body_utf8"])
    request, operation = json.loads(stage_http["request_body_utf8"]), staged["operation"]
    candidate = operation["parameters"]["candidate_id"]
    require(
        candidate in candidates
        and operation["type"] == request["type"] == "replace_nearby"
        and operation["target"] == request["target"] == affected
        and operation["source"] == request["source"] == "map"
        and operation["parameters"] == request["parameters"] == {"candidate_id": candidate}
        and staged["session"]["draft"] == [operation]
        and staged["session"]["accepted_plan_id"] == initial["parent"]["plan_id"],
        "baseline_not_real_replacement",
    )
    preview_http = one_http(phase_records, "/preview", response=payload)
    require(
        json.loads(preview_http["request_body_utf8"])["expected_revision"] == staged["session"]["revision"]
        and payload["session"]["draft"] == [operation]
        and payload["session"]["proposal"] == proposal
        and proposal["compiled_request"]["operations"]
        == [{key: value for key, value in operation.items() if key != "created_at"}],
        "replacement_preview_not_staged_operation",
    )
    _proposal(proposal, checkpoints["accepted_child_preview_ready"])
    execution = proposal["runtime_execution"]
    child, certificate = execution["child_plan"], execution["evaluation_certificate"]
    require(
        execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
        and certificate["failures"] == []
        and child == accepted["parent"]
        and proposal["parent_plan_content_hash"] == initial["parent"]["content_hash"]
        and proposal["parent_plan_id"] == initial["parent"]["plan_id"]
        and child["sequence"] == [candidate if stop == affected else stop for stop in initial["parent"]["sequence"]]
        and [row["stop_id"] for row in execution["repair_request"]["candidate_pois"]] == [candidate],
        "baseline_child_or_evaluation_mismatch",
    )
    route = certificate["route_validation"]
    require(
        route["required_leg_count"] > 0
        and route["road_validated_leg_count"] == route["required_leg_count"]
        and route["missing_leg_count"] == route["fallback_leg_count"] == 0,
        "baseline_route_incomplete",
    )
    result = artifacts["acceptance_response"]
    accept_http = one_http(phase_records, "/accept", response=result)
    accepted_request = json.loads(accept_http["request_body_utf8"])
    decision, pointer = result["result"]["decision"], result["result"]["pointer"]
    for record in (decision, pointer):
        unsigned = {key: value for key, value in record.items() if key != "content_hash"}
        require(
            record["content_hash"]
            == hashlib.sha256(
                json.dumps(
                    unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                ).encode()
            ).hexdigest(),
            "acceptance_record_hash_mismatch",
        )
    require(
        accepted_request["confirmation"] == "accept_evaluated_repair"
        and accepted_request["expected_revision"] == payload["session"]["revision"]
        and decision["decision_type"] == "accept_repair"
        and decision["idempotency_key"] == accepted_request["idempotency_key"]
        and decision["previous_plan_id"] == initial["parent"]["plan_id"]
        and decision["selected_plan_id"] == child["plan_id"]
        and decision["selected_plan_content_hash"] == child["content_hash"]
        and decision["certificate_id"] == certificate["certificate_id"]
        and decision["certificate_content_hash"] == certificate["content_hash"]
        and decision["diff_id"] == proposal["diff_id"]
        and decision["diff_content_hash"] == proposal["diff_content_hash"]
        and pointer["accepted_plan_id"] == child["plan_id"]
        and pointer["accepted_plan_content_hash"] == child["content_hash"]
        and pointer["original_plan_id"] == initial["parent"]["plan_id"]
        and pointer["decision_id"] == decision["decision_id"],
        "accepted_baseline_not_explicit_or_canonical",
    )
    session_id = accepted["session_id"]
    reload_http = one_http(phase_records, "/api/sessions/" + session_id, method="GET")
    saved = json.loads(reload_http["response_body_utf8"])
    require(
        artifacts["restored_workspace"] == saved["workspace"]
        and artifacts["map_ack"] == accepted["map_ack"]
        and saved["session"]["accepted_plan_id"] == result["session"]["accepted_plan_id"] == child["plan_id"]
        and saved["session"]["draft"] == result["session"]["draft"] == []
        and saved["session"]["proposal"] is result["session"]["proposal"] is None
        and saved["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"]
        and saved["workspace"]["alternative_evidence"][child["plan_id"]]["certificate"] == certificate
        and identity(checkpoints["accepted_child_preview_ready"], accepted["map_ack"]),
        "accepted_reload_mismatch",
    )
    require(
        records.index(stage_http)
        < records.index(preview_http)
        < records.index(accept_http)
        < records.index(reload_http),
        "baseline_request_order_invalid",
    )


def pair_inventory(rows, expected, reason):
    require(len(rows) == len(expected) and {(row["candidate_id"], row["day"]) for row in rows} == set(expected), reason)


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    tests = list(junit.iter("testcase"))
    require(
        len(tests) == 1
        and tests[0].get("name") == TEST
        and tests[0].get("classname") == "tests.product_app.browser.test_candidate_day_matrix_live"
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()),
        "browser_test_incomplete",
    )
    base = root / SCENARIO
    require(not any(path.is_dir() for path in base.iterdir()), "unexpected_case_directories")
    immutable = core._load(base / "immutable_inputs.json")
    matrix, anchors, registered_candidates, affected = immutable_sources(immutable)
    snapshots = core._load(base / "capability_snapshots.json")
    require(
        set(snapshots) == set(CONTEXTS) and snapshots["original"]["parent"] == immutable["original_parent"],
        "baseline_universe_invalid",
    )
    records, cases = _http(base, base_url), load(base / "candidate_day_cases.json")
    require(
        all(row.get("phase") for row in records) and not any("/copilot/" in row["url"] for row in records),
        "unexpected_chat_or_missing_phase",
    )
    checkpoints, inventory = core._load(base / "map_checkpoints.json"), load(base / "runtime_control_inventory.json")
    assertions = core._load(base / "browser_assertions.json")
    require(
        assertions["schema_version"] == "candidate-day-matrix-browser-v1"
        and assertions["assertions_passed"] is True
        and assertions["contexts"] == list(CONTEXTS)
        and assertions["all_pairs_recorded"] is True
        and assertions["pair_scope"] == "readiness_and_staging_not_eligibility"
        and assertions["eligible_addition_acceptance_owner"] == "addition-continuation"
        and all(
            assertions[key] == 0
            for key in ("forced_disabled_interactions", "direct_api_setup_calls", "provider_call_count")
        ),
        "scope_or_claims_invalid",
    )
    expected_keys, ready_keys, staged_keys = set(), [], []
    expected_inventory, capture_labels = (
        [],
        [
            "original_baseline",
            "original_matrix_complete",
            "accepted_child_preview_ready",
            "accepted_child_after_reload",
            "accepted_child_matrix_complete",
        ],
    )
    for context in CONTEXTS:
        snapshot = snapshots[context]
        parent, capability = snapshot["parent"], snapshot["capability"]
        parent_geography(snapshot, matrix, anchors)
        candidates = sorted(set(registered_candidates) - set(parent["sequence"]))
        days = sorted(row["day"] for row in parent["ordered_days"])
        choices = snapshot["candidate_choices"]
        require(
            snapshot["days"] == days
            and len(choices) == len(candidates)
            and sorted(row["candidate_id"] for row in choices) == candidates
            and all(
                row["child_plan_id"] == registered_candidates[row["candidate_id"]]
                and row["replaces_stop_id"] == affected
                for row in choices
            ),
            "candidate_day_universe_not_immutable",
        )
        bindings = [
            json.loads(row["response_body_utf8"])
            for row in records
            if row["method"] == "GET" or row["url"].endswith("/api/sessions")
        ]
        require(
            any(
                isinstance(item, dict)
                and item.get("session", {}).get("session_id") == snapshot["session_id"]
                and item["session"]["accepted_plan_id"] == parent["plan_id"]
                and "workspace" in item
                and item["workspace"]["accepted_plan_pointer"]["accepted_plan_content_hash"] == parent["content_hash"]
                and item["workspace"]["draft_capabilities"]["candidate_choices"] == choices
                and item["workspace"]["draft_capabilities"]["operations"]["add_candidate"] == capability
                and snapshot["geography"] in item["workspace"]["geography"]["plans"]
                for item in bindings
            ),
            "capability_snapshot_http_unbound",
        )
        expected = {
            (candidate, day): readiness(parent, matrix, anchors, candidate, day)
            for candidate in candidates
            for day in days
        }
        readiness_rows = capability["candidate_day_readiness"]
        pair_inventory(readiness_rows, expected, "readiness_inventory_incomplete")
        any_ready = any(row[0]["preview_executable"] for row in expected.values())
        all_missing = all(row[0]["status"] == "blocked_missing_route_evidence" for row in expected.values())
        require(
            capability["enabled"] is capability["preview_executable"] is any_ready
            and capability["status"]
            == (
                "route_evidence_ready"
                if any_ready
                else "blocked_missing_route_evidence"
                if all_missing
                else "blocked_invalid_route_evidence"
            ),
            "aggregate_capability_readiness_wrong",
        )
        for actual in readiness_rows:
            check_readiness(actual, expected[(actual["candidate_id"], actual["day"])][0])
        context_cases = [row for row in cases if row["context"] == context]
        pair_inventory(context_cases, expected, "pair_case_inventory_incomplete")
        for case in context_cases:
            key = (context, case["candidate_id"], case["day"])
            expected_keys.add(key)
            expected_row, legs = expected[(key[1], key[2])]
            check_readiness(case["readiness"], expected_row)
            require(
                case["required_leg_evidence"] == legs
                and case["assertions_passed"] is True
                and case["accepted_plan_id"] == parent["plan_id"]
                and case["accepted_plan_content_hash"] == parent["content_hash"]
                and case["comparison_eligibility"] == "not_evaluated",
                "pair_evidence_or_scope_mismatch",
            )
            phase = f"{context}_{key[1]}_day_{key[2]}"
            phase_records = [row for row in records if row["phase"] == phase]
            expected_inventory.append({"phase": phase, "controls": case["controls_selected"]})
            if expected_row["preview_executable"]:
                ready_keys.append(key)
            staged = pair_controls(case, candidates, days, capability)
            if staged:
                staged_pair(case, phase_records, snapshot, candidates, days)
                staged_keys.append(key)
                capture_labels.append(phase + "_staged")
                require(checkpoints[phase + "_staged"] == case["map_ack"], "staged_map_checkpoint_unbound")
            else:
                require(
                    not any(
                        row["method"] == "POST"
                        and row["url"].endswith(("/draft/operations", "/preview", "/accept", "/draft/undo"))
                        for row in phase_records
                    ),
                    "disabled_pair_has_mutation",
                )
        baseline_label = "original_baseline" if context == "original" else "accepted_child_after_reload"
        require(
            checkpoints[baseline_label] == snapshot["map_ack"]
            and identity(checkpoints[context + "_matrix_complete"], snapshot["map_ack"]),
            "context_end_map_changed",
        )
    require(
        len(cases) == len(expected_keys) == assertions["pair_count"]
        and inventory == expected_inventory
        and assertions["pairs_by_context"]
        == {context: sum(key[0] == context for key in expected_keys) for context in CONTEXTS},
        "pair_inventory_claim_mismatch",
    )
    accepted_baseline(
        core._load(base / "accepted_baseline_artifacts.json"),
        snapshots,
        records,
        checkpoints,
        affected,
        registered_candidates,
    )
    from verify_e2e1_poi_controls import accepted_inspection, accepted_rows

    accepted_rows(
        load(base / "accepted_poi_control_journeys.json"),
        core._load(base / "accepted_poi_control_input_log.json"),
        snapshots["accepted_child"],
        records,
    )
    accepted_inspection(core._load(base / "accepted_candidate_inspection.json"), snapshots["accepted_child"], records)
    posts = [row for row in records if row["method"] == "POST"]
    for suffix, count in (
        ("/api/sessions", 1),
        ("/preview", 1),
        ("/accept", 1),
        ("/draft/operations", len(staged_keys) + 1),
        ("/draft/undo", len(staged_keys)),
    ):
        require(sum(row["url"].endswith(suffix) for row in posts) == count, "mutation_inventory_mismatch")
    require(
        assertions["preview_count"] == assertions["acceptance_count"] == 1
        and all(row["phase"] == BASELINE_PHASE for row in posts if row["url"].endswith(("/preview", "/accept")))
        and all(
            row["url"].endswith(
                (
                    "/api/sessions",
                    "/selection",
                    "/draft/impact-preview",
                    "/draft/operations",
                    "/draft/undo",
                    "/preview",
                    "/accept",
                )
            )
            for row in posts
        ),
        "unexpected_setup_or_addition_acceptance",
    )
    require(set(checkpoints) == set(capture_labels), "map_checkpoint_inventory_mismatch")
    core._verify_journey_captures(
        base, capture_labels, {snapshot["parent"]["plan_id"] for snapshot in snapshots.values()}
    )
    browser_health(base, base_url, snapshots["accepted_child"]["map_ack"])
    return {
        "verified": True,
        "test_count": 1,
        "scope": "candidate_day_readiness_and_real_staging_undo",
        "pair_count": len(expected_keys),
        "route_ready_pairs": ready_keys,
        "staged_pairs": staged_keys,
        "addition_evaluation_claimed": False,
        "addition_acceptance_claimed": False,
        "provider_calls": 0,
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
        raise core.E2E1VerificationError("e2e1_candidate_days_evidence_malformed") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8127")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.base_url), indent=2))
