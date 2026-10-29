"""Adversarial offline checks; fixtures are not browser or provider evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import test_e2e1_network_observations_verifier as network_fixture  # noqa: E402
import verify_e2e1_candidate_day_matrix as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def _fixture(blocked=False):
    parent = {"plan_id": "parent", "context_snapshot_id": "context", "sequence": ["kept"],
        "ordered_days": [{"day": 1, "stop_ids": ["kept"]}]}
    parent["content_hash"] = core._artifact_content_hash(parent)
    cells = [{"origin_id": origin, "destination_id": destination, "route_leg_id": "leg_" + str(index),
        "road_validated": True, "fallback_used": False, "distance_m": 1000, "duration_s": 100,
        "context_snapshot_id": "context", "geometry": [[37.0, -122.0], [37.1, -122.1]]}
        for index, (origin, destination) in enumerate((("start", "kept"), ("kept", "candidate"), ("candidate", "end")))]
    if blocked:
        cells.pop(1)
    return parent, {"matrix_id": "matrix", "context_snapshot_id": "context", "cells": cells}, {1: ("start", "end")}


def _http(path, request, response):
    return {"phase": "original_candidate_day_1", "method": "POST", "url": "http://127.0.0.1:8127/api/sessions/session" + path,
        "request_body_utf8": json.dumps(request), "response_body_utf8": json.dumps(response)}


def _controls(enabled=True, day_disabled=False, preview_disabled=True):
    return [{"id": name, "tag": "select" if values else "button", "value": values[0] if values else "",
        "visible": True, "disabled": (not enabled if name == "draft-add" else day_disabled if name == "destination-day"
            else preview_disabled if name == "draft-preview" else False),
        "options": [{"value": value, "label": value, "disabled": False, "selected": index == 0}
            for index, value in enumerate(values)]}
        for name, values in (("registered-candidate", ["candidate"]), ("destination-day", ["1"]), ("draft-add", []), ("draft-preview", []))]


def _case(blocked=False):
    parent, matrix, anchors = _fixture(blocked)
    ready, evidence = verifier.readiness(parent, matrix, anchors, "candidate", 1)
    for finding in ready["blocking_findings"]:
        finding["corrective_action"] = "Provide this directed road evidence before preview."
    operation = {"operation_id": "operation", "type": "add_candidate", "target": "candidate", "parameters": {"day": 1},
        "source": "map", "created_at": "2026-09-08T12:00:00+00:00", "evidence_refs": []}
    request = {key: operation[key] for key in ("type", "target", "parameters", "source")}
    request["expected_revision"] = 4
    session = {"session_id": "session", "revision": 5, "accepted_plan_id": "parent", "draft": [operation], "proposal": None}
    staged = {"operation": operation, "session": session}
    item = {"operation_id": "operation", "type": "add_candidate", "target": "candidate", "preview_executable": not blocked,
        "evaluated_repair": not blocked, "feedback_tier": "draft_only" if blocked else "evaluated",
        "blocking_codes": [ready["status"]] if blocked else []}
    impact = {"session_id": "session", "session_revision": 5, "parent_plan_id": "parent", "parent_plan_content_hash": parent["content_hash"],
        "is_certified": False, "certification_status": "not_certified", "operations": [item],
        "summary": {"operation_count": 1, "can_run_evaluated_preview": not blocked, "evaluated_executable_count": int(not blocked),
            "draft_only_count": int(blocked), "blocking_codes": item["blocking_codes"]}}
    undo = {"undone": operation, "session": {**session, "revision": 6, "draft": []}}
    ack = {"active_plan_id": "parent", "active_plan_content_hash": parent["content_hash"], "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "geometry", "rendered_marker_ids": ["kept"], "preview_state": "PARENT_RENDERED"}
    receipt = {"event_start": 0, "event_end": 1,
        "control": {"id": "draft-add", "visible": True, "disabled": False, "path": "/app/itinerary"},
        "context": "original", "phase": "original_candidate_day_1", "request": request, "response": staged,
        "raw_events": [{"type": "click", "trusted": True, "id": "draft-add", "path": "/app/itinerary"}],
        "assertions_passed": True}
    case = {"context": "original", "accepted_plan_id": "parent", "accepted_plan_content_hash": parent["content_hash"],
        "candidate_id": "candidate", "day": 1, "readiness": ready, "required_leg_evidence": evidence,
        "controls_before": _controls(), "controls_selected": _controls(), "forced_interaction": False,
        "pair_selected_through_enabled_controls": True, "scope": "real_staging_and_impact_only", "draft_request_delta": 1,
        "operation": operation, "request": request, "impact": impact, "controls_after_stage": _controls(preview_disabled=blocked),
        "visible_impact": "Not certified. " + ("1 draft only Preview blocked:" if blocked else "1 evaluated-preview capable"),
        "undo_response": undo, "map_ack": ack, "control_action_receipt": receipt}
    records = [_http("/draft/operations", request, staged), _http("/draft/impact-preview", {"expected_revision": 5}, impact),
        _http("/draft/undo", {"expected_revision": 5}, undo)]
    snapshot = {"parent": parent, "session_id": "session", "map_ack": deepcopy(ack)}
    return case, records, snapshot


def test_append_keeps_existing_stop_and_requires_forward_cells():
    parent, matrix, anchors = _fixture()
    expected, evidence = verifier.readiness(parent, matrix, anchors, "candidate", 1)
    assert expected["route_sequence"] == ["start", "kept", "candidate", "end"]
    assert expected["preview_executable"] is True and expected["comparison_eligibility"] == "not_evaluated"
    assert [(row["origin_id"], row["destination_id"]) for row in evidence] == [("start", "kept"), ("kept", "candidate"), ("candidate", "end")]
    matrix["cells"][1]["origin_id"], matrix["cells"][1]["destination_id"] = "candidate", "kept"
    actual, _ = verifier.readiness(parent, matrix, anchors, "candidate", 1)
    assert actual["preview_executable"] is False and actual["blocking_findings"][0]["code"] == "missing_route_matrix_cell"


@pytest.mark.parametrize("fault,code", [("absent", "missing_route_matrix_cell"), ("fallback", "route_leg_not_road_validated"),
    ("unvalidated", "route_leg_not_road_validated"), ("duration", "route_leg_not_road_validated"),
    ("negative", "route_leg_not_road_validated"), ("boolean", "route_leg_not_road_validated"), ("nan", "route_leg_not_road_validated"),
    ("context", "route_context_mismatch"), ("empty", "route_geometry_missing"), ("short", "route_geometry_missing"),
    ("latitude", "route_geometry_invalid"), ("point_boolean", "route_geometry_invalid")])
def test_raw_road_evidence_counterexamples(fault, code):
    _, matrix, _ = _fixture()
    cell = matrix["cells"][1]
    if fault == "absent":
        cell = None
    elif fault == "fallback":
        cell["fallback_used"] = True
    elif fault == "unvalidated":
        cell["road_validated"] = False
    elif fault in {"duration", "negative", "boolean", "nan"}:
        cell["duration_s"] = {"duration": None, "negative": -1, "boolean": True, "nan": float("nan")}[fault]
    elif fault == "context":
        cell["context_snapshot_id"] = "other"
    else:
        cell["geometry"] = {"empty": [], "short": [[37, -122]], "latitude": [[137, -122], [37, -122]],
            "point_boolean": [[True, -122], [37, -122]]}[fault]
    assert verifier.road_failure(cell, ("kept", "candidate"), "context") == code


@pytest.mark.parametrize("fault", ["plan", "hash", "context", "matrix", "reordered", "reversed_leg", "leg_id", "ready", "eligibility", "findings"])
def test_readiness_payload_tampering_rejected(fault):
    parent, matrix, anchors = _fixture(blocked=fault == "findings")
    expected, _ = verifier.readiness(parent, matrix, anchors, "candidate", 1)
    actual = deepcopy(expected)
    for finding in actual["blocking_findings"]:
        finding["corrective_action"] = "Provide directed evidence."
    verifier.check_readiness(actual, expected)
    if fault in {"plan", "hash", "context", "matrix", "eligibility"}:
        actual[{"plan": "parent_plan_id", "hash": "parent_plan_content_hash", "context": "context_snapshot_id", "matrix": "matrix_id",
            "eligibility": "comparison_eligibility"}[fault]] = "wrong"
    elif fault == "reordered":
        actual["route_sequence"][1:3] = actual["route_sequence"][2:0:-1]
    elif fault == "reversed_leg":
        actual["required_legs"][0]["origin_id"] = "kept"
    elif fault == "leg_id":
        actual["required_legs"][0]["route_leg_id"] = "wrong"
    elif fault == "ready":
        actual["preview_executable"] = False
    elif fault == "findings":
        actual["blocking_findings"] = []
    with pytest.raises(core.E2E1VerificationError, match="e2e1_candidate_days_"):
        verifier.check_readiness(actual, expected)


def test_staged_ready_draft_remains_uncertified(tmp_path):
    case, records, snapshot = _case(blocked=False)
    assert verifier.pair_controls(case, ["candidate"], [1], {"enabled": True}) is True
    verifier.staged_pair(case, records, snapshot, ["candidate"], [1])
    (tmp_path / "verifier_exact_io.json").write_text(json.dumps({"scope": "offline_synthetic_verifier_fixture",
        "input": {"case": case, "http": records, "snapshot": snapshot}, "output": {"verified": True}, "provider_calls": 0}, indent=2), encoding="utf-8")


@pytest.mark.parametrize("fault", ["missing", "untrusted", "wrong_control", "wrong_response"])
def test_legacy_add_requires_exact_trusted_control_receipt(fault):
    case, records, _ = _case(blocked=False)
    if fault == "missing":
        del case["control_action_receipt"]
    elif fault == "untrusted":
        case["control_action_receipt"]["raw_events"][0]["trusted"] = False
    elif fault == "wrong_control":
        case["control_action_receipt"]["control"]["id"] = "draft-replace"
    else:
        case["control_action_receipt"]["response"] = {}
    with pytest.raises((core.E2E1VerificationError, KeyError)):
        verifier.legacy_add_control_receipt(case, records)


@pytest.mark.parametrize("fault", ["operation", "wire_request", "impact_id", "certified", "false_ready", "undo_id", "undo_parent",
    "undo_revision", "out_of_order", "duplicate_stage", "preview_enabled", "map_child", "marker_change", "hidden_control", "forced", "selected_day"])
def test_staging_http_ui_undo_disconnection_rejected(fault):
    case, records, snapshot = _case(blocked=True)
    verifier.staged_pair(case, records, snapshot, ["candidate"], [1])
    if fault == "operation":
        case["operation"]["type"] = "mark_flexible"
    elif fault == "wire_request":
        records[0]["request_body_utf8"] = json.dumps({**case["request"], "parameters": {"day": 2}})
    elif fault in {"impact_id", "certified", "false_ready"}:
        if fault == "impact_id":
            case["impact"]["operations"][0]["operation_id"] = "wrong"
        elif fault == "certified":
            case["impact"]["is_certified"] = True
        else:
            case["impact"]["summary"]["can_run_evaluated_preview"] = True
        records[1]["response_body_utf8"] = json.dumps(case["impact"])
    elif fault.startswith("undo_"):
        if fault == "undo_id":
            case["undo_response"]["undone"] = {**case["operation"], "operation_id": "wrong"}
        else:
            case["undo_response"]["session"]["accepted_plan_id" if fault == "undo_parent" else "revision"] = "wrong"
        records[2]["response_body_utf8"] = json.dumps(case["undo_response"])
    elif fault == "out_of_order":
        records.reverse()
    elif fault == "duplicate_stage":
        records.append(deepcopy(records[0]))
    elif fault == "preview_enabled":
        case["controls_after_stage"][-1]["disabled"] = False
    elif fault == "map_child":
        case["map_ack"]["rendered_map_plan_id"] = "child"
    elif fault == "marker_change":
        case["map_ack"]["rendered_marker_ids"] = ["candidate"]
    elif fault == "hidden_control":
        case["controls_selected"][0]["visible"] = False
    elif fault == "forced":
        case["forced_interaction"] = True
    elif fault == "selected_day":
        case["controls_selected"][1]["value"] = "2"
    with pytest.raises(core.E2E1VerificationError, match="e2e1_candidate_days_"):
        verifier.pair_controls(case, ["candidate"], [1], {"enabled": True})
        verifier.staged_pair(case, records, snapshot, ["candidate"], [1])


def test_disabled_pair_requires_no_staging_and_truthful_control_state():
    case, _, _ = _case(blocked=True)
    for key in ("operation", "request", "impact", "undo_response"):
        del case[key]
    case.update(scope="disabled_pair_not_interacted", draft_request_delta=0, pair_selected_through_enabled_controls=False,
        controls_before=_controls(enabled=False, day_disabled=True), controls_selected=_controls(enabled=False, day_disabled=True),
        disabled_selectors=["destination-day", "draft-add", "draft-preview"], candidate_option_disabled=False,
        day_option_disabled=False, visible_reason="Changes remain drafts until Preview repair finishes.")
    assert verifier.pair_controls(case, ["candidate"], [1], {"enabled": False}) is False
    case["request"] = {"type": "add_candidate"}
    with pytest.raises(core.E2E1VerificationError, match="disabled_pair_mutated_or_misreported"):
        verifier.pair_controls(case, ["candidate"], [1], {"enabled": False})


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_candidate", "wrong_day"])
def test_cartesian_pair_inventory_is_complete_without_hardcoded_counts(fault):
    expected = {(candidate, day) for candidate in ("first", "second", "third") for day in (2, 4)}
    rows = [{"candidate_id": candidate, "day": day} for candidate, day in sorted(expected)]
    verifier.pair_inventory(rows, expected, "inventory")
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = dict(rows[0])
    elif fault == "wrong_candidate":
        rows[0]["candidate_id"] = "not_registered"
    else:
        rows[0]["day"] = 7
    with pytest.raises(core.E2E1VerificationError, match="e2e1_candidate_days_inventory"):
        verifier.pair_inventory(rows, expected, "inventory")


def test_live_configured_immutable_sources_and_two_context_oracle(tmp_path):
    """Read actual pinned files; no server, compiler, renderer, or provider call."""
    from itinerary_system.product_app.product_demo import load_product_demo_package
    from itinerary_system.product_app.registry import ProductRunRegistry
    from itinerary_system.product_app.route_delta import configured_route_matrix

    registry = ProductRunRegistry(core.ROOT, core.ROOT / "configs/product_app_registry.json")
    registered = registry.default
    package = load_product_demo_package(core.ROOT, registry.run_dir(registered.run_id))
    bundle = package.primary_bundle
    record = {"run_id": registered.run_id, "run_directory": registered.relative_path, "manifest_sha256": registered.manifest_hash,
        "original_parent": bundle.parent_plan, "configured_route_matrix": configured_route_matrix(core.ROOT, registered.run_id, bundle.route_matrix)}
    matrix, anchors, candidates, _ = verifier.immutable_sources(record)
    contexts = {"original": bundle.parent_plan, "frozen_recommended_fixture_not_browser_acceptance": bundle.child_plan}
    rows = []
    for context, parent in contexts.items():
        pairs = {(candidate, day) for candidate in candidates if candidate not in parent["sequence"] for day in anchors}
        for candidate, day in sorted(pairs):
            actual, evidence = verifier.readiness(parent, matrix, anchors, candidate, day)
            rows.append({"context": context, "readiness": actual, "directed_cell_evidence": evidence})
        verifier.pair_inventory([row["readiness"] for row in rows if row["context"] == context], pairs, "immutable_oracle_inventory")
    (tmp_path / "immutable_oracle_exact_io.json").write_text(json.dumps({"scope": "offline_immutable_evidence_oracle_not_browser_acceptance",
        "input": record, "output": rows, "provider_calls": 0}, indent=2), encoding="utf-8")
    forged = deepcopy(record)
    forged["configured_route_matrix"]["cells"][0]["duration_s"] += 1
    with pytest.raises(core.E2E1VerificationError, match="immutable_source_mismatch"):
        verifier.immutable_sources(forged)


def _baseline():
    parent, _, _ = _fixture()
    child = {"plan_id": "child", "parent_plan_id": "parent", "planning_request_id": "repair-request", "sequence": ["candidate"]}
    child["content_hash"] = core._artifact_content_hash(child)
    geo = {"plan_id": "child", "content_hash": child["content_hash"], "stops": {"features": [{"properties": {"stop_id": "candidate"}}]},
        "validated_legs": {"type": "FeatureCollection", "features": []}}
    ack = {"active_plan_id": "child", "active_plan_content_hash": child["content_hash"], "rendered_map_plan_id": "child",
        "rendered_geometry_hash": core._completed_race_geometry_hash(geo), "rendered_marker_ids": ["candidate"]}
    certificate = {"plan_id": "child", "plan_content_hash": child["content_hash"], "certificate_id": "certificate",
        "comparison_eligibility": "eligible", "failures": [], "route_validation": {"required_leg_count": 2,
            "road_validated_leg_count": 2, "missing_leg_count": 0, "fallback_leg_count": 0}}
    certificate["content_hash"] = core._artifact_content_hash(certificate)
    diff = {"parent_plan_id": "parent", "child_plan_id": "child", "diff_id": "diff"}
    operation = {"type": "replace_nearby", "target": "kept", "parameters": {"candidate_id": "candidate"},
        "source": "map", "created_at": "timestamp", "operation_id": "replacement"}
    request = {key: operation[key] for key in ("type", "target", "parameters", "source")}
    staged = {"operation": operation, "session": {"revision": 2, "draft": [operation], "accepted_plan_id": "parent"}}
    proposal = {"state": "eligible", "execution_mode": "deterministic_repair_pipeline", "child_plan_id": "child",
        "child_plan_content_hash": child["content_hash"], "parent_plan_id": "parent", "parent_plan_content_hash": parent["content_hash"],
        "certificate_id": "certificate", "certificate_content_hash": certificate["content_hash"], "diff_id": "diff",
        "diff_content_hash": core._artifact_content_hash(diff), "geography_plan": geo,
        "compiled_request": {"operations": [{key: value for key, value in operation.items() if key != "created_at"}]},
        "runtime_execution": {"evaluation_executed": True, "evaluator": "PlanEvaluator.evaluate_final_plan", "request_id": "repair-request",
            "repair_request": {"request_id": "repair-request", "candidate_pois": [{"stop_id": "candidate"}]},
            "child_plan": child, "evaluation_certificate": certificate, "plan_diff": diff}}
    payload = {"proposal": proposal, "session": {"revision": 3, "draft": [operation], "proposal": proposal}}
    accept_request = {"confirmation": "accept_evaluated_repair", "expected_revision": 3, "idempotency_key": "accept-unique"}
    decision = {"decision_id": "decision", "decision_type": "accept_repair", "idempotency_key": "accept-unique", "previous_plan_id": "parent",
        "selected_plan_id": "child", "selected_plan_content_hash": child["content_hash"], "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"], "diff_id": "diff", "diff_content_hash": proposal["diff_content_hash"]}
    pointer = {"accepted_plan_id": "child", "accepted_plan_content_hash": child["content_hash"], "original_plan_id": "parent", "decision_id": "decision"}
    for record in (decision, pointer):
        record["content_hash"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    session = {"accepted_plan_id": "child", "draft": [], "proposal": None}
    accepted = {"result": {"decision": decision, "pointer": pointer}, "session": session}
    workspace = {"accepted_plan_pointer": {"decision_id": "decision"}, "alternative_evidence": {"child": {"certificate": certificate}}}
    saved = {"workspace": workspace, "session": session}
    artifacts = {"preview_response": payload, "acceptance_response": accepted, "restored_workspace": workspace, "map_ack": ack}
    snapshots = {"original": {"parent": parent}, "accepted_child": {"parent": child, "session_id": "session", "map_ack": ack}}
    records = [_http("/draft/operations", request, staged), _http("/preview", {"expected_revision": 2}, payload),
        _http("/accept", accept_request, accepted), {"method": "GET", "url": "http://127.0.0.1:8127/api/sessions/session", "response_body_utf8": json.dumps(saved)}]
    for row in records:
        row["phase"] = verifier.BASELINE_PHASE
    return artifacts, snapshots, records, {"accepted_child_preview_ready": ack}


def test_baseline_requires_one_real_replacement_preview_accept_and_reload():
    artifacts, snapshots, records, checkpoints = _baseline()
    verifier.accepted_baseline(artifacts, snapshots, records, checkpoints, "kept", {"candidate": "registered-child"})


@pytest.mark.parametrize("fault", ["addition_instead", "compiled_operation", "no_confirmation", "decision_hash", "coherent_wrong_decision",
    "accepted_parent", "reload_parent", "duplicate_preview", "no_acceptance", "out_of_order", "wrong_map"])
def test_baseline_acceptance_counterexamples_fail(fault):
    artifacts, snapshots, records, checkpoints = _baseline()
    verifier.accepted_baseline(artifacts, snapshots, records, checkpoints, "kept", {"candidate": "registered-child"})
    if fault == "addition_instead":
        request = json.loads(records[0]["request_body_utf8"])
        stage = json.loads(records[0]["response_body_utf8"])
        request["type"] = stage["operation"]["type"] = "add_candidate"
        records[0]["request_body_utf8"], records[0]["response_body_utf8"] = json.dumps(request), json.dumps(stage)
    elif fault == "compiled_operation":
        artifacts["preview_response"]["proposal"]["compiled_request"]["operations"] = []
        records[1]["response_body_utf8"] = json.dumps(artifacts["preview_response"])
    elif fault == "no_confirmation":
        request = json.loads(records[2]["request_body_utf8"])
        request["confirmation"] = "automatic"
        records[2]["request_body_utf8"] = json.dumps(request)
    elif fault in {"decision_hash", "coherent_wrong_decision"}:
        decision = artifacts["acceptance_response"]["result"]["decision"]
        if fault == "decision_hash":
            decision["content_hash"] = "wrong"
        else:
            decision["selected_plan_id"] = "other"
            unsigned = {key: value for key, value in decision.items() if key != "content_hash"}
            decision["content_hash"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records[2]["response_body_utf8"] = json.dumps(artifacts["acceptance_response"])
    elif fault == "accepted_parent":
        snapshots["accepted_child"]["parent"] = snapshots["original"]["parent"]
    elif fault == "reload_parent":
        saved = json.loads(records[3]["response_body_utf8"])
        saved["session"]["accepted_plan_id"] = "parent"
        records[3]["response_body_utf8"] = json.dumps(saved)
    elif fault == "duplicate_preview":
        records.append(deepcopy(records[1]))
    elif fault == "no_acceptance":
        records.pop(2)
    elif fault == "out_of_order":
        records.reverse()
    elif fault == "wrong_map":
        checkpoints["accepted_child_preview_ready"]["rendered_map_plan_id"] = "parent"
    with pytest.raises(core.E2E1VerificationError):
        verifier.accepted_baseline(artifacts, snapshots, records, checkpoints, "kept", {"candidate": "registered-child"})


def _network_fixture(cancelled=True):
    document, records, configuration, style = network_fixture.inputs()
    assets, parent = network_fixture.asset_inputs()
    assets.update(context="accepted_child", phase="final_candidate_matrix",
        map_configuration_http_file="http_001.json", vector_sources=style["sources"], sprite=style["sprite"])
    document["rendered_map_asset_sources_file"] = "map_asset_sources.json"
    if not cancelled:
        records = []
        document.update(raw_record_count=0, classifications=[], browser_cancelled_local_map_asset_indexes=[])
    files = {
        "browser_errors.json": {"page_errors": [], "console_errors": [], "request_failures": records,
            "http_errors": [], "capture_errors": []},
        "llm_io.json": {"provider_call_count": 0, "llm_input": None, "llm_output": None},
        "http_001.json": network_fixture.receipt(configuration),
        "network_classification.json": document,
        "map_asset_sources.json": assets,
    }
    return files, parent


def _network_run(tmp_path, files, parent):
    for name, body in files.items():
        (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    return verifier.browser_health(tmp_path, network_fixture.BASE, parent)


@pytest.mark.parametrize("cancelled", [False, True])
def test_candidate_network_preserves_and_binds_observations(tmp_path, cancelled):
    files, parent = _network_fixture(cancelled)
    before = deepcopy(files)
    result = _network_run(tmp_path, files, parent)
    assert result["verified"] is True and result["raw_observations"] == int(cancelled)
    assert result["map_disposal_cause_claimed"] is False
    assert files == before


@pytest.mark.parametrize("fault", ["api_cancel", "connection_refused", "http_error", "missing_method",
    "wrong_plan", "incomplete_tiles", "omitted_record", "wrong_archive", "console_error", "provider_call"])
def test_candidate_network_cannot_hide_errors_or_unready_map(tmp_path, fault):
    files, parent = _network_fixture()
    record = files["browser_errors.json"]["request_failures"][0]
    if fault == "api_cancel":
        record["url"] = network_fixture.BASE + "/api/sessions/session/preview"
    elif fault == "connection_refused":
        record["failure"] = "net::ERR_CONNECTION_REFUSED"
    elif fault == "http_error":
        files["browser_errors.json"]["http_errors"] = [{"url": record["url"], "status": 503}]
    elif fault == "missing_method":
        del record["method"]
    elif fault == "wrong_plan":
        files["map_asset_sources.json"]["map_dataset"]["renderedMapPlanId"] = "wrong"
    elif fault == "incomplete_tiles":
        files["map_asset_sources.json"]["map_dataset"]["mapTileCoverage"] = "0.94"
    elif fault == "omitted_record":
        files["network_classification.json"]["classifications"] = []
    elif fault == "wrong_archive":
        files["map_asset_sources.json"]["vector_sources"]["base"]["url"] = "pmtiles://" + network_fixture.MAP + "/data/other.pmtiles"
    elif fault == "console_error":
        files["browser_errors.json"]["console_errors"] = ["Map failed"]
    elif fault == "provider_call":
        files["llm_io.json"]["provider_call_count"] = 1
    with pytest.raises(core.E2E1VerificationError):
        _network_run(tmp_path, files, parent)
