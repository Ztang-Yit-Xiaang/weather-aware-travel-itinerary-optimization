"""Independently bind real UI remount observations to the evaluated preview."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _proposal

SCENARIO = "same_plan_remount"
TEST = "test_ui_remount_withdraws_acceptance_until_same_plan_is_ready"
CONTROLS = (("select_remount", "#map-select-button"), ("edit_remount", "#map-edit-button"))
ACK_FIELDS = {"mapReadyAt", "mapTileCoverage", "mapCoverageMethod", "renderedMapPlanId",
    "renderedPlanContentHash", "renderedGeometryHash", "renderedMarkerIds"}
STAGE_ACK_FIELDS = {"renderedMapPlanId", "renderedGeometryHash", "renderedMarkerIds"}


def require(condition, reason):
    core._require(condition, "e2e1_remount_" + reason)


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp_timezone_missing")
    return result


def verify_observation(observation, label, control, ack, request_id):
    require(observation["label"] == label and observation["overflow"] is False, "observer_invalid")
    rows = observation["rows"]
    require(len(rows) >= 4 and rows[0]["reason"] == "armed" and rows[-1]["reason"] == "stopped"
        and [row["sequence"] for row in rows] == list(range(1, len(rows) + 1)), "sequence_incomplete")
    times = [stamp(row["observed_at"]) for row in rows]
    monotonic = [row["performance_ms"] for row in rows]
    require(times == sorted(times) and all(math.isfinite(value) for value in monotonic)
        and monotonic == sorted(monotonic), "observation_order_invalid")
    trusted = [row for row in rows if row["reason"] == "trusted_control_event"]
    require(len(trusted) == 1 and trusted[0]["detail"] == {
        "trusted": True, "target_id": control.removeprefix("#")}, "control_not_trusted")
    previous = rows[0]["previous_map_id"]
    require(previous is not None and rows[0]["current_map_id"] == previous
        and rows[0]["previous_connected"] is True
        and all(row["previous_map_id"] == previous and row["path"] == "/app/itinerary" for row in rows),
        "previous_map_identity_invalid")
    armed = rows[0]
    require(armed["current_map"]["mapReady"] == "true"
        and armed["current_map"]["renderedPlanContentHash"] == ack["active_plan_content_hash"]
        and armed["stage"]["previewState"] == "PREVIEW_RENDERED"
        and armed["stage"]["activePlanId"] == ack["active_plan_id"]
        and armed["stage"]["activePlanContentHash"] == ack["active_plan_content_hash"]
        and armed["stage"]["previewRequestId"] == request_id
        and armed["accept"] == {"present": True, "disabled": False, "visible": True}, "armed_preview_invalid")
    for dataset in (armed["current_map"], armed["stage"]):
        require(dataset["renderedMapPlanId"] == ack["rendered_map_plan_id"]
            and dataset["renderedGeometryHash"] == ack["rendered_geometry_hash"]
            and sorted(json.loads(dataset["renderedMarkerIds"])) == sorted(ack["rendered_marker_ids"]),
            "armed_identity_invalid")
    # Audit the entire post-click interval, including a missing map or an
    # invalidated old container before the replacement element exists.
    for row in rows[trusted[0]["sequence"]:]:
        current = row["current_map"]
        if current is not None and current.get("mapReady") == "true":
            continue
        require((current is None or current.get("mapReady") == "false")
            and not ACK_FIELDS.intersection(current or {})
            and not ACK_FIELDS.intersection(row["previous_map"])
            and not STAGE_ACK_FIELDS.intersection(row["stage"])
            and row["stage"]["previewState"] == "PREVIEW_DATA_READY"
            and row["stage"]["activePlanId"] == ack["active_plan_id"]
            and row["stage"]["activePlanContentHash"] == ack["active_plan_content_hash"]
            and row["stage"]["previewRequestId"] == request_id
            and row["accept"] == {"present": True, "disabled": True, "visible": True},
            "transitional_acceptance_or_ack")
    replaced = [row for row in rows if row["current_map_id"] is not None and row["current_map_id"] != previous]
    require(replaced and len({row["current_map_id"] for row in replaced}) == 1
        and replaced[0]["sequence"] > trusted[0]["sequence"], "map_not_replaced")
    pending = [row for row in replaced if row["current_map"].get("mapReady") != "true"]
    require(pending, "loading_interval_missing")
    for row in replaced:
        require(row["previous_connected"] is False and row["current_connected"] is True
            and not ACK_FIELDS.intersection(row["previous_map"]), "disposed_ack_survives")
        stage = row["stage"]
        require(stage["activePlanId"] == ack["active_plan_id"]
            and stage["activePlanContentHash"] == ack["active_plan_content_hash"]
            and stage["previewRequestId"] == request_id, "active_preview_changed")
    for row in pending:
        require(row["current_map"]["mapReady"] == "false"
            and not ACK_FIELDS.intersection(row["current_map"])
            and not STAGE_ACK_FIELDS.intersection(row["stage"])
            and row["stage"]["previewState"] == "PREVIEW_DATA_READY"
            and row["accept"] == {"present": True, "disabled": True, "visible": True}, "premature_acceptance_or_ack")
    ready = [row for row in replaced if row["current_map"].get("mapReady") == "true"]
    require(ready and ready[-1] == rows[-1] and ready[0]["sequence"] > pending[-1]["sequence"], "ready_order_invalid")
    for row in ready:
        data, stage = row["current_map"], row["stage"]
        coverage = float(data["mapTileCoverage"])
        require(math.isfinite(coverage) and .95 <= coverage <= 1
            and data["mapCoverageMethod"] == "loaded_tile_projected_union_area_v1"
            and stamp(pending[-1]["observed_at"]) <= stamp(data["mapReadyAt"]) <= stamp(row["observed_at"]),
            "ready_tile_gate_invalid")
        require(data["renderedPlanContentHash"] == ack["active_plan_content_hash"]
            and stage["previewState"] == "PREVIEW_RENDERED"
            and row["accept"] == {"present": True, "disabled": False, "visible": True}, "ready_state_invalid")
        for dataset in (data, stage):
            require(dataset["renderedMapPlanId"] == ack["rendered_map_plan_id"]
                and dataset["renderedGeometryHash"] == ack["rendered_geometry_hash"]
                and sorted(json.loads(dataset["renderedMarkerIds"])) == sorted(ack["rendered_marker_ids"]),
                "rendered_identity_changed")
    return len(pending)


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require(len(cases) == 1 and cases[0].get("name") == TEST
        and cases[0].get("classname") == "tests.product_app.browser.test_same_plan_remount_live"
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()), "test_inventory_invalid")
    base = root / SCENARIO
    assertions = core._load(base / "browser_assertions.json")
    require(assertions["schema_version"] == "same-plan-remount-browser-v1"
        and assertions["assertions_passed"] is True and assertions["observer_is_read_only"] is True
        and assertions["api_or_digest_or_readiness_injection"] is False
        and assertions["acceptance_performed"] is False, "assertions_invalid")
    records = _http(base, base_url)
    posts = [row for row in records if row["method"] == "POST"]
    previews = [row for row in posts if row["url"].endswith("/preview")]
    stages = [row for row in posts if row["url"].endswith("/draft/operations")]
    require(len(previews) == len(stages) == 1
        and not any(row["url"].endswith(("/accept", "/copilot/messages", "/draft/undo")) for row in posts),
        "mutation_universe_invalid")
    payload = core._load(base / "preview_response.json")
    require(payload == json.loads(previews[0]["response_body_utf8"]), "preview_http_unbound")
    proposal, ack = payload["proposal"], core._load(base / "map_render_ack.json")
    _proposal(proposal, ack)
    operation = json.loads(stages[0]["request_body_utf8"])
    staged = json.loads(stages[0]["response_body_utf8"])
    expected = {"type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}, "source": "map"}
    require(all(operation[key] == staged["operation"][key] == value for key, value in expected.items())
        and payload["session"]["draft"] == [staged["operation"]]
        and payload["session"]["accepted_plan_id"] == proposal["parent_plan_id"], "draft_or_parent_unbound")
    execution, geo = proposal["runtime_execution"], proposal["geography_plan"]
    require(core._load(base / "evaluation_summary.json") == execution["evaluation_certificate"]
        and core._load(base / "plan_diff.json") == execution["plan_diff"]
        and geo["plan_id"] == proposal["child_plan_id"]
        and geo["content_hash"] == proposal["child_plan_content_hash"]
        and ack["rendered_geometry_hash"] == core._completed_race_geometry_hash(geo)
        and sorted(ack["rendered_marker_ids"]) == sorted(row["properties"]["stop_id"] for row in geo["stops"]["features"]),
        "canonical_artifacts_mismatch")
    require(assertions["child_plan_id"] == proposal["child_plan_id"]
        and assertions["child_plan_content_hash"] == proposal["child_plan_content_hash"]
        and assertions["request_id"] == proposal["compiled_request"]["source_request_id"]
        and assertions["evaluation_id"] == proposal["certificate_id"]
        and assertions["geometry_hash"] == ack["rendered_geometry_hash"]
        and assertions["parent"]["active_plan_id"] == assertions["parent"]["rendered_map_plan_id"] == proposal["parent_plan_id"],
        "trace_identity_invalid")
    checkpoints = assertions["checkpoints"]
    require(checkpoints == json.loads((base / "completed_checkpoints.json").read_text(encoding="utf-8"))
        and [row["control"] for row in checkpoints] == [control for _, control in CONTROLS], "checkpoint_inventory_invalid")
    for checkpoint, (label, control) in zip(checkpoints, CONTROLS, strict=True):
        require(checkpoint["observation"] == label + "_observer.json" and checkpoint["ready_ack"] == ack
            and checkpoint["no_mutation_requests"] is True and checkpoint["assertions_passed"] is True,
            "checkpoint_invalid")
        observation = core._load(base / checkpoint["observation"])
        require(checkpoint["pending_samples"] == verify_observation(observation, label, control, ack, execution["request_id"]),
            "pending_count_invalid")
        require(not any(row["phase"] == label and row["url"].endswith(("/draft/operations", "/preview", "/accept")) for row in posts),
            "remount_mutated_plan")
        capture = core._load(base / (label + "_ready.readiness.json"))
        require(stamp(observation["rows"][0]["observed_at"]) <= stamp(capture["screenshot_started_at"])
            <= stamp(capture["screenshot_completed_at"]) <= stamp(observation["rows"][-1]["observed_at"]), "capture_outside_observer")
    labels = ("parent_ready", "preview_ready_before_remount", "select_remount_ready", "edit_remount_ready")
    core._verify_journey_captures(base, labels, {proposal["parent_plan_id"], proposal["child_plan_id"]})
    require({path.stem for path in base.glob("*.png")} == set(labels)
        and json.loads((base / "console_errors.json").read_text(encoding="utf-8"))
        == json.loads((base / "network_failures.json").read_text(encoding="utf-8")) == [], "browser_errors_or_extra_capture")
    return {"verified": True, "scenario": SCENARIO, "test_count": 1, "remount_count": 2,
        "child_plan_id": proposal["child_plan_id"], "provider_calls": 0}


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_remount_evidence_malformed") from error
