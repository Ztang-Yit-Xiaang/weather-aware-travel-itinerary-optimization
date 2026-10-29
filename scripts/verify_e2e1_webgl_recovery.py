"""Independent real-WebGL fault, canonical renderer, and recovery evidence checks."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlsplit

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _proposal

SCENARIO = "webgl_recovery"
TEST = "test_real_webgl_loss_withdraws_ack_and_restores_exact_evaluated_child"
CONTROLLER = "src/itinerary_system/product_app/static/js/map-controller-v2.js"
SOURCE_PATHS = (CONTROLLER, "tests/product_app/browser/test_webgl_recovery_live.py",
    "tests/product_app/browser/test_e2e1_live_copilot_chat.py", "tests/product_app/browser/waypoint_ui_fixture.py")
LABELS = ("before_context_loss_before_capture", "before_context_loss_after_capture", "context_lost_immediate",
    "context_lost_after_1200ms", "context_restored_event", "after_context_restoration_before_capture",
    "after_context_restoration_after_capture")


def require(condition, reason):
    core._require(condition, "e2e1_webgl_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp_timezone_missing")
    return result


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def withdrawn(observation, *, lost, state, child_id):
    require(observation["context_lost"] is lost and observation["map_dataset"]["mapReady"] == "false"
        and not set(observation["map_dataset"]) & {"mapReadyAt", "mapTileCoverage", "mapCoverageMethod",
            "renderedMapPlanId", "renderedPlanContentHash", "renderedGeometryHash", "renderedMarkerIds"}
        and not set(observation["stage_dataset"]) & {"renderedMapPlanId", "renderedGeometryHash", "renderedMarkerIds"}
        and observation["stage_dataset"]["previewState"] == state
        and observation["stage_dataset"]["activePlanId"] == child_id, "stale_render_acknowledgement")


def renderer(observation, geography, request_id):
    expected_hash = core._completed_race_geometry_hash(geography)
    expected_markers = sorted(row["properties"]["stop_id"] for row in geography["stops"]["features"])
    require(observation["context_lost"] is False and observation["map_loaded"] is True
        and observation["all_tiles_loaded"] is True and observation["map_dataset"]["mapReady"] == "true",
        "renderer_not_ready")
    coverage = float(observation["map_dataset"]["mapTileCoverage"])
    require(math.isfinite(coverage) and .95 <= coverage <= 1
        and observation["map_dataset"]["mapCoverageMethod"] == "loaded_tile_projected_union_area_v1"
        and stamp(observation["map_dataset"]["mapReadyAt"]) <= stamp(observation["observed_at"]), "tile_readiness_invalid")
    for dataset in (observation["map_dataset"], observation["stage_dataset"]):
        require(dataset["renderedMapPlanId"] == geography["plan_id"]
            and dataset["renderedGeometryHash"] == expected_hash
            and json.loads(dataset["renderedMarkerIds"]) == expected_markers, "rendered_identity_mismatch")
    require(observation["map_dataset"]["renderedPlanContentHash"] == geography["content_hash"]
        and observation["stage_dataset"]["activePlanId"] == geography["plan_id"]
        and observation["stage_dataset"]["activePlanContentHash"] == geography["content_hash"]
        and observation["stage_dataset"]["previewState"] == "PREVIEW_RENDERED"
        and observation["stage_dataset"]["previewRequestId"] == request_id, "active_preview_identity_mismatch")
    for source, field in (("v2-routes", "validated_legs"), ("v2-stops", "stops")):
        features = observation["sources"][source]["data"]["features"]
        active = [row for row in features if row["properties"].get("display_role") == "active"]
        require(active and all(row["properties"]["plan_id"] == geography["plan_id"] for row in active), "foreign_active_source")
        child = [row for row in features if row["properties"]["plan_id"] == geography["plan_id"]]
        normalized = [{**row, "properties": {key: value for key, value in row["properties"].items() if key != "display_role"}}
                      for row in child]
        require(normalized == geography[field]["features"] and child == active, "canonical_source_mismatch")
    active = [row for row in observation["rendered_features"] if row["layer"] in {"v2-route-active", "v2-stops-active"}]
    require({row["layer"] for row in active} == {"v2-route-active", "v2-stops-active"}
        and all(row["properties"]["plan_id"] == geography["plan_id"]
            and row["properties"]["content_hash"] == geography["content_hash"] for row in active), "actual_rendered_features_mismatch")
    visible = {row["properties"]["stop_id"] for row in active if row["layer"] == "v2-stops-active"}
    require("pfeiffer_big_sur_state_park" in visible and "bixby_creek_bridge_viewpoint" not in visible,
        "replacement_not_visibly_rendered")


def graphics_sequence(observations, faults, geography, request_id):
    require(len(observations) == len(LABELS) and len(faults) == 2, "graphics_sequence_incomplete")
    times = [stamp(row["observed_at"]) for row in observations]
    require(times == sorted(times), "observation_order_invalid")
    for index, action in enumerate(("loseContext", "restoreContext")):
        row = faults[index]
        require(row["action"] == "real_WEBGL_lose_context." + action
            and row["javascript"] == "window.__e2eWebglRecovery.extension." + action + "()"
            and row["api_response_mocking"] is False and row["synthetic_event_dispatch"] is False,
            "synthetic_graphics_fault")
    require(times[1] <= stamp(faults[0]["at"]) <= times[2]
        and (times[3] - times[2]).total_seconds() >= 1.2
        and times[3] <= stamp(faults[1]["at"]) <= times[4], "fault_timing_invalid")
    before, after = observations[0], observations[5]
    for index in (0, 1, 5, 6):
        renderer(observations[index], geography, request_id)
    for index in (2, 3):
        row = observations[index]
        withdrawn(row, lost=True, state="MAP_RENDER_FAILED", child_id=geography["plan_id"])
        require(row["map_loaded"] is False and row["all_tiles_loaded"] is False and row["sources"] == {}
            and row["layers"] == row["rendered_features"] == [] and row["status"]["hidden"] is False,
            "lost_context_still_claims_renderer")
    events = observations[6]["events"]
    require(len(events) == 2 and [event["type"] for event in events] == ["webglcontextlost", "webglcontextrestored"]
        and all(event["trusted"] is True for event in events), "graphics_events_not_trusted")
    withdrawn(events[0], lost=True, state="MAP_RENDER_FAILED", child_id=geography["plan_id"])
    withdrawn(events[1], lost=False, state="PREVIEW_DATA_READY", child_id=geography["plan_id"])
    require(observations[0]["events"] == observations[1]["events"] == []
        and observations[2]["events"] == observations[3]["events"] == events[:1]
        and observations[4]["events"] == observations[5]["events"] == events
        and stamp(faults[0]["at"]) <= stamp(events[0]["at"]) <= times[2]
        and stamp(faults[1]["at"]) <= stamp(events[1]["at"]) <= times[4]
        and stamp(after["map_dataset"]["mapReadyAt"]) >= stamp(events[1]["at"]), "event_or_ready_order_invalid")
    for first, last in ((observations[0], observations[1]), (observations[5], observations[6])):
        require(all(first[field] == last[field] for field in ("map_dataset", "stage_dataset", "sources", "layers", "rendered_features", "events")),
            "renderer_changed_during_capture")
    require(before["sources"] == after["sources"] and before["layers"] == after["layers"]
        and sorted(map(json_hash, before["rendered_features"])) == sorted(map(json_hash, after["rendered_features"])),
        "renderer_not_restored_exactly")
    return after


def _http(base, base_url):
    index = load(base / "http_index.json")
    require(isinstance(index, list) and len(index) == len({row["file"] for row in index})
        and {row["file"] for row in index} == {path.name for path in base.glob("http_*.json") if path.name != "http_index.json"}, "http_inventory_invalid")
    records = []
    for item in index:
        row = core._load(core._safe_path(base, item["file"]))
        core._verify_product_http_capture(row)
        require(all(row[key] == item[key] for key in ("phase", "method", "url"))
            and row["response_status_code"] == item["status"] == 200 and row["url"].startswith(base_url + "/")
            and (urlsplit(row["url"]).path.startswith("/api/") or urlsplit(row["url"]).path == "/static/js/map-controller-v2.js"), "http_capture_invalid")
        records.append((item["file"], row))
    return records


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require(len(cases) == 1 and cases[0].get("name") == TEST
        and cases[0].get("classname") == "tests.product_app.browser.test_webgl_recovery_live"
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()), "test_inventory_invalid")
    base = root / SCENARIO
    assertions = core._load(base / "browser_assertions.json")
    require(assertions["schema_version"] == "webgl-recovery-browser-v1"
        and all(assertions[key] is True for key in ("assertions_passed", "real_context_loss", "real_context_restoration",
            "trusted_graphics_events", "ack_withdrawn_immediately", "ack_stays_withdrawn_1200ms", "restoration_waits_for_real_ready",
            "renderer_matches_backend_child", "renderer_exact_before_after", "no_screenshot_while_lost"))
        and all(assertions[key] is False for key in ("accept_called", "api_response_mocking", "graphics_implementation_mocking")), "assertions_invalid")
    records = _http(base, base_url)
    posts = [row for _, row in records if row["method"] == "POST"]
    require(not any(row["url"].endswith(("/accept", "/copilot/messages")) for row in posts), "unexpected_accept_or_chat")
    previews = [row for row in posts if row["url"].endswith("/preview")]
    stages = [row for row in posts if row["url"].endswith("/draft/operations")]
    require(len(previews) == len(stages) == 1, "mutation_universe_invalid")
    payload = json.loads(previews[0]["response_body_utf8"])
    proposal = core._load(base / "preview_response.json")
    ack = core._load(base / "map_render_ack.json")
    require(payload["proposal"] == proposal and payload["session"]["accepted_plan_id"] == proposal["parent_plan_id"], "preview_http_unbound")
    _proposal(proposal, ack)
    staged = json.loads(stages[0]["response_body_utf8"])
    operation = json.loads(stages[0]["request_body_utf8"])
    require(operation["type"] == "replace_nearby" and operation["target"] == "bixby_creek_bridge_viewpoint"
        and operation["parameters"] == {"candidate_id": "pfeiffer_big_sur_state_park"}
        and all(staged["operation"][key] == operation[key] for key in ("type", "target", "parameters"))
        and payload["session"]["draft"] == [staged["operation"]], "replacement_operation_unbound")
    execution = proposal["runtime_execution"]
    geo = core._load(base / "route_geometry.json")
    evaluation = core._load(base / "evaluation_summary.json")
    diff = core._load(base / "plan_diff.json")
    require(geo == proposal["geography_plan"] and evaluation == execution["evaluation_certificate"] and diff == execution["plan_diff"]
        and evaluation["eligible"] is True and evaluation["failure_count"] == 0
        and evaluation["route_validation"]["fallback_leg_count"] == evaluation["route_validation"]["missing_leg_count"] == 0
        and [row["stop_id"] for row in diff["added_stops"]] == ["pfeiffer_big_sur_state_park"]
        and [row["stop_id"] for row in diff["deleted_stops"]] == ["bixby_creek_bridge_viewpoint"], "canonical_artifacts_mismatch")
    require(assertions["request_id"] == execution["request_id"] and assertions["parent_plan_id"] == proposal["parent_plan_id"]
        and assertions["child_plan_id"] == assertions["rendered_map_plan_id"] == ack["active_plan_id"]
        and assertions["child_plan_content_hash"] == ack["active_plan_content_hash"]
        and assertions["evaluation_id"] == execution["evaluation_id"] == evaluation["certificate_id"]
        and assertions["comparison_eligibility"] == "eligible" and assertions["geometry_hash"] == ack["rendered_geometry_hash"]
        and assertions["rendered_marker_ids"] == ack["rendered_marker_ids"], "trace_identity_mismatch")
    checkpoints = load(base / "checkpoints.json")
    require([row["label"] for row in checkpoints] == list(LABELS), "checkpoint_universe_invalid")
    observations = []
    for row in checkpoints:
        require(row["artifact"] == row["label"] + ".json", "checkpoint_path_invalid")
        observation = core._load(core._safe_path(base, row["artifact"]))
        require(observation["observed_at"] == row["observed_at"], "checkpoint_projection_invalid")
        observations.append(observation)
    after = graphics_sequence(observations, load(base / "graphics_fault_inputs.json"), geo, execution["request_id"])
    require(assertions["sources_sha256"] == json_hash(after["sources"])
        and assertions["layers_sha256"] == json_hash(after["layers"])
        and assertions["rendered_features_sha256"] == json_hash(sorted(map(json_hash, after["rendered_features"]))), "renderer_hash_mismatch")
    core._verify_journey_captures(base, ("initial_parent", "before_context_loss", "after_context_restoration"),
        {proposal["parent_plan_id"], proposal["child_plan_id"]})
    require({path.stem for path in base.glob("*.png")} == {"initial_parent", "before_context_loss", "after_context_restoration"}, "unexpected_screenshot")
    for label, first, last in (("before_context_loss", observations[0], observations[1]), ("after_context_restoration", observations[5], observations[6])):
        ready = core._load(base / (label + ".readiness.json"))
        require(stamp(first["observed_at"]) <= stamp(ready["screenshot_started_at"])
            <= stamp(ready["screenshot_completed_at"]) <= stamp(last["observed_at"]), "capture_not_bracketed_by_renderer")
    source = core._load(base / "source_identity.json")
    expected = [{"path": path, "sha256": core._sha256(core.ROOT / path), "size_bytes": (core.ROOT / path).stat().st_size} for path in SOURCE_PATHS]
    health_record = core._load(base / "preflight_health.json")
    require(health_record["method"] == "GET" and health_record["url"] == base_url + "/api/health"
        and health_record["response_status"] == 200
        and hashlib.sha256(health_record["response_body_utf8"].encode()).hexdigest() == health_record["response_body_sha256"], "health_http_invalid")
    health = json.loads(health_record["response_body_utf8"])
    require(source["schema_version"] == "webgl-browser-source-identity-v1" and source["source_unchanged"] is True
        and source["inputs_before"] == source["inputs_after"] == expected
        and source["html_build_id"] == source["health_build_id"] == health["build_id"]
        and all(row["request_product_build_id"] == health["build_id"] for row in posts), "source_build_binding_invalid")
    served = [(name, row) for name, row in records if urlsplit(row["url"]).path.endswith("/map-controller-v2.js")]
    require(len(served) == 1 and served[0][0] == source["served_controller_http"]
        and served[0][1]["response_body_sha256"] == expected[0]["sha256"], "served_controller_mismatch")
    errors = core._load(base / "browser_errors.json")
    require(errors["page_errors"] == errors["console_errors"] == errors["request_failures"] == [], "browser_errors")
    return {"verified": True, "scenario": SCENARIO, "test_count": 1, "graphics_fault": "real_WEBGL_lose_context",
        "child_plan_id": proposal["child_plan_id"], "provider_calls": 0}


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_webgl_evidence_malformed") from error
