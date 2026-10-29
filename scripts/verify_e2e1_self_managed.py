"""Independent evidence checks for browser-owned sequential server lifecycles."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import verify_e2e1_vertical_slice as core
from e2e1_runtime_environment import validate_runtime_environment

CONVERSATION_CASES = {
    "clarification_reload": "test_clarification_survives_reload_and_short_answer_renders_exact_child",
    "combined_chat": "test_combined_chat_stages_all_edits_and_renders_exact_child",
    "combined_chat_one_click": "test_combined_chat_continuation_and_one_click[original-one_click]",
    "combined_chat_accepted_child": "test_combined_chat_continuation_and_one_click[accepted_child-staged]",
    "combined_chat_one_click_accepted_child": "test_combined_chat_continuation_and_one_click[accepted_child-one_click]",
    "preference_memory": "test_preference_survives_seven_turns_corrects_and_resets_without_plan_change",
    "capacity_turns": "test_capacity_blocks_real_chat_before_dispatch_and_settings_reset_recovers[turns]",
    "capacity_bytes": "test_capacity_blocks_real_chat_before_dispatch_and_settings_reset_recovers[bytes]",
    "fake_sdk_cancel": "test_fake_sdk_stop_waiting_reload_and_retry_reconciles_one_attempt",
    "selection_conversation_generation": "test_late_selection_preserves_deleted_conversation_generation",
    "preference_memory_accepted_child": "test_accepted_child_conversation_controls[delete_all]",
    "capacity_turns_accepted_child": "test_accepted_child_conversation_controls[capacity_turns]",
    "capacity_bytes_accepted_child": "test_accepted_child_conversation_controls[capacity_bytes]",
    "fake_sdk_cancel_accepted_child": "test_accepted_child_conversation_controls[recovery]",
}
BUILD_TEST = "test_cached_page_requires_reload_and_preserves_active_plan_across_real_restart"
COMBINED_CONFLICT_CASES = {
    "combined_conflicts"
    + (
        "_accepted_child" if context == "accepted_child" else ""
    ): f"test_conflicting_combined_chat_never_offers_or_stages_partial_edits[{context}]"
    for context in ("original", "accepted_child")
}
QUEUED_PREVIEW_CASES = {
    "combined_chat_async_delete_"
    + kind
    + (
        "_accepted_child" if context == "accepted_child" else ""
    ): f"test_deleted_chat_cannot_resume_queued_preview[{context}-{kind}]"
    for context in ("original", "accepted_child")
    for kind in ("single", "combined")
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, code):
    core._require(condition, "e2e1_self_" + code)


def stamp(value):
    return core._isolation_timestamp(value)


def _junit(root, mode):
    tree = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(tree.iter("testcase"))
    filename = "test_build_coherence_live" if mode == "build-coherence" else "test_copilot_conversation_journeys_live"
    names = (
        [BUILD_TEST + "[" + context + "]" for context in ("original", "accepted_child")]
        if mode == "build-coherence"
        else list(CONVERSATION_CASES.values())
    )
    if mode == "bootstrap-recovery":
        from verify_e2e1_bootstrap_recovery import CASES, TEST

        filename = "test_bootstrap_recovery_live"
        names = [TEST + "[" + case + "]" for case in CASES]
    if mode == "optional-evidence-fault":
        from verify_e2e1_optional_evidence_fault import TEST

        filename = "test_optional_evidence_fault_live"
        names = [TEST]
    if mode == "preview-geography-fault":
        from verify_e2e1_preview_geography_fault import CASES, TEST

        filename = "test_preview_geography_fault_live"
        names = [TEST + "[" + kind + "]" for kind in CASES]
    if mode == "combined-conflicts":
        filename = "test_copilot_combined_conflicts_live"
        names = list(COMBINED_CONFLICT_CASES.values())
    if mode == "queued-preview":
        filename = "test_copilot_queued_preview_live"
        names = list(QUEUED_PREVIEW_CASES.values())
    if mode == "owned-edits":
        from verify_e2e1_owned_edits import CASES

        filename = "test_owned_combined_journey_live"
        names = list(CASES.values())
    require(
        sorted(case.get("name", "") for case in cases) == sorted(names)
        and all(case.get("classname") == "tests.product_app.browser." + filename for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in tree.iter()),
        "junit_inventory_invalid",
    )
    return {"tests/product_app/browser/" + filename + ".py::" + name for name in names}


def _lifecycle(base, run_id, process, *, fake=False, child_process=False):
    receipt = core._load(base / "provider_isolation.json")
    lifecycle = core._load(base / "server_lifecycle.json")
    require(
        receipt["run_id"] == lifecycle["run_id"] == run_id
        and type(receipt["process_id"]) is int
        and receipt["process_id"] == lifecycle["pid"]
        and (child_process or receipt["process_id"] == process["pid"]),
        "lifecycle_identity_invalid",
    )
    require(
        all(
            receipt.get(key) is True
            for key in ("finalized", "api_key_disabled", "app_initialized", "server_started", "server_stopped")
        )
        and receipt.get("failure_code") is None
        and receipt["scope"] == "application_server"
        and receipt["guard_boundaries"] == core.ISOLATION_BOUNDARIES
        and type(receipt["sent_provider_calls"]) is int
        and receipt["sent_provider_calls"] == 0
        and receipt["blocked_attempts"] == []
        and type(receipt["allowed_local_connections"]) is int
        and receipt["allowed_local_connections"] > 0,
        "lifecycle_isolation_incomplete",
    )
    require(
        stamp(process["started_at"])
        <= stamp(receipt["installed_at"])
        <= stamp(receipt["finalized_at"])
        <= stamp(process["completed_at"]),
        "lifecycle_times_invalid",
    )
    if fake:
        require(
            receipt["schema_version"] == "conversation-browser-fake-sdk-isolation-v1"
            and receipt["actual_adapter"] == "OpenAICopilotAdapter"
            and receipt["effective_adapter"] == "openai"
            and receipt["credential_mode"] == "synthetic_unusable_fixture_value"
            and receipt["external_transport"] == "httpx.MockTransport"
            and receipt["actual_sdk"] == "openai.AsyncOpenAI"
            and receipt["sdk_max_retries"] == 0
            and receipt["live_openai_test"] is False
            and receipt["fake_provider_endpoint"] == "http://127.0.0.1:9/v1/responses"
            and lifecycle["fake_sdk_attempt_count"] == 1
            and lifecycle["external_transport_mocked_only"] is True,
            "fake_transport_scope_invalid",
        )
    else:
        require(
            receipt["schema_version"] == "e2e1-provider-isolation-v1"
            and receipt["effective_adapter"] == "deterministic"
            and receipt["actual_adapter"] == "DeterministicCopilotAdapter",
            "deterministic_scope_invalid",
        )
    if child_process:
        require(
            lifecycle["exit_code"] == 0
            and lifecycle["isolation_verified"] is True
            and lifecycle["receipt_sha256"] == core._sha256(base / "provider_isolation.json")
            and stamp(lifecycle["process_started_at"])
            <= stamp(receipt["installed_at"])
            <= stamp(lifecycle["ready_at"])
            <= stamp(receipt["finalized_at"])
            <= stamp(lifecycle["process_stopped_at"]),
            "child_lifecycle_invalid",
        )
    else:
        require(
            lifecycle["thread_alive_after_shutdown"] is False
            and lifecycle["api_response_mocking"] is False
            and lifecycle["live_provider_call_count"] == 0,
            "thread_lifecycle_invalid",
        )
    return receipt


def _runtime(base, receipt, nodes, base_url):
    referenced, observed = set(), set()
    captures = list(base.glob("*.readiness.json"))
    require(captures, "runtime_capture_missing")
    for path in captures:
        ready = core._load(path)
        binding = ready["runtime_environment"]
        node = binding["test_nodeid"]
        name = "runtime_environment_" + hashlib.sha256(node.encode()).hexdigest()[:16] + ".json"
        require(node in nodes and binding["artifact"] == name, "runtime_node_binding_invalid")
        runtime_path = core._safe_path(base, name)
        record = core._load(runtime_path)
        validate_runtime_environment(record, run_id=receipt["run_id"], root=core.ROOT)
        require(
            binding["sha256"] == core._sha256(runtime_path)
            and binding["browser"] == record["browser"]
            and record["application_origin"] == base_url
            and stamp(receipt["installed_at"])
            <= stamp(record["observed_at"])
            <= stamp(record["completed_at"])
            <= stamp(ready["screenshot_started_at"])
            <= stamp(ready["screenshot_completed_at"])
            <= stamp(receipt["finalized_at"]),
            "runtime_lifecycle_binding_invalid",
        )
        referenced.add(runtime_path)
        observed.add(node)
    require(referenced == set(base.glob("runtime_environment_*.json")), "runtime_file_universe_invalid")
    return observed


def _http(base, base_url, *, statuses=(200,), allow_partial=False):
    index = load(base / "http_index.json")
    require(
        isinstance(index, list)
        and all(isinstance(row, dict) and {"file", "method", "url", "status"} <= row.keys() for row in index),
        "http_index_invalid",
    )
    names = [row["file"] for row in index]
    require(
        all(isinstance(name, str) for name in names)
        and len(names) == len(set(names))
        and set(names) == {path.name for path in base.glob("http_*.json") if path.name != "http_index.json"},
        "http_inventory_invalid",
    )
    records = []
    for item in index:
        record = core._load(core._safe_path(base, item["file"]))
        core._verify_product_http_capture(record)
        require(
            record["url"] == item["url"]
            and record["url"].startswith(base_url + "/api/")
            and record["method"] == item["method"]
            and record["response_status_code"] == item["status"]
            and record["response_status_code"] in statuses,
            "http_origin_status_invalid",
        )
        if "response_capture_complete" in item:
            require(
                record.get("response_capture_complete") is item["response_capture_complete"]
                and type(item["response_capture_complete"]) is bool
                and (allow_partial or item["response_capture_complete"]),
                "http_capture_scope_invalid",
            )
        records.append(record)
    return records


def _proposal(proposal, ack, *, combined=False):
    require(
        proposal["state"] == "eligible"
        and proposal["execution_mode"]
        == ("atomic_combined_independent_evaluation" if combined else "deterministic_repair_pipeline"),
        "preview_not_eligible",
    )
    execution = proposal["runtime_execution"]
    child, certificate, diff = execution["child_plan"], execution["evaluation_certificate"], execution["plan_diff"]
    require(
        execution["evaluation_executed"] is True
        and child["plan_id"] == certificate["plan_id"] == diff["child_plan_id"] == proposal["child_plan_id"]
        and child["content_hash"] == certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
        and child["content_hash"] == core._artifact_content_hash(child)
        and certificate["content_hash"]
        == core._artifact_content_hash(certificate)
        == proposal["certificate_content_hash"]
        and certificate["certificate_id"] == proposal["certificate_id"]
        and certificate["comparison_eligibility"] == "eligible"
        and core._artifact_content_hash(diff) == proposal["diff_content_hash"]
        and child["parent_plan_id"] == diff["parent_plan_id"] == proposal["parent_plan_id"]
        and execution["request_id"] == child["planning_request_id"] == execution["repair_request"]["request_id"],
        "canonical_preview_identity_invalid",
    )
    core._verify_plan_ack(ack, proposal)


def _same_plan(first, second):
    return all(
        first.get(key) == second.get(key)
        for key in (
            "active_plan_id",
            "active_plan_content_hash",
            "rendered_map_plan_id",
            "rendered_geometry_hash",
            "rendered_marker_ids",
            "preview_state",
        )
    )


def queued_preview_control_receipt(read, *, name):
    """Bind a delayed real response to deletion, retained draft and unchanged map."""
    require(name in QUEUED_PREVIEW_CASES, "queued_case_invalid")
    continued = name.endswith("_accepted_child")
    combined = name.removesuffix("_accepted_child").endswith("_combined")
    context = "accepted_child" if continued else "original"
    trace, race, turns = (read(path) for path in ("trace.json", "race_io.json", "exact_chat_io.json"))
    records = [read(row["file"]) for row in read("http_index.json")]
    for record in records:
        core._verify_product_http_capture(record)
        require(
            record["response_status_code"] == 200 and record["response_capture_complete"] is True,
            "queued_http_incomplete",
        )
    if continued:
        baseline = read("accepted_baseline_artifacts.json")
        require(
            baseline["context"] == context and _same_plan(baseline["accepted_map"], trace["initial"]),
            "queued_baseline_invalid",
        )
        records = records[len(baseline["http_boundary_files"]) :]
    message = (
        "Set Pfeiffer Big Sur State Park duration to 90 minutes"
        if continued
        else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
    )
    if combined:
        message += (
            "; set Stearns Wharf earliest arrival to 10:00."
            if continued
            else "; set Griffith Observatory duration to 90 minutes."
        )
    chats = [row for row in records if row["url"].endswith("/copilot/messages")]
    require(
        len(chats) == len(turns) == 1
        and trace["scenario"] == name
        and trace["context"] == context
        and trace["combined"] is race["combined"] is combined
        and trace["assertions_passed"] is True
        and trace["accepted"] is False
        and trace["preview_request_count"] == 0
        and race["input"] == message
        and race["preview_requests"] == []
        and race["llm_input"] is race["llm_output"] is None
        and race["provider_attempts"] == 0,
        "queued_input_invalid",
    )
    request, events = _stream(chats[0])
    require(
        turns[0] == {"input": request, "output": events[-1], "events": events}
        and request["message"] == message
        and events[-1]["type"] == "turn.completed"
        and race["turn"] == events[-1]["payload"]["turn"],
        "queued_stream_unbound",
    )
    turn, session = (events[-1]["payload"][key] for key in ("turn", "session"))
    initial = trace["initial"]
    observed = read("context_queued_before_stage.json")
    require(
        initial == race["initial_map"] == race["final_map"] == trace["final"]
        and initial["preview_state"] == "PARENT_RENDERED"
        and initial["rendered_map_plan_id"] == initial["active_plan_id"] == session["accepted_plan_id"]
        and all(
            isinstance(initial[key], str) and initial[key]
            for key in ("active_plan_id", "active_plan_content_hash", "rendered_geometry_hash")
        )
        and initial["rendered_marker_ids"]
        and len(initial["rendered_marker_ids"]) == len(set(initial["rendered_marker_ids"]))
        and observed["accepted_plan_id"] == initial["active_plan_id"]
        and observed["accepted_plan_content_hash"] == initial["active_plan_content_hash"]
        and observed["session_revision"] == session["revision"] == turn["context_revision"]
        and observed["draft_operations"] == session["draft"] == []
        and ("bixby_creek_bridge_viewpoint" in observed["allowed_stop_ids"]) is not continued
        and ("pfeiffer_big_sur_state_park" in observed["allowed_stop_ids"]) is continued,
        "queued_parent_identity_invalid",
    )
    control_records = records[records.index(chats[0]) + 1 :]
    writes = [row for row in control_records if row["method"] not in {"GET", "HEAD"}]
    stages = [row for row in writes if row["url"].endswith("/draft/operations")]
    deletes = [row for row in writes if row["method"] == "DELETE" and row["url"].endswith("/conversation")]
    require(
        len(stages) == len(deletes) == 1
        and all(
            row in stages + deletes or row["method"] == "POST" and row["url"].endswith("/draft/impact-preview")
            for row in writes
        ),
        "queued_mutation_inventory_invalid",
    )
    stage_request = json.loads(stages[0]["request_body_utf8"])
    staged = json.loads(stages[0]["response_body_utf8"])["session"]
    operations = (
        stage_request["operations"]
        if combined
        else [{key: val for key, val in stage_request.items() if key != "expected_revision"}]
    )
    duration = {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}
    expected = (
        [{"type": "set_stop_duration", "target": "pfeiffer_big_sur_state_park", "parameters": duration}]
        if continued
        else [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
            }
        ]
    )
    if combined:
        expected.append(
            {
                "type": "set_stop_time_window",
                "target": "stearns_wharf",
                "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
            }
            if continued
            else {"type": "set_stop_duration", "target": "griffith_observatory", "parameters": duration}
        )
    intents = turn["interpretation"]["intents"]
    require(
        turn["interpretation"]["state"] == "proposal_ready"
        and len(intents) == len(operations) == len(staged["draft"]) == len(expected)
        and stage_request["expected_revision"] == session["revision"]
        and staged["revision"] == session["revision"] + 1
        and [{key: row[key] for key in ("type", "target", "parameters")} for row in operations] == expected,
        "queued_staging_semantics_invalid",
    )
    for intent, operation, canonical in zip(intents, operations, staged["draft"], strict=True):
        parameters = (
            {"candidate_id": intent["candidate_id"]} if intent["type"] == "replace_nearby" else intent["parameters"]
        )
        require(
            operation["source"] == "copilot"
            and intent["type"] == operation["type"]
            and intent["target_stop_id"] == operation["target"]
            and parameters == operation["parameters"]
            and all(canonical[key] == val for key, val in operation.items()),
            "queued_intent_mapping_invalid",
        )
    sid = session["session_id"]
    require(
        all(row["url"].startswith("http://127.0.0.1:8127/api/sessions/" + sid + "/") for row in writes),
        "queued_session_unbound",
    )
    deleted = json.loads(deletes[0]["response_body_utf8"])["session"]
    require(
        json.loads(deletes[0]["request_body_utf8"]) == {"expected_revision": staged["revision"]}
        and staged["conversation_id"] == session["conversation_id"]
        and staged["conversation_id"]
        and deleted["conversation_id"] is None
        and {key: val for key, val in deleted.items() if key != "conversation_id"}
        == {key: val for key, val in staged.items() if key != "conversation_id"},
        "queued_delete_unbound",
    )
    restore = read("reset_restore_http.json")
    require(
        any(
            {key: val for key, val in row.items() if key != "response_capture_complete"} == restore
            and records.index(row) > records.index(deletes[0])
            for row in records
        ),
        "queued_restore_unbound",
    )
    restored = json.loads(restore["response_body_utf8"])
    require(
        restore["method"] == "GET"
        and restore["url"].endswith("/api/sessions/" + sid + "/conversation")
        and restored["conversation"]["turns"] == []
        and restored["conversation"]["conversation_id"]
        == restored["session"]["conversation_id"]
        != session["conversation_id"]
        and {key: val for key, val in restored["session"].items() if key != "conversation_id"}
        == {key: val for key, val in deleted.items() if key != "conversation_id"},
        "queued_restore_invalid",
    )
    reloads = [
        row
        for row in records
        if row["method"] == "GET"
        and row["url"].endswith("/api/sessions/" + sid)
        and records.index(row) > records.index(deletes[0])
    ]
    require(
        len(reloads) == 1 and json.loads(reloads[0]["response_body_utf8"])["session"] == restored["session"],
        "queued_reload_invalid",
    )
    chat_reload = read("reload_conversation_http.json")
    require(
        any(
            {key: val for key, val in row.items() if key != "response_capture_complete"} == chat_reload
            and index > records.index(reloads[0])
            for index, row in enumerate(records)
        )
        and json.loads(chat_reload["response_body_utf8"])["conversation"]["turns"] == [],
        "queued_reload_transcript_invalid",
    )
    for label in ("before_delete", "after_delete", "after_release", "after_reload"):
        value = read("context_" + label + ".json")
        require(
            (label == "after_reload" or value == race[label])
            and value["draft_operations"] == expected
            and value["session_revision"] == staged["revision"]
            and value["evaluated_proposal"] is None
            and all(value[key] == session[key] for key in ("session_id", "trip_id", "run_id", "accepted_plan_id"))
            and value["accepted_plan_content_hash"] == initial["active_plan_content_hash"],
            "queued_context_drift",
        )
    timing = race["timing"]
    held = timing["held"]
    held_records = [
        row
        for row in records
        if row["url"].endswith(held["url"])
        and row["request_body_utf8"] == held["request_body_utf8"]
        and row["response_body_utf8"] == held["response_body_utf8"]
    ]
    require(
        held["method"] == "POST"
        and held["url"].endswith("/draft/impact-preview")
        and held["response_status"] == 200
        and json.loads(held["request_body_utf8"]) == {"expected_revision": staged["revision"]}
        and any(records.index(stages[0]) < records.index(row) < records.index(deletes[0]) for row in held_records)
        and timing["released"] is timing["json_read"] is True,
        "queued_delayed_body_unbound",
    )
    impact = json.loads(held["response_body_utf8"])
    require(
        impact["session_id"] == sid
        and impact["session_revision"] == staged["revision"]
        and impact["parent_plan_id"] == initial["active_plan_id"]
        and impact["parent_plan_content_hash"] == initial["active_plan_content_hash"]
        and impact["summary"]["operation_count"] == len(expected)
        and impact["summary"]["can_run_evaluated_preview"] is True,
        "queued_impact_owner_invalid",
    )
    clicks = read("controls_before_reload.json")
    preview_clicks = [row for row in clicks if row["label"].startswith("Preview ")]
    delete_clicks = [row for row in clicks if row["id"] == "delete-current-conversation"]
    calls = [row for row in timing["calls"] if row["method"] not in {"GET", "HEAD"}]
    require(
        len(preview_clicks) == len(delete_clicks) == 1
        and all(row["trusted"] is True for row in preview_clicks + delete_clicks)
        and preview_clicks[0]["label"] == ("Preview combined repair" if combined else "Preview this repair")
        and len(calls) == 3
        and [row["url"].rsplit("/", 1)[-1] for row in calls] == ["operations", "impact-preview", "conversation"]
        and all(
            row["method"] == actual["method"]
            and actual["url"].endswith(row["url"])
            and row["request_body_utf8"] == actual["request_body_utf8"]
            for row, actual in zip(calls, [stages[0], held_records[0], deletes[0]], strict=True)
        )
        and preview_clicks[0]["at"] / 1000
        <= stamp(calls[0]["dispatch_at"]).timestamp()
        <= stamp(held["dispatch_at"]).timestamp()
        <= stamp(held["held_at"]).timestamp()
        <= delete_clicks[0]["at"] / 1000
        <= stamp(calls[2]["dispatch_at"]).timestamp()
        < stamp(timing["released_at"]).timestamp(),
        "queued_trusted_timing_invalid",
    )
    return {
        "context": context,
        "combined": combined,
        "session_id": sid,
        "parent_plan_id": initial["active_plan_id"],
        "draft_operation_count": len(expected),
        "preview_request_count": 0,
    }


def combined_chat_control_receipt(read, *, name):
    """One complete browser batch, including accepted-baseline continuation.

    Shared by the standalone verifier and recurring action-coverage projection;
    neither receives credit from a screenshot or a trace boolean alone.
    """
    continued = name.endswith("_accepted_child")
    stage_first = "_one_click" not in name
    context = "accepted_child" if continued else "original"
    trace, turns, stage, before, controls, disabled, payload, accepted = (
        read(filename)
        for filename in (
            "trace.json",
            "exact_chat_io.json",
            "staging_io.json",
            "before_preview_observation.json",
            "combined_controls_before_reload.json",
            "combined_added_disabled.json",
            "preview_response.json",
            "acceptance_response.json",
        )
    )
    records = [read(row["file"]) for row in read("http_index.json")]
    for record in records:
        core._verify_product_http_capture(record)
        require(
            record["response_status_code"] == 200 and record["response_capture_complete"] is True,
            "combined_chat_http_failed",
        )
    if continued:
        baseline = read("accepted_baseline_artifacts.json")
        require(
            baseline["context"] == context
            and _same_plan(baseline["accepted_map"], trace["initial"])
            and baseline["original"]["active_plan_id"] != trace["initial"]["active_plan_id"],
            "combined_chat_accepted_baseline_invalid",
        )
        records = records[len(baseline["http_boundary_files"]) :]
    message = (
        "Set Pfeiffer Big Sur State Park duration to 90 minutes; set Stearns Wharf earliest arrival to 10:00."
        if continued
        else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes."
    )
    chats = [row for row in records if row["url"].endswith("/copilot/messages")]
    require(
        trace["scenario"] == name
        and trace["context"] == context
        and trace["stage_first"] is stage_first
        and trace["assertions_passed"] is True
        and len(turns) == len(chats) == 1
        and turns[0]["input"]["message"] == message,
        "combined_chat_input_invalid",
    )
    request, events = _stream(chats[0])
    require(
        turns[0]["input"] == request and turns[0]["events"] == events and turns[0]["output"] == events[-1],
        "combined_chat_stream_unbound",
    )
    turn, session = (events[-1]["payload"][key] for key in ("turn", "session"))
    observed = read("context_combined_before_stage.json")
    require(
        observed["accepted_plan_id"] == session["accepted_plan_id"] == trace["initial"]["active_plan_id"]
        and observed["accepted_plan_content_hash"] == trace["initial"]["active_plan_content_hash"]
        and observed["draft_operations"] == session["draft"] == []
        and observed["session_revision"] == session["revision"] == turn["context_revision"]
        and ("bixby_creek_bridge_viewpoint" in observed["allowed_stop_ids"]) is not continued
        and ("pfeiffer_big_sur_state_park" in observed["allowed_stop_ids"]) is continued,
        "combined_chat_current_context_invalid",
    )
    mutations = [
        row
        for row in records
        if row["method"] == "POST"
        and row["url"].endswith(
            ("/draft/operations", "/preview", "/accept", "/keep-original", "/rollback", "/draft/undo")
        )
    ]
    require(
        len(mutations) == 3
        and [row["url"].rsplit("/", 1)[-1] for row in mutations] == ["operations", "preview", "accept"]
        and all("/api/sessions/" + session["session_id"] + "/" in row["url"] for row in mutations),
        "combined_chat_mutation_inventory_invalid",
    )
    staged = stage["output"]["session"]
    require(
        stage["input"] == json.loads(mutations[0]["request_body_utf8"])
        and stage["output"] == json.loads(mutations[0]["response_body_utf8"])
        and stage["input"]["expected_revision"] == session["revision"]
        and staged["revision"] == session["revision"] + 1
        and len(stage["input"]["operations"]) == len(staged["draft"]) == 2
        and staged["accepted_plan_id"] == session["accepted_plan_id"]
        and staged["proposal"] is None
        and all(before[key] == trace["initial"][key] for key in ("active_plan_id", "active_plan_content_hash"))
        and all(
            before[key] in (None, trace["initial"][key])
            for key in ("rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids")
        )
        and before == trace["staged"]
        and before["request"]
        == json.loads(mutations[1]["request_body_utf8"])
        == {"expected_revision": staged["revision"]},
        "combined_chat_staging_invalid",
    )
    duration = {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}
    expected = (
        [
            {"type": "set_stop_duration", "target": "pfeiffer_big_sur_state_park", "parameters": duration},
            {
                "type": "set_stop_time_window",
                "target": "stearns_wharf",
                "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
            },
        ]
        if continued
        else [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
            },
            {"type": "set_stop_duration", "target": "griffith_observatory", "parameters": duration},
        ]
    )
    operations = stage["input"]["operations"]
    require(
        [{key: row[key] for key in ("type", "target", "parameters")} for row in operations] == expected
        and all(row["source"] == "copilot" for row in operations),
        "combined_chat_semantics_invalid",
    )
    interpretation = turn["interpretation"]
    intents = interpretation["intents"]
    require(interpretation["state"] == "proposal_ready" and len(intents) == 2, "combined_chat_intent_mapping_invalid")
    for intent, operation, canonical in zip(intents, operations, staged["draft"], strict=True):
        parameters = (
            {"candidate_id": intent["candidate_id"]} if intent["type"] == "replace_nearby" else intent["parameters"]
        )
        require(
            intent["type"] == operation["type"]
            and intent["target_stop_id"] == operation["target"]
            and parameters == operation["parameters"]
            and all(canonical[key] == value for key, value in operation.items()),
            "combined_chat_intent_mapping_invalid",
        )
    proposal = payload["proposal"]
    require(
        payload == json.loads(mutations[1]["response_body_utf8"])
        and proposal["parent_plan_id"] == trace["initial"]["active_plan_id"]
        and proposal["parent_plan_content_hash"] == trace["initial"]["active_plan_content_hash"]
        and proposal["compiled_request"]["operations"]
        == [{key: value for key, value in row.items() if key != "created_at"} for row in staged["draft"]],
        "combined_chat_preview_io_invalid",
    )
    _proposal(proposal, trace["rendered"], combined=True)
    execution = proposal["runtime_execution"]
    child = execution["child_plan"]
    duration_stop = next(
        stop for stop in child["selected_stops"] if stop["stop_id"] == expected[0 if continued else 1]["target"]
    )
    require(
        duration_stop["visit_duration_minutes"] == 90
        and duration_stop["duration_constraint"] == duration["duration"]
        and execution["operations"]
        == execution["repair_request"]["operations"]
        == proposal["compiled_request"]["operations"]
        and execution["request_content_hash"] == core._artifact_content_hash(execution["repair_request"]),
        "combined_chat_rendered_semantics_invalid",
    )
    if continued:
        require(
            next(stop for stop in child["selected_stops"] if stop["stop_id"] == "stearns_wharf")[
                "time_window_constraint"
            ]
            == {
                "schema_version": "stop-time-window-constraint-v1",
                **expected[1]["parameters"],
                "early_arrival_policy": "wait_until_earliest_arrival",
                "latest_departure_semantics": "departure_after_visit",
            }
            and trace["rendered"]["rendered_geometry_hash"] != trace["initial"]["rendered_geometry_hash"]
            and set(trace["rendered"]["rendered_marker_ids"]) == set(trace["initial"]["rendered_marker_ids"]),
            "combined_chat_continuation_semantics_invalid",
        )
        accepted_setup = json.loads(read("accepted_setup_accept_http.json")["response_body_utf8"])
        parent = next(
            plan
            for plan in accepted_setup["workspace"]["geography"]["plans"]
            if plan["plan_id"] == trace["initial"]["active_plan_id"]
        )
        for collection, keys in (
            (
                "route_path",
                (
                    "occurrence_index",
                    "node_id",
                    "selected_stop",
                    "selected_sequence_index",
                    "route_anchor",
                    "arrival_day",
                    "departure_day",
                    "coordinate_source",
                ),
            ),
            (
                "validated_legs",
                (
                    "leg_index",
                    "day",
                    "from_day",
                    "to_day",
                    "cross_day",
                    "travel_mode",
                    "origin_id",
                    "destination_id",
                    "validation_status",
                    "route_leg_id",
                    "road_validated",
                    "fallback_used",
                    "geometry_source",
                    "distance_m",
                    "duration_s",
                ),
            ),
            ("gaps", ()),
        ):
            require(
                [
                    {"geometry": feature["geometry"], **{key: feature["properties"].get(key) for key in keys}}
                    for feature in parent[collection]["features"]
                ]
                == [
                    {"geometry": feature["geometry"], **{key: feature["properties"].get(key) for key in keys}}
                    for feature in proposal["geography_plan"][collection]["features"]
                ],
                "combined_chat_timing_route_changed",
            )
    else:
        require(
            set(trace["rendered"]["rendered_marker_ids"])
            == (set(trace["initial"]["rendered_marker_ids"]) - {"bixby_creek_bridge_viewpoint"})
            | {"pfeiffer_big_sur_state_park"}
            and trace["rendered"]["rendered_geometry_hash"] != trace["initial"]["rendered_geometry_hash"],
            "combined_chat_rendered_semantics_invalid",
        )
    require(
        controls["schema_version"] == "combined-chat-controls-v2"
        and controls["stage_first"] is stage_first
        and controls["preview_plan_id"] == controls["accepted_plan_id"] == child["plan_id"]
        and sum(row["trusted"] is True and row["label"] == "Add all to draft" for row in controls["clicks"])
        == int(stage_first)
        and sum(row["trusted"] is True and row["label"] == "Preview combined repair" for row in controls["clicks"])
        == 1,
        "combined_chat_real_controls_missing",
    )
    require(
        disabled["label"] == "All changes added to draft"
        and disabled["disabled"] is True
        and disabled["visible"] is True
        and disabled["session"] == payload["session"]
        and "All requested changes are staged" in disabled["note"]
        and not any(row["label"] == disabled["label"] for row in controls["clicks"]),
        "combined_chat_added_control_invalid",
    )
    decision, pointer = accepted["result"]["decision"], accepted["result"]["pointer"]
    for artifact in (decision, pointer):
        canonical = json.dumps(
            {key: value for key, value in artifact.items() if key != "content_hash"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        require(
            artifact["content_hash"] == hashlib.sha256(canonical).hexdigest(), "combined_chat_acceptance_hash_invalid"
        )
    accept_request = json.loads(mutations[2]["request_body_utf8"])
    require(
        accepted == json.loads(mutations[2]["response_body_utf8"])
        and accept_request["expected_revision"] == payload["session"]["revision"]
        and accept_request["confirmation"] == "accept_evaluated_repair"
        and decision["previous_plan_id"] == proposal["parent_plan_id"]
        and decision["selected_plan_id"]
        == pointer["accepted_plan_id"]
        == accepted["session"]["accepted_plan_id"]
        == child["plan_id"]
        and decision["selected_plan_content_hash"] == pointer["accepted_plan_content_hash"] == child["content_hash"]
        and decision["certificate_content_hash"] == proposal["certificate_content_hash"]
        and decision["diff_id"] == proposal["diff_id"]
        and accepted["session"]["draft"] == []
        and accepted["session"]["proposal"] is None
        and trace["accepted"] is True
        and trace["accepted_map"]["preview_state"] == "PARENT_RENDERED"
        and all(
            trace["accepted_map"][key] == trace["rendered"][key]
            for key in (
                "active_plan_id",
                "active_plan_content_hash",
                "rendered_map_plan_id",
                "rendered_geometry_hash",
                "rendered_marker_ids",
            )
        ),
        "combined_chat_acceptance_invalid",
    )
    restored = [
        json.loads(row["response_body_utf8"])
        for row in records[records.index(mutations[2]) + 1 :]
        if row["method"] == "GET" and row["url"].endswith("/api/sessions/" + session["session_id"])
    ]
    require(
        len(restored) == 1
        and restored[0]["session"] == accepted["session"]
        and restored[0]["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"],
        "combined_chat_accepted_reload_invalid",
    )
    panel = read("evidence_panel.json")["scope"]
    require(
        panel["evidencePlanId"] == child["plan_id"] and panel["planContentHash"] == child["content_hash"],
        "combined_chat_evidence_invalid",
    )
    if continued:
        final = read("context_control_final.json")
        require(
            final["accepted_plan_id"] == final["conversation_memory"]["plan_id"] == child["plan_id"]
            and final["accepted_plan_content_hash"]
            == final["conversation_memory"]["plan_content_hash"]
            == child["content_hash"]
            and final["draft_operations"] == [],
            "combined_chat_final_context_invalid",
        )
    return {
        "scope": "Actual combined chat controls, real evaluated canonical child, explicit acceptance and reload",
        "context": context,
        "child_plan_id": child["plan_id"],
        "variants": (
            ["proposal_combined_add", "proposal_combined_preview_staged", "proposal_combined_added_disabled"]
            if stage_first
            else ["proposal_combined_preview_empty", "proposal_combined_added_disabled"]
        ),
    }


def _stream(record, *, complete=True):
    events = core._sse_events(record["response_body_utf8"])
    request = json.loads(record["request_body_utf8"])
    require(
        events
        and all(
            event["schema_version"] == "copilot-stream-v1"
            and event["sequence"] == index
            and event["client_message_id"] == request["client_message_id"]
            and event["request_id"] == events[0]["request_id"]
            for index, event in enumerate(events, 1)
        ),
        "sse_identity_invalid",
    )
    require([row["type"] for row in events[:2]] == ["message.accepted", "assistant.status"], "sse_prefix_invalid")
    if complete:
        require(len(events) == 3 and events[-1]["type"] in {"turn.completed", "turn.failed"}, "sse_terminal_invalid")
        if events[-1]["type"] == "turn.completed":
            core._verify_copilot_stream(events, expected_client_message_id=request["client_message_id"])
        else:
            require(
                set(events[-1])
                == {
                    "schema_version",
                    "type",
                    "sequence",
                    "request_id",
                    "client_message_id",
                    "code",
                    "status",
                    "retryable",
                },
                "sse_failed_fields_invalid",
            )
    else:
        require(len(events) == 2, "sse_aborted_prefix_misrepresented")
    return request, events


def combined_conflict_control_receipt(read, *, name):
    """Recompute the negative journey from exact chat, HTTP and visible state."""
    continued = name.endswith("_accepted_child")
    trace = read("trace.json")
    turns = read("exact_chat_io.json")
    visible = read("blocked_visible_results.json")
    messages = [
        "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 120 minutes.",
        "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 90 minutes.",
        (
            "Set Stearns Wharf role to meal; set Stearns Wharf role to activity."
            if continued
            else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Bixby Creek Bridge duration to 90 minutes."
        ),
    ]
    questions = [
        "Different values were requested for the same stop attribute. Which value should I use?",
        "The same edit was requested more than once. Should I keep just one copy?",
        (
            "Different values were requested for the same stop attribute. Which value should I use?"
            if continued
            else "A stop cannot be preserved or edited and replaced in the same request. Which change should I keep?"
        ),
    ]
    require(
        name in COMBINED_CONFLICT_CASES
        and trace["scenario"] == name
        and trace["accepted"] is False
        and trace["chat_turn_count"] == len(turns) == len(visible) == 3
        and _same_plan(trace["initial"], trace["final"]),
        "combined_conflict_trace_invalid",
    )
    sessions = []
    for row, shown, message, question in zip(turns, visible, messages, questions, strict=True):
        payload = row["output"]["payload"]
        session, turn = payload["session"], payload["turn"]
        interpretation = turn["interpretation"]
        require(
            row["input"]["message"] == turn["user_message"] == shown["user_message"] == message
            and interpretation["state"] == "clarification_required"
            and [intent["type"] for intent in interpretation["intents"]] == ["request_clarification"]
            and interpretation["intents"][0]["clarification_question"] == question
            and turn["assistant_message"] == shown["assistant_message"] == question + " No part was staged or changed."
            and session["revision"] == row["input"]["expected_revision"] == turn["context_revision"]
            and session["draft"] == []
            and session["proposal"] is None
            and session["accepted_plan_id"] == trace["initial"]["active_plan_id"]
            and shown["proposal_card_count"] == 0
            and _same_plan(shown["map"], trace["initial"]),
            "combined_conflict_semantics_invalid",
        )
        sessions.append(session)
    require(len({session["session_id"] for session in sessions}) == 1, "combined_conflict_owner_invalid")
    for label in ("conflicts_before_reload", "conflicts_after_reload"):
        context = read(f"context_{label}.json")
        require(
            context["session_id"] == sessions[0]["session_id"]
            and context["accepted_plan_id"] == trace["initial"]["active_plan_id"]
            and context["accepted_plan_content_hash"] == trace["initial"]["active_plan_content_hash"]
            and context["draft_operations"] == []
            and context["evaluated_proposal"] is None,
            "combined_conflict_context_changed",
        )
    clicks = read("conflict_controls_before_reload.json")
    require(
        sum(row["trusted"] is True and row["id"] == "copilot-submit" for row in clicks) == 3,
        "combined_conflict_real_submit_missing",
    )


def _conversation(root, run_id, process, nodes, *, cases=None):
    from openai.lib._pydantic import to_strict_json_schema

    from itinerary_system.product_app.openai_copilot import OpenAIInterpretationSchemaV2

    base = root / "conversation"
    cases = CONVERSATION_CASES if cases is None else cases
    require({path.name for path in base.iterdir() if path.is_dir()} == set(cases), "conversation_case_universe_invalid")
    redaction = load(root / "state_credential_redaction.json")
    state_files = sorted((root / "_state").rglob("session_*.json"))
    redaction_rows = redaction.get("files", [])
    require(
        redaction.get("schema_version") == "conversation-state-credential-redaction-v1"
        and redaction.get("scope") == "completed_runtime_state_evidence"
        and redaction.get("assertions_passed") is True
        and redaction.get("redacted_file_count") == len(redaction_rows) == len(state_files)
        and [row.get("path") for row in redaction_rows] == [path.relative_to(root).as_posix() for path in state_files],
        "conversation_state_redaction_invalid",
    )
    for path, row in zip(state_files, redaction_rows, strict=True):
        state = load(path)
        require(
            set(row) == {"path", "redacted_field_names", "pre_redaction_sha256", "post_redaction_sha256"}
            and row["redacted_field_names"] == ["token_verifier"]
            and row["pre_redaction_sha256"] != row["post_redaction_sha256"] == core._sha256(path)
            and state.get("token_verifier") == "OMITTED_SECRET",
            "conversation_state_redaction_invalid",
        )
    witnesses, observed = [], set()
    previous = stamp(process["started_at"])
    for name, test_name in cases.items():
        case = base / name
        base_name = name.removesuffix("_accepted_child")
        accepted_context = name != base_name
        fake = base_name == "fake_sdk_cancel"
        receipt = _lifecycle(case, run_id + "_" + name, process, fake=fake)
        state_directory = root / "_state" / hashlib.sha256(name.encode()).hexdigest()[:12]
        require(
            Path(core._load(case / "server_lifecycle.json")["state_directory"]).resolve() == state_directory
            and state_directory.is_dir(),
            "conversation_state_artifact_binding_invalid",
        )
        require(previous <= stamp(receipt["installed_at"]), "conversation_lifecycle_overlap")
        previous = stamp(receipt["finalized_at"])
        observed.update(_runtime(case, receipt, nodes, "http://127.0.0.1:8127"))
        trace = core._load(case / "trace.json")
        require(
            trace["schema_version"] == "conversation-browser-v1"
            and trace["scenario"] == name
            and trace["assertions_passed"] is True,
            "conversation_trace_invalid",
        )
        records = _http(case, "http://127.0.0.1:8127", allow_partial=fake)
        chats = [row for row in records if row["url"].endswith("/copilot/messages")]
        turns = load(case / "exact_chat_io.json")
        require(isinstance(turns, list), "conversation_exact_io_invalid")
        complete = [row for row in chats if row.get("response_capture_complete", True)]
        require(len(turns) == len(complete), "conversation_stream_inventory_invalid")
        for record, turn in zip(complete, turns, strict=True):
            request, events = _stream(record)
            require(
                turn["input"] == request and turn["events"] == events and turn["output"] == events[-1],
                "conversation_exact_io_mismatch",
            )
        scope = core._load(case / "provider_scope.json")
        require(
            scope["live_provider_call_count"] == 0
            and scope["real_chat_endpoint"] is True
            and scope["internal_api_mocked"] is False
            and scope["live_openai_connectivity_proven"] is False
            and scope["fake_sdk_attempt_count"] == (1 if fake else 0),
            "conversation_provider_scope_invalid",
        )
        require(load(case / "console_errors.json") == [], "conversation_console_error")
        failures = load(case / "network_failures.json")
        require(
            isinstance(failures, list)
            and (
                not failures
                or fake
                and all(
                    row["url"].endswith("/copilot/messages") and "ERR_ABORTED" in str(row["failure"])
                    for row in failures
                )
            ),
            "conversation_network_failure",
        )
        mutations = [
            row
            for row in records
            if row["url"].endswith(("/draft/operations", "/preview", "/accept", "/keep-original", "/rollback"))
        ]
        if name == "clarification_reload":
            require(len(turns) == len(chats) == 2, "clarification_turn_count_invalid")
            first, answer = (row["output"]["payload"] for row in turns)
            before, after = (
                core._load(case / ("context_pending_" + point + "_reload.json")) for point in ("before", "after")
            )
            pending = before["conversation_memory"]["pending_clarification"]
            require(
                first["turn"]["user_message"] == "Replace Bixby Creek Bridge."
                and first["turn"]["interpretation"]["state"] == "clarification_required"
                and pending["target_stop_id"] == "bixby_creek_bridge_viewpoint"
                and pending["missing_slots"] == ["candidate_id"]
                and before["conversation_memory"] == after["conversation_memory"]
                and answer["turn"]["user_message"] == "Pfeiffer Big Sur State Park."
                and [
                    (row["type"], row["target_stop_id"], row["candidate_id"])
                    for row in answer["turn"]["interpretation"]["intents"]
                ]
                == [("replace_nearby", "bixby_creek_bridge_viewpoint", "pfeiffer_big_sur_state_park")]
                and core._load(case / "context_answered.json")["conversation_memory"]["pending_clarification"] is None,
                "clarification_memory_binding_invalid",
            )
            stages = [row for row in mutations if row["url"].endswith("/draft/operations")]
            previews = [row for row in mutations if row["url"].endswith("/preview")]
            require(len(stages) == len(previews) == 1 and len(mutations) == 2, "clarification_mutations_invalid")
            operation = json.loads(stages[0]["request_body_utf8"])
            require(
                operation["type"] == "replace_nearby"
                and operation["target"] == "bixby_creek_bridge_viewpoint"
                and operation["parameters"] == {"candidate_id": "pfeiffer_big_sur_state_park"},
                "clarification_semantics_invalid",
            )
            payload = core._load(case / "preview_response.json")
            require(payload == json.loads(previews[0]["response_body_utf8"]), "clarification_preview_io_mismatch")
            _proposal(payload["proposal"], trace["rendered"])
            require(
                trace["accepted"] is False
                and trace["initial"]["active_plan_id"] == payload["proposal"]["parent_plan_id"],
                "clarification_not_acceptance",
            )
            labels = ("initial", "clarified_preview")
        elif name in QUEUED_PREVIEW_CASES:
            if accepted_context:
                boundary = core._load(case / "accepted_baseline_artifacts.json")["http_boundary_files"]
                _conversation_accepted_baseline(case, trace, records[: len(boundary)], continuation=True)
            queued_preview_control_receipt(lambda filename, directory=case: load(directory / filename), name=name)
            from run_e2e1_vertical_slice import _source_rows

            expected_sources = {
                row["path"]: row["sha256"]
                for row in _source_rows("tests/product_app/browser/test_copilot_queued_preview_live.py")
            }
            require(
                load(case / "source_inputs_before.json") == load(case / "source_inputs_after.json") == expected_sources,
                "queued_source_inputs_changed",
            )
            labels = ("initial", "after_delete_before_release", "after_release", "after_reload")
        elif name.startswith("combined_chat"):
            if accepted_context:
                boundary = core._load(case / "accepted_baseline_artifacts.json")["http_boundary_files"]
                _conversation_accepted_baseline(case, trace, records[: len(boundary)], continuation=True)
            combined_chat_control_receipt(lambda filename, directory=case: load(directory / filename), name=name)
            labels = ("initial", "combined_preview", "combined_compare", "combined_accepted_reload")
        else:
            if accepted_context:
                _conversation_accepted_baseline(case, trace, records)
            else:
                require(not mutations, "conversation_unexpected_plan_mutation")
            require(_same_plan(trace["initial"], trace["final"]), "conversation_unexpected_plan_mutation")
            if base_name == "combined_conflicts":
                combined_conflict_control_receipt(
                    lambda filename, directory=case: load(directory / filename), name=name
                )
                labels = ("initial", "conflicts_blocked", "conflicts_reloaded")
            elif base_name == "preference_memory":
                require(
                    trace["filler_turn_count"] == 7 and trace["chat_turn_count"] == len(turns) == len(chats) == 11,
                    "memory_turn_count_invalid",
                )
                first = turns[0]["output"]["payload"]["turn"]
                corrected = turns[9]["output"]["payload"]["turn"]
                memory = core._load(case / "context_after_seven_turns.json")["conversation_memory"]
                correction = core._load(case / "context_corrected.json")["conversation_memory"]
                reset = core._load(case / "context_reset.json")["conversation_memory"]
                require(
                    memory["proposed_preferences"]
                    == [
                        {
                            "preference": "reduce_driving",
                            "source_turn_id": first["turn_id"],
                            "disposition": "proposed_only",
                        }
                    ]
                    and correction["proposed_preferences"]
                    == [
                        {
                            "preference": "preserve_experience",
                            "source_turn_id": corrected["turn_id"],
                            "disposition": "proposed_only",
                        }
                    ]
                    and reset["proposed_preferences"] == []
                    and reset["pending_clarification"] is None,
                    "memory_semantics_invalid",
                )
                require(
                    turns[0]["input"]["message"] == "I prefer less driving."
                    and turns[9]["input"]["message"] == "Actually, I prefer preserving the experience instead."
                    and "reduce driving" not in turns[-1]["output"]["payload"]["turn"]["assistant_message"]
                    and "preserve experience" not in turns[-1]["output"]["payload"]["turn"]["assistant_message"],
                    "memory_correction_or_reset_invalid",
                )
                labels = ("initial", "remembered_and_corrected", "reset")
            elif base_name.startswith("capacity_"):
                require(len(turns) == len(chats) == 3, "capacity_turn_count_invalid")
                fixture = core._load(case / "capacity_fixture_io.json")
                for field in ("before", "after"):
                    require(
                        hashlib.sha256(fixture[field + "_utf8"].encode()).hexdigest() == fixture[field + "_sha256"],
                        "capacity_fixture_hash_invalid",
                    )
                require(
                    len(fixture["after_utf8"].encode()) == fixture["serialized_bytes"] < fixture["max_bytes"]
                    and fixture["kind"] == base_name.removeprefix("capacity_")
                    and (
                        fixture["turn_count"] == fixture["max_turns"]
                        if fixture["kind"] == "turns"
                        else fixture["turn_count"] < fixture["max_turns"]
                        and fixture["serialized_bytes"] + fixture["reserved_headroom_bytes"] > fixture["max_bytes"]
                    ),
                    "capacity_fixture_limit_invalid",
                )
                terminal = turns[1]["output"]
                expected = (
                    "conversation_turn_limit_reached"
                    if base_name == "capacity_turns"
                    else "conversation_size_limit_reached"
                )
                require(
                    terminal["type"] == "turn.failed"
                    and terminal["code"] == trace["blocked_code"] == expected
                    and terminal["status"] == 422
                    and terminal["retryable"] is False
                    and trace["blocked_attempt_delta"] == 0
                    and trace["old_conversation_id"] != trace["new_conversation_id"],
                    "capacity_block_invalid",
                )
                labels = ("initial", "capacity_blocked", "capacity_reset")
            elif name == "selection_conversation_generation":
                from verify_e2e1_selection_generation import verify_selection_generation

                labels = verify_selection_generation(case, trace, records, turns)
            else:
                require(len(chats) == 2 and len(turns) == 1, "cancel_stream_count_invalid")
                (prefix,) = [row for row in chats if row.get("response_capture_complete") is False]
                original, events = _stream(prefix, complete=False)
                interrupted = core._load(case / "interrupted_chat_io.json")
                require(
                    interrupted["input"] == original
                    and interrupted["received_stream"]["raw"] == prefix["response_body_utf8"]
                    and interrupted["received_stream"]["done"] is True
                    and interrupted["received_stream"]["error"],
                    "cancel_prefix_scope_invalid",
                )
                journal, retry = (
                    core._load(case / filename)
                    for filename in ("dispatch_after_cancel.json", "dispatch_after_retry.json")
                )
                terminal = turns[0]["output"]
                require(
                    journal == retry
                    and journal["attempt_count"] == journal["attempt_budget"] == 1
                    and journal["outcome"] == trace["outcome"] == "unknown"
                    and turns[0]["input"]["client_message_id"]
                    == original["client_message_id"]
                    == trace["client_message_id"]
                    and terminal["type"] == "turn.failed"
                    and terminal["code"] == "copilot_outcome_unknown"
                    and terminal["retryable"] is False
                    and trace["fake_sdk_attempt_count"] == 1,
                    "cancel_retry_redispatch_invalid",
                )
                wire_rows = load(case / "fake_sdk_io.json")
                require(isinstance(wire_rows, list) and len(wire_rows) == 1, "fake_wire_attempt_count_invalid")
                wire = wire_rows[0]
                require(
                    wire["schema_version"] == "fake-provider-exact-http-v1"
                    and wire["method"] == "POST"
                    and wire["url"] == "http://127.0.0.1:9/v1/responses"
                    and wire["sdk_max_retries"] == 0
                    and wire["transport"] == "httpx.MockTransport"
                    and wire["never_fallthrough"] is True
                    and all(
                        wire[key] is False
                        for key in (
                            "request_headers_persisted",
                            "response_headers_persisted",
                            "authorization_persisted",
                        )
                    )
                    and hashlib.sha256(wire["request_body_utf8"].encode()).hexdigest() == wire["request_body_sha256"],
                    "fake_wire_binding_invalid",
                )
                request = json.loads(wire["request_body_utf8"])
                require(
                    request["tools"] == []
                    and request["store"] is False
                    and request["text"]["format"]
                    == {
                        "type": "json_schema",
                        "name": "OpenAIInterpretationSchemaV2",
                        "strict": True,
                        "schema": to_strict_json_schema(OpenAIInterpretationSchemaV2),
                    },
                    "fake_real_sdk_schema_invalid",
                )
                if wire["response_body_utf8"] is None:
                    require(
                        wire["outcome"] == "cancelled"
                        and wire["response_body_sha256"] is None
                        and wire["response_status"] is None,
                        "fake_response_null_scope_invalid",
                    )
                else:
                    require(
                        wire["outcome"] == "completed"
                        and wire["response_status"] == 200
                        and hashlib.sha256(wire["response_body_utf8"].encode()).hexdigest()
                        == wire["response_body_sha256"],
                        "fake_response_hash_invalid",
                    )
                labels = ("initial", "cancelled", "reconciled")
        _conversation_control_closure(case, name, trace, records)
        plan_ids = {trace["initial"]["active_plan_id"], trace.get("child_plan_id", trace["initial"]["active_plan_id"])}
        if accepted_context:
            labels = ("accepted_setup_original", "accepted_setup_preview", *labels)
            plan_ids.add(core._load(case / "accepted_baseline_artifacts.json")["original"]["active_plan_id"])
        core._verify_journey_captures(case, labels, plan_ids)
        witnesses.append(
            {
                "case": name,
                "test_name": test_name,
                "run_id": receipt["run_id"],
                "pid": receipt["process_id"],
                "installed_at": receipt["installed_at"],
                "finalized_at": receipt["finalized_at"],
                "provider_calls": receipt["sent_provider_calls"],
                "fake_external_http": fake,
            }
        )
    require(observed == nodes, "conversation_runtime_node_universe_invalid")
    return witnesses


def _private_input_universe(repository):
    registration = core._load(repository / "configs/product_app_registry.json")
    delta = core._load(repository / "configs/product_route_delta.json")
    descriptor = core._safe_path(repository, delta["descriptor_path"])
    require(descriptor.is_relative_to((repository / "results/route_deltas").resolve()), "private_delta_path_invalid")
    paths = {repository / "scripts/run_product_app.py", repository / "scripts/e2e1_provider_isolation.py"}
    for directory in (repository / "src", repository / "configs", descriptor.parent):
        paths.update(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for run in registration["runs"]:
        directory = core._safe_path(repository, run["relative_path"])
        require(directory.is_relative_to((repository / "runs").resolve()), "private_run_path_invalid")
        paths.update(
            (path for path in directory.rglob("*") if path.is_file())
            if run["default"]
            else (directory / "manifest.json",)
        )
    return {path.relative_to(repository).as_posix() for path in paths}


def _private_build_id(repository, rows, registry_body):
    # Recompute the documented fingerprint bytes, without trusting reported IDs.
    suffixes = {".css", ".html", ".js", ".json", ".py", ".svg", ".webmanifest"}
    paths = {"scripts/run_product_app.py", "configs/product_app_registry.json", "configs/product_route_delta.json"}
    paths.add(core._load(repository / "configs/product_route_delta.json")["descriptor_path"])
    paths.update(
        row["path"]
        for row in rows
        if row["path"].startswith("src/itinerary_system/product_app/") and Path(row["path"]).suffix.lower() in suffixes
    )
    require(paths <= {row["path"] for row in rows}, "private_build_input_missing")
    digest = hashlib.sha256()
    for relative in sorted(paths, key=lambda value: repository / value):
        body = (
            registry_body
            if relative == "configs/product_app_registry.json"
            else core._safe_path(repository, relative).read_bytes()
        )
        digest.update(relative.encode() + b"\0" + body + b"\0")
    return digest.hexdigest()[:16]


def _conversation_accepted_baseline(case, trace, records, *, continuation=False):
    value = core._load(case / "accepted_baseline_artifacts.json")
    require(
        value["schema_version"] == "conversation-accepted-baseline-v1"
        and value["context"] == trace["context"] == "accepted_child"
        and value["assertions_passed"] is True
        and value["chat_request_count"] == value["fake_sdk_attempt_count"] == 0
        and _same_plan(value["accepted_map"], trace["initial"]),
        "conversation_baseline_scope_invalid",
    )
    index = load(case / "http_index.json")
    boundary = value["http_boundary_files"]
    require(
        boundary
        and boundary == [row["file"] for row in index[: len(boundary)]]
        and not any(row["url"].endswith("/copilot/messages") for row in records[: len(boundary)]),
        "conversation_baseline_http_boundary_invalid",
    )
    sources = []
    for label, method, suffix in (
        ("stage", "POST", "/draft/operations"),
        ("preview", "POST", "/preview"),
        ("accept", "POST", "/accept"),
        ("restored", "GET", None),
    ):
        saved = core._load(case / f"accepted_setup_{label}_http.json")
        matches = [
            i
            for i, row in enumerate(records)
            if i < len(boundary)
            and {key: val for key, val in row.items() if key != "response_capture_complete"} == saved
        ]
        require(
            len(matches) == 1
            and matches[0] < len(boundary)
            and saved["method"] == method
            and saved["response_status_code"] == 200
            and (suffix is None or saved["url"].endswith(suffix)),
            "conversation_baseline_http_unbound",
        )
        sources.append((matches[0], saved))
    require([i for i, _ in sources] == sorted({i for i, _ in sources}), "conversation_baseline_order_invalid")
    mutations = [
        row
        for row in records
        if row["url"].endswith(("/draft/operations", "/preview", "/accept", "/keep-original", "/rollback"))
    ]
    require(
        len(mutations) == 3 and all(records[i] in mutations for i, _ in sources[:3]),
        "conversation_unexpected_plan_mutation",
    )
    stage, preview, accepted, restored = [json.loads(row["response_body_utf8"]) for _, row in sources]
    operation, proposal = stage["operation"], preview["proposal"]
    request = json.loads(sources[0][1]["request_body_utf8"])
    original = value["original"]
    require(
        operation["type"] == request["type"] == "replace_nearby"
        and operation["target"] == request["target"] == "bixby_creek_bridge_viewpoint"
        and operation["source"] == request["source"] == "map"
        and operation["parameters"] == request["parameters"] == {"candidate_id": "pfeiffer_big_sur_state_park"}
        and stage["session"]["draft"] == preview["session"]["draft"] == [operation]
        and stage["session"]["accepted_plan_id"] == original["active_plan_id"]
        and preview["session"]["proposal"] == proposal
        and json.loads(sources[1][1]["request_body_utf8"])["expected_revision"] == stage["session"]["revision"]
        and proposal["compiled_request"]["operations"]
        == [{key: val for key, val in operation.items() if key != "created_at"}],
        "conversation_baseline_replacement_invalid",
    )
    _proposal(proposal, value["preview_map"])
    core._verify_plan_ack(trace["initial"], proposal)
    execution = proposal["runtime_execution"]
    child, certificate = execution["child_plan"], execution["evaluation_certificate"]
    route = certificate["route_validation"]
    require(
        proposal["parent_plan_id"] == original["active_plan_id"]
        and proposal["parent_plan_content_hash"] == original["active_plan_content_hash"]
        and child["plan_id"] != original["active_plan_id"]
        and execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
        and certificate["failures"] == []
        and route["required_leg_count"] > 0
        and route["road_validated_leg_count"] == route["required_leg_count"]
        and route["missing_leg_count"] == route["fallback_leg_count"] == 0
        and set(trace["initial"]["rendered_marker_ids"])
        == (set(original["rendered_marker_ids"]) - {"bixby_creek_bridge_viewpoint"}) | {"pfeiffer_big_sur_state_park"},
        "conversation_baseline_child_invalid",
    )
    decision, pointer = accepted["result"]["decision"], accepted["result"]["pointer"]
    accept_request = json.loads(sources[2][1]["request_body_utf8"])
    for row in (decision, pointer):
        canonical = json.dumps(
            {key: val for key, val in row.items() if key != "content_hash"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        require(
            row["content_hash"] == hashlib.sha256(canonical).hexdigest(),
            "conversation_baseline_acceptance_hash_invalid",
        )
    require(
        accept_request["confirmation"] == "accept_evaluated_repair"
        and accept_request["expected_revision"] == preview["session"]["revision"]
        and decision["decision_type"] == "accept_repair"
        and decision["idempotency_key"] == accept_request["idempotency_key"]
        and decision["previous_plan_id"] == original["active_plan_id"]
        and decision["selected_plan_id"] == child["plan_id"]
        and decision["selected_plan_content_hash"] == child["content_hash"]
        and decision["certificate_id"] == certificate["certificate_id"]
        and decision["certificate_content_hash"] == certificate["content_hash"]
        and decision["diff_id"] == proposal["diff_id"]
        and decision["diff_content_hash"] == proposal["diff_content_hash"]
        and pointer["accepted_plan_id"] == child["plan_id"]
        and pointer["accepted_plan_content_hash"] == child["content_hash"]
        and pointer["original_plan_id"] == original["active_plan_id"]
        and pointer["decision_id"] == decision["decision_id"],
        "conversation_baseline_acceptance_invalid",
    )
    sid = accepted["session"]["session_id"]
    require(
        sources[3][1]["url"].endswith("/api/sessions/" + sid)
        and restored["session"]["session_id"] == sid
        and restored["session"]["accepted_plan_id"] == child["plan_id"]
        and restored["session"]["draft"] == []
        and restored["session"]["proposal"] is None
        and restored["workspace"]["accepted_plan_pointer"]
        == {
            key: pointer[key]
            for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
        }
        and restored["workspace"]["alternative_evidence"][child["plan_id"]]["certificate"] == certificate,
        "conversation_baseline_reload_invalid",
    )
    for label in ("accepted_baseline",) if continuation else ("accepted_baseline", "control_final"):
        context = core._load(case / f"context_{label}.json")
        require(
            context["accepted_plan_id"] == child["plan_id"]
            and context["accepted_plan_content_hash"] == child["content_hash"]
            and context["draft_operations"] == []
            and context["conversation_memory"]["plan_id"] == child["plan_id"]
            and context["conversation_memory"]["plan_content_hash"] == child["content_hash"],
            "conversation_baseline_context_changed",
        )
    for row in records:
        if row["url"].endswith("/copilot/messages") and row.get("response_capture_complete") is not False:
            _, events = _stream(row)
            if events[-1]["type"] == "turn.completed":
                session = events[-1]["payload"]["session"]
                require(
                    session["session_id"] == sid
                    and session["accepted_plan_id"] == child["plan_id"]
                    and session["draft"] == [],
                    "conversation_baseline_chat_context_changed",
                )
    clicks = load(case / "accepted_setup_clicks.json")
    require(
        all(
            any(row["trusted"] is True and row["id"] == identity for row in clicks)
            for identity in ("draft-replace", "inline-preview", "accept-repair")
        ),
        "conversation_baseline_trusted_click_missing",
    )


def _conversation_control_closure(case, name, trace, records):
    if name in {
        "clarification_reload",
        "selection_conversation_generation",
        *COMBINED_CONFLICT_CASES,
    } or name.startswith("combined_chat"):
        return
    accepted_context = name.endswith("_accepted_child")
    value = core._load(case / "control_closure.json")
    clicks = load(case / "control_clicks.json")
    require(
        value["schema_version"] == "conversation-control-closure-v1"
        and value["scenario"] == name
        and value["assertions_passed"] is True
        and value["new_chat_request_count"] == 0
        and type(value["streams_before"]) is int
        and value["streams_before"] == value["streams_after"]
        and value["raw_clicks"] == clicks[: len(value["raw_clicks"])]
        and _same_plan(value["accepted_map"], trace["initial"]),
        "control_closure_scope_invalid",
    )
    name = name.removesuffix("_accepted_child")

    def clicked(*, label=None, identity=None, group=None):
        return any(
            row["trusted"] is True
            and (label is None or row["label"] == label)
            and (identity is None or row["id"] == identity)
            and (group is None or row["action_group"] == group)
            for row in value["raw_clicks"]
        )

    if name == "preference_memory":
        require(
            clicked(identity="delete-all-conversations") and value["visible_turn_count"] == 0,
            "delete_all_click_missing",
        )
        matching = [row for row in records if row["method"] == "DELETE" and row["url"].endswith("/api/conversations")]
        require(
            len(matching) == 1
            and json.loads(matching[0]["request_body_utf8"]) == value["delete_all_request"]
            and json.loads(matching[0]["response_body_utf8"]) == value["delete_all_response"]
            and value["delete_all_request"]["confirmation"] == "delete_all_conversations"
            and type(value["delete_all_response"]["deleted_count"]) is int
            and value["delete_all_response"]["deleted_count"] >= 1,
            "delete_all_http_unbound",
        )
        before, after = (core._load(case / ("context_" + side + "_delete_all.json")) for side in ("before", "after"))
        require(
            value["before"] == before
            and value["after"] == after
            and before["accepted_plan_id"] == after["accepted_plan_id"] == trace["initial"]["active_plan_id"]
            and before["accepted_plan_content_hash"]
            == after["accepted_plan_content_hash"]
            == trace["initial"]["active_plan_content_hash"]
            and after["draft_operations"] == []
            and after["conversation_memory"]["proposed_preferences"] == [],
            "delete_all_context_changed",
        )
    else:
        group = "Message recovery actions" if name == "fake_sdk_cancel" else "Conversation capacity actions"
        expected = (
            (
                "Suggest a safe itinerary change for the accepted plan."
                if accepted_context
                else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park."
            )
            if name == "fake_sdk_cancel"
            else "Review the safer repair."
        )
        require(
            clicked(label="Edit message", group=group) and value["edit_message_composer"] == expected,
            "edit_message_result_unbound",
        )
        if name == "fake_sdk_cancel":
            require(value["fake_attempts_before"] == value["fake_attempts_after"] == 1, "edit_message_dispatched")
        else:
            require(
                clicked(label="Conversation settings", group=group)
                and clicked(identity="close-transcript-settings")
                and value["settings_opened"] is True
                and value["settings_closed"] is True,
                "capacity_settings_not_executed",
            )


def _build(root, run_id, process, nodes):
    base = root / "build_coherence"
    require(
        {path.name for path in base.iterdir() if path.is_dir()} == {"original", "accepted_child"},
        "build_context_universe_invalid",
    )
    witnesses = []
    for context in ("original", "accepted_child"):
        selected_nodes = {node for node in nodes if node.endswith("[" + context + "]")}
        require(len(selected_nodes) == 1, "build_context_node_invalid")
        witnesses.extend(_build_case(base / context, context, run_id + "_" + context, process, selected_nodes))
    return witnesses


def _build_case(base, context, run_id, process, nodes):
    assertions = core._load(base / "browser_assertions.json")
    require(
        assertions["schema_version"] == "build-coherence-browser-v1"
        and assertions["context"] == context
        and all(
            assertions.get(key) is True
            for key in (
                "assertions_passed",
                "blocked_mutation_state_unchanged",
                "explicit_reload_session_preserved",
                "source_drift_latched",
            )
        ),
        "build_assertions_invalid",
    )
    bindings = load(base / "generation_bindings.json")
    require(
        isinstance(bindings, list) and [row["generation"] for row in bindings] == ["generation_a", "generation_b"],
        "build_generation_universe_invalid",
    )
    witnesses, observed = [], set()
    for binding in bindings:
        generation = binding["generation"]
        directory = base / generation
        receipt = _lifecycle(directory, run_id + "_" + generation, process, child_process=True)
        observed.update(_runtime(directory, receipt, nodes, "http://127.0.0.1:8127"))
        health = core._load(directory / "health.json")
        lifecycle = core._load(directory / "server_lifecycle.json")
        command = core._load(directory / "server_command.json")
        require(
            binding["run_id"] == receipt["run_id"] == command["run_id"]
            and binding["pid"] == receipt["process_id"]
            and binding["build_id"] == health["build_id"]
            and binding["provider_receipt"] == generation + "/provider_isolation.json"
            and binding["provider_receipt_sha256"] == core._sha256(directory / "provider_isolation.json")
            and binding["installed_at"] == receipt["installed_at"]
            and binding["finalized_at"] == receipt["finalized_at"]
            and binding["provider_call_count"] == 0
            and command["generation"] == generation
            and command["cwd_scope"] == "private_product_copy"
            and command["command"][:2] == [process["command"][0], "scripts/e2e1_provider_isolation.py"]
            and "--run-id" in command["command"]
            and command["command"][command["command"].index("--run-id") + 1] == receipt["run_id"]
            and lifecycle["provider_call_count"] == 0,
            "build_generation_binding_invalid",
        )
        expected_labels = (
            (
                ("initial_parent", "evaluated_child", "accepted_child")
                if context == "accepted_child"
                else ("initial_parent",)
            )
            if generation == "generation_a"
            else ("active_plan_after_update",)
        )
        require(
            binding["readiness_artifacts"]
            == sorted(generation + "/" + label + ".readiness.json" for label in expected_labels),
            "build_capture_inventory_invalid",
        )
        core._verify_journey_captures(
            directory,
            expected_labels,
            {
                core._load(base / "map_checkpoints.json")["initial_parent"]["active_plan_id"],
                assertions["accepted_plan_id"],
            },
        )
        witnesses.append(
            {
                "generation": generation,
                "run_id": receipt["run_id"],
                "pid": receipt["process_id"],
                "installed_at": receipt["installed_at"],
                "finalized_at": receipt["finalized_at"],
                "build_id": health["build_id"],
                "provider_calls": 0,
            }
        )
    require(
        observed == nodes
        and witnesses[0]["build_id"] != witnesses[1]["build_id"]
        and stamp(witnesses[0]["finalized_at"]) <= stamp(witnesses[1]["installed_at"]),
        "build_restart_order_invalid",
    )
    copied = core._load(base / "private_fixture_inputs.json")
    require(
        copied["schema_version"] == "build-coherence-private-inputs-v1"
        and copied["production_source_modified"] is False
        and copied["api_response_mocking"] is False
        and stamp(process["started_at"]) <= stamp(copied["copied_at"]) <= stamp(witnesses[0]["installed_at"]),
        "private_copy_scope_invalid",
    )
    rows = copied["files"]
    require(
        isinstance(rows, list)
        and rows
        and len({row["path"] for row in rows}) == len(rows)
        and {row["path"] for row in rows} == _private_input_universe(core.ROOT),
        "private_copy_inventory_invalid",
    )
    for row in rows:
        path = core._safe_path(core.ROOT, row["path"])
        require(
            path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and row["sha256"] == row["copied_sha256"] == core._sha256(path),
            "private_copy_source_mismatch",
        )
    registry_hash = next(row["sha256"] for row in rows if row["path"] == "configs/product_app_registry.json")
    changes = [core._load(base / ("registry_change_" + name + ".json")) for name in ("b", "drift")]
    previous = registry_hash
    for row in changes:
        require(
            row["path"] == "configs/product_app_registry.json" and row["before_sha256"] == previous,
            "private_registry_lineage_invalid",
        )
        for side in ("before", "after"):
            require(
                hashlib.sha256(row[side + "_utf8"].encode()).hexdigest() == row[side + "_sha256"],
                "private_registry_hash_invalid",
            )
        before, after = json.loads(row["before_utf8"]), json.loads(row["after_utf8"])
        selected_before = next(item for item in before["runs"] if item["default"])
        selected_after = next(item for item in after["runs"] if item["default"])
        require(selected_before["label"] != selected_after["label"], "private_registry_label_unchanged")
        selected_before["label"] = selected_after["label"]
        require(before == after, "private_registry_non_label_mutation")
        previous = row["after_sha256"]
    expected_builds = [
        _private_build_id(core.ROOT, rows, changes[0][side + "_utf8"].encode()) for side in ("before", "after")
    ]
    require(
        expected_builds
        == [witness["build_id"] for witness in witnesses]
        == [assertions["first_build_id"], assertions["second_build_id"]],
        "build_fingerprint_identity_invalid",
    )
    restore = core._load(base / "registry_restore.json")
    require(
        restore["process_restarted"] is False
        and restore["before_sha256"] == changes[1]["after_sha256"]
        and restore["after_sha256"] == restore["expected_generation_b_sha256"] == changes[0]["after_sha256"]
        and hashlib.sha256(restore["before_utf8"].encode()).hexdigest() == restore["before_sha256"]
        and hashlib.sha256(restore["after_utf8"].encode()).hexdigest() == restore["after_sha256"],
        "private_registry_restore_invalid",
    )
    records = _http(base, "http://127.0.0.1:8127", statuses=(200, 409, 503))
    requests = load(base / "request_index.json")
    api_requests = [row for row in requests if row["url"].startswith("http://127.0.0.1:8127/api/")]
    require(len(api_requests) == len(records), "build_request_inventory_invalid")
    for record in records:
        matches = [row for row in api_requests if all(row[key] == record[key] for key in ("phase", "method", "url"))]
        require(matches, "build_request_response_unbound")
        expected_build = expected_builds[0 if record["phase"] in {"generation_a", "generation_b_old_page"} else 1]
        if record["method"] not in {"GET", "HEAD", "OPTIONS"}:
            require(all(row["build_header"] == expected_build for row in matches), "build_request_header_mismatch")
    errors = [row for row in records if row["response_status_code"] != 200]
    require(
        [row["response_status_code"] for row in errors] == [409, 503]
        and [json.loads(row["response_body_utf8"]) for row in errors]
        == [{"detail": "product_client_build_mismatch"}, {"detail": "product_server_build_changed"}]
        and all(row["url"].endswith("/draft/operations") for row in errors),
        "build_blocked_write_invalid",
    )
    for filename in ("stale_write_state_hashes.json", "source_drift_write_state_hashes.json"):
        state = core._load(base / filename)
        require(
            state["before"] == state["after"] and isinstance(state["before"], dict), "build_blocked_write_mutated_state"
        )
    identity = assertions["session"]
    for code in ("product_client_build_mismatch", "product_server_build_changed"):
        recovery = core._load(base / (code + "_visible_recovery.json"))
        require(
            recovery["code"] == code
            and recovery["stored_session"] == identity
            and recovery["text"]
            and "reload-current-build" in recovery["html"],
            "build_visible_recovery_invalid",
        )
    latched = core._load(base / "restored_source_latched_health.json")
    require(
        latched["status"] == 503 and json.loads(latched["body_utf8"]) == {"detail": "product_server_build_changed"},
        "build_source_drift_not_latched",
    )
    previews = [json.loads(row["response_body_utf8"]) for row in records if row["url"].endswith("/preview")]
    accepted = [json.loads(row["response_body_utf8"]) for row in records if row["url"].endswith("/accept")]
    created = [
        json.loads(row["response_body_utf8"])
        for row in records
        if row["method"] == "POST" and row["url"].endswith("/api/sessions")
    ]
    require(
        len(created) == 1 and created[0]["session"]["session_id"] == identity["session_id"],
        "build_session_creation_invalid",
    )
    checkpoints = core._load(base / "map_checkpoints.json")
    if context == "accepted_child":
        require(len(previews) == len(accepted) == 1, "build_preview_accept_count_invalid")
        proposal = previews[0]["proposal"]
        _proposal(proposal, checkpoints["evaluated_child"])
        decision = accepted[0]["result"]["decision"]
        require(
            decision["selected_plan_id"] == assertions["accepted_plan_id"] == proposal["child_plan_id"]
            and decision["selected_plan_content_hash"]
            == assertions["accepted_plan_content_hash"]
            == proposal["child_plan_content_hash"]
            and decision["decision_id"] == assertions["decision_id"]
            and decision["certificate_id"] == assertions["certificate_id"] == proposal["certificate_id"]
            and decision["content_hash"]
            == core._canonical_sha256({key: value for key, value in decision.items() if key != "content_hash"})
            and _same_plan(checkpoints["accepted_child"], checkpoints["active_plan_after_update"]),
            "build_acceptance_lineage_invalid",
        )
        core._verify_plan_ack(checkpoints["active_plan_after_update"], proposal)
        before_geography = accepted[0]["workspace"]["geography"]
    else:
        require(
            previews == accepted == [] and assertions["decision_id"] is assertions["certificate_id"] is None,
            "original_build_fabricated_acceptance",
        )
        plan = next(
            plan
            for plan in created[0]["workspace"]["geography"]["plans"]
            if plan["plan_id"] == created[0]["session"]["accepted_plan_id"]
        )
        initial = checkpoints["initial_parent"]
        require(
            initial["active_plan_id"]
            == initial["rendered_map_plan_id"]
            == assertions["accepted_plan_id"]
            == plan["plan_id"]
            and initial["active_plan_content_hash"] == assertions["accepted_plan_content_hash"] == plan["content_hash"]
            and initial["rendered_geometry_hash"] == core._completed_race_geometry_hash(plan)
            and set(initial["rendered_marker_ids"])
            == {feature["properties"]["stop_id"] for feature in plan["stops"]["features"]}
            and initial["preview_state"] == "PARENT_RENDERED"
            and checkpoints["active_plan_after_update"] == initial,
            "original_build_plan_disconnected",
        )
        require(
            not any(
                row["response_status_code"] == 200
                and row["method"] != "GET"
                and row["url"].endswith("/draft/operations")
                for row in records
            ),
            "original_build_unexpected_staging",
        )
        before_geography = created[0]["workspace"]["geography"]
    restored = [
        json.loads(row["response_body_utf8"])
        for row in records
        if row["phase"] == "generation_b_reloaded"
        and row["method"] == "GET"
        and row["url"].endswith("/api/sessions/" + identity["session_id"])
    ]
    require(
        len(restored) == 1
        and restored[0]["session"]["accepted_plan_id"] == assertions["accepted_plan_id"]
        and restored[0]["session"]["draft"] == []
        and restored[0]["workspace"]["accepted_plan_pointer"].get("decision_id") == assertions["decision_id"],
        "build_restored_session_disconnected",
    )
    restored_session = restored[0]["session"]
    pointer = restored[0]["workspace"]["accepted_plan_pointer"]
    before_plans = [plan for plan in before_geography["plans"] if plan["plan_id"] == assertions["accepted_plan_id"]]
    restored_plans = [
        plan
        for plan in restored[0]["workspace"]["geography"]["plans"]
        if plan["plan_id"] == assertions["accepted_plan_id"]
    ]
    require(
        all(restored_session[field] == created[0]["session"][field] for field in ("session_id", "trip_id", "run_id"))
        and pointer["accepted_plan_id"] == assertions["accepted_plan_id"]
        and pointer["accepted_plan_content_hash"] == assertions["accepted_plan_content_hash"]
        and len(before_plans) == len(restored_plans) == 1
        and restored_plans[0] == before_plans[0],
        "build_restored_canonical_plan_disconnected",
    )
    rendered = checkpoints["active_plan_after_update"]
    restored_plan = restored_plans[0]
    require(
        restored_plan["plan_id"] == rendered["active_plan_id"] == rendered["rendered_map_plan_id"]
        and restored_plan["content_hash"]
        == assertions["accepted_plan_content_hash"]
        == rendered["active_plan_content_hash"]
        and core._completed_race_geometry_hash(restored_plan) == rendered["rendered_geometry_hash"]
        and {feature["properties"]["stop_id"] for feature in restored_plan["stops"]["features"]}
        == set(rendered["rendered_marker_ids"]),
        "build_restored_map_disconnected",
    )
    reload_input = core._load(base / "reload_control_input.json")
    recovery = core._load(base / "product_client_build_mismatch_visible_recovery.json")
    ready = core._load(base / "generation_b/active_plan_after_update.readiness.json")
    require(
        reload_input["trusted"] is True
        and reload_input["selector"] == "#reload-current-build"
        and reload_input["client_build"] == assertions["first_build_id"]
        and reload_input["path"] == "/app/itinerary"
        and reload_input["title"] == recovery["text"]
        and stamp(recovery["captured_at"])
        <= stamp(reload_input["clicked_at"])
        <= stamp(ready["screenshot_started_at"]),
        "build_reload_control_unbound",
    )
    browser_errors = core._load(base / "browser_errors.json")
    require(
        browser_errors["page_errors"] == browser_errors["request_failures"] == []
        and all(
            "server responded with a status of" in message
            and any(" " + str(status) + " " in message for status in (409, 503))
            for message in browser_errors["console_errors"]
        ),
        "build_unexpected_browser_errors",
    )
    require(not any(row["url"].endswith("/copilot/messages") for row in records), "build_unexpected_chat")
    return witnesses


def verify_scenarios(root, *, mode, run_id, process):
    try:
        require(
            mode
            in {
                "conversation",
                "combined-conflicts",
                "owned-edits",
                "build-coherence",
                "bootstrap-recovery",
                "optional-evidence-fault",
                "preview-geography-fault",
                "queued-preview",
            }
            and process["exit_code"] == 0
            and type(process["pid"]) is int
            and process["pid"] > 0
            and process["outer_app_server_started"] is False,
            "test_process_invalid",
        )
        nodes = _junit(root, mode)
        core._verify_text_security(root)
        if mode == "owned-edits":
            from verify_e2e1_owned_edits import verify_scenarios as owned_scenarios

            witnesses = owned_scenarios(root, run_id, process, nodes)
        elif mode == "bootstrap-recovery":
            from verify_e2e1_bootstrap_recovery import verify_scenarios as bootstrap_scenarios

            witnesses = bootstrap_scenarios(root, run_id, process, nodes)
        elif mode == "optional-evidence-fault":
            from verify_e2e1_optional_evidence_fault import verify_scenarios as optional_scenarios

            witnesses = optional_scenarios(root, run_id, process, nodes)
        elif mode == "preview-geography-fault":
            from verify_e2e1_preview_geography_fault import verify_scenarios as geography_scenarios

            witnesses = geography_scenarios(root, run_id, process, nodes)
        else:
            witnesses = (
                _build(root, run_id, process, nodes)
                if mode == "build-coherence"
                else _conversation(
                    root,
                    run_id,
                    process,
                    nodes,
                    cases=QUEUED_PREVIEW_CASES
                    if mode == "queued-preview"
                    else COMBINED_CONFLICT_CASES
                    if mode == "combined-conflicts"
                    else None,
                )
            )
        return {
            "verified": True,
            "provider_calls": 0,
            "test_count": len(nodes),
            "lifecycles": witnesses,
            "live_openai_connectivity_proven": False,
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, StopIteration, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_self_evidence_malformed") from error


def verify(root):
    from run_e2e1_self_managed import MODES, SCHEMA, files, render_report
    from run_e2e1_vertical_slice import _source_rows

    root = root.resolve()
    try:
        manifest = core._load(root / "file_io_manifest.json")
        require(manifest["schema_version"] == SCHEMA and manifest["mode"] in MODES, "manifest_schema_invalid")
        seed = dict(manifest)
        digest = seed.pop("content_hash")
        require(digest == core._canonical_sha256(seed), "manifest_hash_invalid")
        spec = MODES[manifest["mode"]]
        require(
            manifest["scenarios"] == [spec["scenario"]]
            and manifest["base_url"] == "http://127.0.0.1:8127"
            and manifest["assertions_passed"] is True
            and manifest["pytest_exit_code"] == 0
            and manifest["provider_calls"] == 0
            and manifest["source_unchanged"] is True
            and manifest["source_inputs"] == _source_rows(spec["browser_test"]),
            "manifest_run_binding_invalid",
        )
        require(manifest["files"] == files(root), "manifest_file_binding_invalid")
        process = core._load(root / "test_process.json")
        require(
            process == manifest["test_process"]
            and process["command"] == manifest["command"]
            and process["command"][1:4] == ["-m", "pytest", spec["browser_test"]]
            and "--junitxml" in process["command"]
            and Path(process["command"][process["command"].index("--junitxml") + 1]).resolve()
            == root / "pytest_junit.xml"
            and stamp(process["completed_at"]) <= stamp(manifest["generated_at"]),
            "manifest_process_binding_invalid",
        )
        result = verify_scenarios(root, mode=manifest["mode"], run_id=manifest["run_id"], process=process)
        require(result == manifest["verification"], "manifest_scenario_projection_invalid")
        require(
            (root / "exact_io_report.md").read_text(encoding="utf-8") == render_report(manifest),
            "report_binding_invalid",
        )
        return {
            **result,
            "manifest_content_hash": digest,
            "manifest_sha256": core._sha256(root / "file_io_manifest.json"),
            "report_sha256": core._sha256(root / "exact_io_report.md"),
            "artifact_files": len(manifest["files"]),
            "source_inputs": len(manifest["source_inputs"]),
            "mode": manifest["mode"],
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_self_manifest_malformed") from error
