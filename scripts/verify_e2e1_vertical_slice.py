"""Independently verify a frozen E2E-1 exact-I/O evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from e2e1_source_universe import deterministic_source_inputs  # noqa: E402 - script bootstrap above

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "file_io_manifest.json"
REPORT_NAME = "exact_io_report.md"
SCENARIOS = (
    "registered_replacement",
    "poi_discovery",
    "unsupported_removal",
    "ineligible_duration",
    "stale_response",
    "accepted_reload",
)
SOURCE_INPUTS = deterministic_source_inputs(
    "tests/product_app/browser/test_e2e1_vertical_slice_live.py"
)
MAP_RECOVERY_BROWSER_TEST = "tests/product_app/browser/test_map_initialization_recovery_live.py"
COMPLETED_RACE_BROWSER_TEST = "tests/product_app/browser/test_completed_preview_race_live.py"
WORKSPACE_STABILITY_BROWSER_TEST = "tests/product_app/browser/test_workspace_stability_live.py"
DIRECT_EDITS_BROWSER_TEST = "tests/product_app/browser/test_direct_edit_journeys_live.py"
ADDITION_BROWSER_TEST = "tests/product_app/browser/test_addition_continuation_live.py"
MOBILE_CHAT_BROWSER_TEST = "tests/product_app/browser/test_mobile_chat_journey_live.py"
WAYPOINT_BROWSER_TEST = "tests/product_app/browser/test_waypoint_pointer_live.py"
CONTROLS_BROWSER_TEST = "tests/product_app/browser/test_capability_controls_live.py"
EVALUATED_BROWSER_TEST = "tests/product_app/browser/test_evaluated_variants_live.py"
WEBGL_BROWSER_TEST = "tests/product_app/browser/test_webgl_recovery_live.py"
CHAT_CLOSURE_BROWSER_TEST = "tests/product_app/browser/test_chat_runtime_closure_live.py"
REMOUNT_BROWSER_TEST = "tests/product_app/browser/test_same_plan_remount_live.py"
WAYPOINT_CONTEXTS_BROWSER_TEST = "tests/product_app/browser/test_waypoint_contexts_live.py"
CANDIDATE_DAYS_BROWSER_TEST = "tests/product_app/browser/test_candidate_day_matrix_live.py"
WORKSPACE_VIEWPORTS = ((1440, 900), (1280, 800), (1024, 768), (390, 844))
MANIFEST_KEYS = {
    "assertions_passed",
    "base_url",
    "command",
    "content_hash",
    "files",
    "generated_at",
    "health",
    "llm_input_output_policy",
    "llm_records",
    "provider_calls",
    "provider_isolation",
    "pytest_exit_code",
    "scenarios",
    "schema_version",
    "source_inputs",
}
FILE_KEYS = {"path", "role", "size_bytes", "sha256", "exact_content_location"}
SOURCE_KEYS = {"path", "size_bytes", "sha256"}
LLM_ITEM_KEYS = {"path", "record"}
COMMON_LLM_KEYS = {"llm_input", "llm_output", "provider_call_count", "reason"}
ISOLATION_KEYS = {"artifact", "run_id", "server_pid", "record"}
ISOLATION_RECORD_KEYS = {
    "schema_version", "run_id", "process_id", "scope", "guard_boundaries",
    "installed_at", "finalized_at", "finalized", "effective_adapter",
    "actual_adapter", "api_key_disabled", "app_initialized", "server_started",
    "server_stopped", "blocked_attempts", "sent_provider_calls",
    "allowed_local_connections", "failure_code",
}
ISOLATION_BOUNDARIES = ["httpx.Client.send", "httpx.AsyncClient.send", "socket.connect"]
UNSUPPORTED_TEST = (
    "tests/product_app/test_openai_copilot.py::"
    "test_removal_language_is_rejected_before_provider_call"
)
TARGET = "bixby_creek_bridge_viewpoint"
CANDIDATE = "pfeiffer_big_sur_state_park"
PARENT_MARKERS = {
    "bixby_creek_bridge_viewpoint",
    "golden_gate_bridge",
    "griffith_observatory",
    "kings_canyon_national_park",
    "sequoia_national_park",
    "stanford_university_main_quad",
    "stearns_wharf",
    "yosemite_national_park",
}
CHILD_MARKERS = {
    "golden_gate_bridge",
    "griffith_observatory",
    "kings_canyon_national_park",
    "pfeiffer_big_sur_state_park",
    "sequoia_national_park",
    "stanford_university_main_quad",
    "stearns_wharf",
    "yosemite_national_park",
}


class E2E1VerificationError(RuntimeError):
    """Stable evidence-verification failure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise E2E1VerificationError(code)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "e2e1_json_object_required")
    return value


def _sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        data_lines = [
            line.removeprefix("data:").lstrip()
            for line in frame.splitlines()
            if line.startswith("data:")
        ]
        _require(bool(data_lines), "e2e1_copilot_stream_frame_invalid")
        payload = json.loads("\n".join(data_lines))
        _require(isinstance(payload, dict), "e2e1_copilot_stream_event_invalid")
        events.append(payload)
    return events


def _verify_copilot_stream(
    events: object,
    *,
    expected_client_message_id: object,
    expected_response: object | None = None,
) -> str:
    _require(isinstance(events, list) and len(events) == 3, "e2e1_copilot_stream_length_invalid")
    _require(
        isinstance(expected_client_message_id, str)
        and re.fullmatch(r"client_message_[0-9a-f]{32}", expected_client_message_id) is not None,
        "e2e1_copilot_client_message_id_invalid",
    )
    expected_types = ["message.accepted", "assistant.status", "turn.completed"]
    expected_fields = [
        {"schema_version", "type", "sequence", "request_id", "client_message_id"},
        {
            "schema_version", "type", "sequence", "request_id", "client_message_id",
            "state", "message",
        },
        {"schema_version", "type", "sequence", "request_id", "client_message_id", "payload"},
    ]
    for index, event in enumerate(events):
        _require(isinstance(event, dict), "e2e1_copilot_stream_event_invalid")
        _require(set(event) == expected_fields[index], "e2e1_copilot_stream_event_fields_invalid")
        _require(event["schema_version"] == "copilot-stream-v1", "e2e1_copilot_stream_schema_invalid")
        _require(event["type"] == expected_types[index], "e2e1_copilot_stream_type_invalid")
        _require(event["sequence"] == index + 1, "e2e1_copilot_stream_sequence_invalid")
        _require(
            event["client_message_id"] == expected_client_message_id,
            "e2e1_copilot_stream_client_message_id_mismatch",
        )
    request_id = events[0]["request_id"]
    _require(
        isinstance(request_id, str)
        and re.fullmatch(r"copilot_request_[0-9a-f]{32}", request_id) is not None
        and all(event["request_id"] == request_id for event in events),
        "e2e1_copilot_stream_request_id_invalid",
    )
    _require(
        events[1]["state"] == "interpreting"
        and events[1]["message"]
        == "Reading the current itinerary and validating a typed suggestion.",
        "e2e1_copilot_stream_status_invalid",
    )
    _require(isinstance(events[2]["payload"], dict), "e2e1_copilot_stream_payload_invalid")
    if expected_response is not None:
        _require(events[2]["payload"] == expected_response, "e2e1_copilot_stream_payload_mismatch")
    turn = events[2]["payload"].get("turn")
    _require(
        isinstance(turn, dict)
        and turn.get("client_message_id") == expected_client_message_id,
        "e2e1_copilot_stream_turn_identity_mismatch",
    )
    return request_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: object) -> Path:
    _require(isinstance(relative, str) and relative, "e2e1_relative_path_invalid")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and ".." not in pure.parts, "e2e1_relative_path_invalid")
    path = (root / Path(*pure.parts)).resolve()
    _require(path == root or root in path.parents, "e2e1_relative_path_escape")
    return path


def _role(path: Path) -> str:
    if path.name.endswith("llm_io.json"):
        return "exact_llm_input_output"
    return "exact_input" if "request" in path.name else "exact_output"


def render_report(manifest: dict[str, Any]) -> str:
    isolation_record = manifest.get("provider_isolation", {}).get("record")
    blocked_attempts = (
        len(isolation_record["blocked_attempts"])
        if isinstance(isolation_record, dict)
        and isinstance(isolation_record.get("blocked_attempts"), list)
        else "unknown"
    )
    lines = [
        "# E2E-1 Exact File and LLM Input/Output Report",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Command: {' '.join(manifest['command'])}",
        f"- Assertions passed: {str(manifest['assertions_passed']).lower()}",
        f"- Provider calls: {manifest['provider_calls']}",
        f"- Blocked provider attempts: {blocked_attempts}",
        "- Provider isolation: provider_isolation.json (application-server Python transports)",
        f"- Manifest content hash: {manifest['content_hash']}",
        "",
        "Every exact file is preserved at the linked content location. "
        "Large responses are linked rather than copied or truncated.",
        "",
        "## Exact file input and output",
        "",
        "| Role | Path | Bytes | SHA-256 | Exact content |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in manifest["files"]:
        location = row["exact_content_location"]
        lines.append(
            f"| {row['role']} | {row['path']} | {row['size_bytes']} | "
            f"{row['sha256']} | [{location}]({location}) |"
        )
    lines += [
        "",
        "## Exact source inputs",
        "",
        "| Path | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for row in manifest["source_inputs"]:
        lines.append(f"| {row['path']} | {row['size_bytes']} | {row['sha256']} |")
    lines += ["", "## Exact LLM input and output", ""]
    for item in manifest["llm_records"]:
        lines += [
            f"### {item['path']}",
            "",
            "Exact JSON:",
            "",
            json.dumps(item["record"], indent=2, sort_keys=True),
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def _verify_root(manifest: dict[str, Any]) -> datetime:
    _require(set(manifest) == MANIFEST_KEYS, "e2e1_manifest_fields_invalid")
    _require(
        manifest.get("schema_version") == "e2e1-file-io-manifest-v3",
        "e2e1_manifest_schema_invalid",
    )
    seed = dict(manifest)
    expected_hash = seed.pop("content_hash")
    _require(expected_hash == _canonical_sha256(seed), "e2e1_manifest_hash_mismatch")
    try:
        generated_at = datetime.fromisoformat(str(manifest["generated_at"]))
    except ValueError as error:
        raise E2E1VerificationError("e2e1_generated_at_invalid") from error
    _require(generated_at.tzinfo is not None, "e2e1_generated_at_timezone_missing")
    base_url = re.fullmatch(
        r"http://127\.0\.0\.1:([1-9]\d{0,4})",
        str(manifest["base_url"]),
    )
    _require(
        base_url is not None and int(base_url.group(1)) <= 65_535,
        "e2e1_base_url_invalid",
    )
    _require(manifest["assertions_passed"] is True, "e2e1_assertions_not_passed")
    _require(manifest["pytest_exit_code"] == 0, "e2e1_pytest_exit_invalid")
    _require(
        type(manifest["provider_calls"]) is int and manifest["provider_calls"] == 0,
        "e2e1_provider_calls_nonzero",
    )
    _verify_provider_isolation_record(manifest, generated_at)
    recovery = manifest["scenarios"] == ["map_initialization_recovery"]
    completed_race = manifest["scenarios"] == ["completed_preview_race"]
    stability = manifest["scenarios"] == ["workspace_stability"]
    direct_edits = manifest["scenarios"] == ["direct_edits"]
    addition = manifest["scenarios"] == ["addition_continuation"]
    mobile_chat = manifest["scenarios"] == ["mobile_chat"]
    waypoint = manifest["scenarios"] == ["waypoint_pointer"]
    controls = manifest["scenarios"] == ["capability_controls"]
    evaluated = manifest["scenarios"] == ["evaluated_variants"]
    webgl = manifest["scenarios"] == ["webgl_recovery"]
    chat_closure = manifest["scenarios"] == ["chat_closure"]
    remount = manifest["scenarios"] == ["same_plan_remount"]
    waypoint_contexts = manifest["scenarios"] == ["waypoint_contexts"]
    candidate_days = manifest["scenarios"] == ["candidate_day_matrix"]
    _require(recovery or completed_race or stability or direct_edits or addition or mobile_chat or waypoint or controls or evaluated or webgl or chat_closure or remount or waypoint_contexts or candidate_days or tuple(manifest["scenarios"]) == SCENARIOS, "e2e1_scenario_universe_invalid")
    health = manifest.get("health")
    _require(
        isinstance(health, dict)
        and set(health) == {"build_id", "map", "routing", "copilot"}
        and isinstance(health["build_id"], str)
        and health["map"] == "maplibre_ready"
        and health["routing"] == "runtime_osrm_ready"
        and health["copilot"] == "deterministic_adapter_selected",
        "e2e1_health_binding_invalid",
    )
    command = manifest.get("command")
    _require(
        isinstance(command, list)
        and (CANDIDATE_DAYS_BROWSER_TEST if candidate_days else WAYPOINT_CONTEXTS_BROWSER_TEST if waypoint_contexts else REMOUNT_BROWSER_TEST if remount else CHAT_CLOSURE_BROWSER_TEST if chat_closure else WEBGL_BROWSER_TEST if webgl else EVALUATED_BROWSER_TEST if evaluated else CONTROLS_BROWSER_TEST if controls else WAYPOINT_BROWSER_TEST if waypoint else MOBILE_CHAT_BROWSER_TEST if mobile_chat else DIRECT_EDITS_BROWSER_TEST if direct_edits else ADDITION_BROWSER_TEST if addition else WORKSPACE_STABILITY_BROWSER_TEST if stability else COMPLETED_RACE_BROWSER_TEST if completed_race else MAP_RECOVERY_BROWSER_TEST if recovery else "tests/product_app/browser/test_e2e1_vertical_slice_live.py") in command
        and "-q" in command
        and "no:cacheprovider" in command,
        "e2e1_command_binding_invalid",
    )
    return generated_at


def _isolation_timestamp(value: object) -> datetime:
    _require(isinstance(value, str), "e2e1_isolation_timestamp_invalid")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise E2E1VerificationError("e2e1_isolation_timestamp_invalid") from error
    _require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timedelta(0),
        "e2e1_isolation_timestamp_not_utc",
    )
    return timestamp


def _verify_provider_isolation_record(
    manifest: dict[str, Any], generated_at: datetime,
) -> None:
    isolation = manifest.get("provider_isolation")
    _require(
        isinstance(isolation, dict) and set(isolation) == ISOLATION_KEYS,
        "e2e1_isolation_fields_invalid",
    )
    _require(
        isolation["artifact"] == "provider_isolation.json",
        "e2e1_isolation_artifact_invalid",
    )
    _require(
        isinstance(isolation["run_id"], str)
        and bool(isolation["run_id"].strip())
        and type(isolation["server_pid"]) is int
        and isolation["server_pid"] > 0,
        "e2e1_isolation_identity_invalid",
    )
    record = isolation["record"]
    _require(
        isinstance(record, dict) and set(record) == ISOLATION_RECORD_KEYS,
        "e2e1_isolation_record_fields_invalid",
    )
    _require(
        record["schema_version"] == "e2e1-provider-isolation-v1"
        and record["scope"] == "application_server"
        and record["guard_boundaries"] == ISOLATION_BOUNDARIES,
        "e2e1_isolation_scope_invalid",
    )
    _require(
        record["run_id"] == isolation["run_id"]
        and type(record["process_id"]) is int
        and record["process_id"] == isolation["server_pid"],
        "e2e1_isolation_identity_mismatch",
    )
    _require(
        all(record[key] is True for key in (
            "finalized", "api_key_disabled", "app_initialized",
            "server_started", "server_stopped",
        ))
        and record["failure_code"] is None,
        "e2e1_isolation_incomplete",
    )
    _require(
        record["effective_adapter"] == "deterministic"
        and record["actual_adapter"] == "DeterministicCopilotAdapter",
        "e2e1_isolation_adapter_invalid",
    )
    installed_at = _isolation_timestamp(record["installed_at"])
    finalized_at = _isolation_timestamp(record["finalized_at"])
    _require(
        installed_at <= finalized_at <= generated_at,
        "e2e1_isolation_timestamp_order_invalid",
    )
    _require(
        type(record["sent_provider_calls"]) is int
        and record["sent_provider_calls"] == 0
        and type(manifest.get("provider_calls")) is int
        and manifest["provider_calls"] == record["sent_provider_calls"],
        "e2e1_isolation_provider_calls_invalid",
    )
    _require(
        type(record["allowed_local_connections"]) is int
        and record["allowed_local_connections"] >= 1,
        "e2e1_isolation_local_connections_invalid",
    )
    attempts = record["blocked_attempts"]
    _require(isinstance(attempts, list), "e2e1_isolation_attempts_invalid")
    for index, attempt in enumerate(attempts, 1):
        _require(
            isinstance(attempt, dict)
            and set(attempt) == {"sequence", "boundary", "reason", "at"}
            and type(attempt["sequence"]) is int
            and attempt["sequence"] == index
            and attempt["boundary"] in ISOLATION_BOUNDARIES
            and attempt["reason"] == "non_loopback_target",
            "e2e1_isolation_attempt_invalid",
        )
        _require(
            installed_at <= _isolation_timestamp(attempt["at"]) <= finalized_at,
            "e2e1_isolation_attempt_time_invalid",
        )
    _require(not attempts, "e2e1_isolation_attempts_blocked")


def _verify_provider_isolation(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "provider_isolation.json"
    _require(path.is_file(), "e2e1_isolation_artifact_missing")
    _require(
        _canonical_sha256(_load(path))
        == _canonical_sha256(manifest["provider_isolation"]["record"]),
        "e2e1_isolation_file_projection_mismatch",
    )


def _verify_files(root: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("files")
    _require(isinstance(rows, list) and rows, "e2e1_file_rows_invalid")
    declared: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == FILE_KEYS, "e2e1_file_row_invalid")
        relative = row["path"]
        _require(relative not in declared, "e2e1_duplicate_file_path")
        declared.add(relative)
        path = _safe_path(root, relative)
        _require(path.is_file(), "e2e1_artifact_missing")
        _require(row["exact_content_location"] == relative, "e2e1_content_location_invalid")
        _require(row["role"] == _role(path), "e2e1_artifact_role_invalid")
        _require(path.stat().st_size == row["size_bytes"], "e2e1_artifact_size_mismatch")
        _require(_sha256(path) == row["sha256"], "e2e1_artifact_hash_mismatch")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {MANIFEST_NAME, REPORT_NAME}
    }
    _require(actual == declared, "e2e1_artifact_inventory_mismatch")


def _verify_sources(manifest: dict[str, Any], generated_at: datetime) -> None:
    rows = manifest.get("source_inputs")
    _require(isinstance(rows, list), "e2e1_source_rows_invalid")
    expected = (deterministic_source_inputs(MAP_RECOVERY_BROWSER_TEST)
                if manifest.get("scenarios") == ["map_initialization_recovery"] else SOURCE_INPUTS)
    if manifest.get("scenarios") == ["completed_preview_race"]:
        expected = deterministic_source_inputs(COMPLETED_RACE_BROWSER_TEST)
    if manifest.get("scenarios") == ["workspace_stability"]:
        expected = deterministic_source_inputs(WORKSPACE_STABILITY_BROWSER_TEST)
    if manifest.get("scenarios") == ["direct_edits"]:
        expected = deterministic_source_inputs(DIRECT_EDITS_BROWSER_TEST)
    if manifest.get("scenarios") == ["addition_continuation"]:
        expected = deterministic_source_inputs(ADDITION_BROWSER_TEST)
    if manifest.get("scenarios") == ["mobile_chat"]:
        expected = deterministic_source_inputs(MOBILE_CHAT_BROWSER_TEST)
    if manifest.get("scenarios") == ["waypoint_pointer"]:
        expected = deterministic_source_inputs(WAYPOINT_BROWSER_TEST)
    if manifest.get("scenarios") == ["capability_controls"]:
        expected = deterministic_source_inputs(CONTROLS_BROWSER_TEST)
    if manifest.get("scenarios") == ["evaluated_variants"]:
        expected = deterministic_source_inputs(EVALUATED_BROWSER_TEST)
    if manifest.get("scenarios") == ["webgl_recovery"]:
        expected = deterministic_source_inputs(WEBGL_BROWSER_TEST)
    if manifest.get("scenarios") == ["chat_closure"]:
        expected = deterministic_source_inputs(CHAT_CLOSURE_BROWSER_TEST)
    if manifest.get("scenarios") == ["same_plan_remount"]:
        expected = deterministic_source_inputs(REMOUNT_BROWSER_TEST)
    if manifest.get("scenarios") == ["waypoint_contexts"]:
        expected = deterministic_source_inputs(WAYPOINT_CONTEXTS_BROWSER_TEST)
    if manifest.get("scenarios") == ["candidate_day_matrix"]:
        expected = deterministic_source_inputs(CANDIDATE_DAYS_BROWSER_TEST)
    _require([row.get("path") for row in rows] == list(expected), "e2e1_source_universe_invalid")
    for row in rows:
        _require(isinstance(row, dict) and set(row) == SOURCE_KEYS, "e2e1_source_row_invalid")
        path = _safe_path(ROOT, row["path"])
        _require(path.is_file(), "e2e1_source_missing")
        _require(path.stat().st_size == row["size_bytes"], "e2e1_source_size_mismatch")
        _require(_sha256(path) == row["sha256"], "e2e1_source_hash_mismatch")
        _require(
            datetime.fromtimestamp(path.stat().st_mtime, generated_at.tzinfo) <= generated_at,
            "e2e1_source_postdates_run",
        )


def _expected_llm_keys(relative: str) -> set[str]:
    keys = set(COMMON_LLM_KEYS)
    if relative == "unsupported_removal/llm_io.json":
        keys.add("verification_test")
    if relative in {"chat_closure/original/llm_io.json", "chat_closure/accepted_child/llm_io.json"}:
        keys.add("actual_openai_connectivity_claimed")
    return keys


def _verify_llm_records(root: Path, manifest: dict[str, Any]) -> None:
    items = manifest.get("llm_records")
    _require(isinstance(items, list), "e2e1_llm_records_invalid")
    expected_paths = sorted(
        [f"{scenario}/llm_io.json" for scenario in SCENARIOS]
        + ["registered_replacement/one_click_llm_io.json"]
    )
    if manifest.get("scenarios") == ["map_initialization_recovery"]:
        expected_paths = [f"map_initialization_recovery/{context}/llm_io.json" for context in ("accepted_child", "original")]
    if manifest.get("scenarios") == ["completed_preview_race"]:
        expected_paths = ["completed_preview_race/llm_io.json"]
    if manifest.get("scenarios") == ["workspace_stability"]:
        expected_paths = sorted(f"workspace_stability/{width}x{height}/llm_io.json" for width, height in WORKSPACE_VIEWPORTS)
    if manifest.get("scenarios") == ["direct_edits"]:
        expected_paths = sorted(f"direct_edits/{kind}/llm_io.json" for kind in ("combined", "duration", "time_window", "role", "order"))
    if manifest.get("scenarios") == ["addition_continuation"]:
        expected_paths = ["addition_continuation/llm_io.json"]
    if manifest.get("scenarios") == ["mobile_chat"]:
        expected_paths = ["mobile_chat/llm_io.json"]
    if manifest.get("scenarios") == ["waypoint_pointer"]:
        expected_paths = ["waypoint_pointer/llm_io.json"]
    if manifest.get("scenarios") == ["capability_controls"]:
        expected_paths = ["capability_controls/accepted_child/llm_io.json", "capability_controls/original/llm_io.json"]
    if manifest.get("scenarios") == ["evaluated_variants"]:
        expected_paths = ["evaluated_variants/accepted_child/llm_io.json", "evaluated_variants/original/llm_io.json"]
    if manifest.get("scenarios") == ["webgl_recovery"]:
        expected_paths = ["webgl_recovery/llm_io.json"]
    if manifest.get("scenarios") == ["chat_closure"]:
        expected_paths = ["chat_closure/accepted_child/llm_io.json", "chat_closure/original/llm_io.json"]
    if manifest.get("scenarios") == ["same_plan_remount"]:
        expected_paths = ["same_plan_remount/llm_io.json"]
    if manifest.get("scenarios") == ["waypoint_contexts"]:
        expected_paths = ["waypoint_contexts/accepted_child/llm_io.json", "waypoint_contexts/original/llm_io.json"]
    if manifest.get("scenarios") == ["candidate_day_matrix"]:
        expected_paths = ["candidate_day_matrix/llm_io.json"]
    _require([item.get("path") for item in items] == expected_paths, "e2e1_llm_universe_invalid")
    for item in items:
        _require(isinstance(item, dict) and set(item) == LLM_ITEM_KEYS, "e2e1_llm_item_invalid")
        relative = item["path"]
        record = item["record"]
        _require(isinstance(record, dict), "e2e1_llm_record_invalid")
        _require(set(record) == _expected_llm_keys(relative), "e2e1_llm_fields_invalid")
        _require(record["provider_call_count"] == 0, "e2e1_llm_calls_nonzero")
        _require(record["llm_input"] is None, "e2e1_llm_input_persisted")
        _require(record["llm_output"] is None, "e2e1_llm_output_persisted")
        _require(isinstance(record["reason"], str) and record["reason"].strip(), "e2e1_llm_reason_invalid")
        if relative == "unsupported_removal/llm_io.json":
            _require(record["verification_test"] == UNSUPPORTED_TEST, "e2e1_llm_test_binding_invalid")
        if "actual_openai_connectivity_claimed" in record:
            _require(record["actual_openai_connectivity_claimed"] is False, "e2e1_llm_live_connectivity_falsely_claimed")
        _require(_load(_safe_path(root, relative)) == record, "e2e1_llm_file_projection_mismatch")


def _verify_product_http_capture(row: dict[str, Any]) -> None:
    _require(row.get("schema_version") == "e2e1-product-http-io-v1"
             and all(row.get(name) is False for name in (
                 "request_headers_persisted", "response_headers_persisted",
                 "authorization_persisted", "session_token_persisted")), "e2e1_http_capture_headers_invalid")
    redactions = row.get("credential_redactions")
    _require(isinstance(redactions, dict) and set(redactions) == {"request", "response"},
             "e2e1_http_redactions_invalid")
    for side in ("request", "response"):
        paths = redactions[side]
        body = row.get(f"{side}_body_utf8")
        original = row.get(f"{side}_original_body_sha256")
        _require(isinstance(body, str) and isinstance(paths, list)
                 and len(paths) == len(set(paths))
                 and all(isinstance(path, str) and path.startswith("/") for path in paths)
                 and re.fullmatch(r"[0-9a-f]{64}", str(original)) is not None
                 and hashlib.sha256(body.encode()).hexdigest() == row.get(f"{side}_body_sha256"),
                 "e2e1_http_capture_body_invalid")
        if not paths:
            _require(original == row[f"{side}_body_sha256"], "e2e1_http_undisclosed_body_change")
        else:
            value = json.loads(body)
            for path in paths:
                selected = value
                for part in path[1:].split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    selected = selected[int(key)] if isinstance(selected, list) else selected[key]
                _require(selected == "OMITTED_SECRET", "e2e1_http_redaction_projection_invalid")
    _require(row.get("body_capture") == (
        "credential_redacted" if any(redactions.values()) else "exact_utf8"),
        "e2e1_http_capture_disclosure_invalid")


def _verify_report(root: Path, manifest: dict[str, Any]) -> None:
    report_path = root / REPORT_NAME
    _require(report_path.is_file(), "e2e1_report_missing")
    _require(
        report_path.read_text(encoding="utf-8") == render_report(manifest),
        "e2e1_report_bytes_mismatch",
    )


def _verify_text_security(root: Path) -> None:
    sensitive_fields = {"authorization", "api_key", "openai_api_key", "cookie", "set-cookie", "x-session-token", "mutation_token", "session_token", "token_verifier"}

    def inspect_json(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in sensitive_fields:
                    _require(
                        child == "OMITTED_SECRET",
                        "e2e1_structured_secret_persisted",
                    )
                inspect_json(child)
        elif isinstance(value, list):
            for child in value:
                inspect_json(child)
        elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return
            inspect_json(decoded)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".log"}:
            continue
        data = path.read_text(encoding="utf-8", errors="replace")
        _require(
            re.search(r"(?i)authorization\s*[:=]\s*bearer\s+\S+", data) is None,
            "e2e1_authorization_persisted",
        )
        _require(
            re.search(r"(?i)\bsk-(?:proj-)?[a-z0-9_-]{12,}", data) is None,
            "e2e1_api_key_persisted",
        )
        for match in re.finditer(
            r"(?i)x-session-token[\"']?\s*[:=]\s*[\"']?([a-z0-9._-]+)",
            data,
        ):
            _require(match.group(1).lower() == "omitted_secret", "e2e1_session_token_persisted")
        if path.suffix.lower() == ".json":
            try:
                inspect_json(json.loads(data))
            except json.JSONDecodeError as exc:
                raise E2E1VerificationError("e2e1_json_security_parse_invalid") from exc


def _valid_request_target(
    url: object,
    *,
    base_url: str,
    method: object,
    expected_method: str,
    endpoint: str,
    query_keys: set[str] | None = None,
) -> bool:
    if not isinstance(url, str) or method != expected_method:
        return False
    parsed = urlparse(url)
    base = urlparse(base_url)
    if (
        parsed.scheme != base.scheme
        or parsed.netloc != base.netloc
        or not re.fullmatch(
            rf"/api/sessions/session_[0-9a-f]{{32}}{re.escape(endpoint)}",
            parsed.path,
        )
        or parsed.fragment
    ):
        return False
    actual_query_keys = set(parse_qs(parsed.query, keep_blank_values=True))
    return actual_query_keys == (query_keys or set())


def _verify_recorded_request_origins(root: Path, base_url: str) -> None:
    expected = {
        "registered_replacement/copilot_message_request.json": (
            "POST",
            "/copilot/messages",
            None,
        ),
        "registered_replacement/draft_request.json": (
            "POST",
            "/draft/operations",
            None,
        ),
        "registered_replacement/one_click_copilot_request.json": (
            "POST",
            "/copilot/messages",
            None,
        ),
        "registered_replacement/one_click_draft_request.json": (
            "POST",
            "/draft/operations",
            None,
        ),
        "registered_replacement/one_click_preview_request.json": (
            "POST",
            "/preview",
            None,
        ),
        "poi_discovery/browser_poi_candidate_request.json": (
            "GET",
            "/map/poi-candidates",
            {
                "expected_revision", "day", "route_leg_id", "max_detour_minutes",
                "limit", "replacement_target_id",
            },
        ),
        "poi_discovery/browser_preview_request.json": ("POST", "/preview", None),
        "unsupported_removal/copilot_removal_request.json": (
            "POST", "/copilot/messages", None,
        ),
        "ineligible_duration/preview_request.json": ("POST", "/preview", None),
        "accepted_reload/preview_http_io.json": ("POST", "/preview", None),
        "accepted_reload/accept_http_io.json": ("POST", "/accept", None),
        "accepted_reload/reload_http_io.json": ("GET", "", None),
        "accepted_reload/conversation_before_http_io.json": ("GET", "/conversation", None),
        "accepted_reload/conversation_after_http_io.json": ("GET", "/conversation", None),
        "accepted_reload/copilot_message_http_io.json": ("POST", "/copilot/messages", None),
    }
    for relative, (method, endpoint, query_keys) in expected.items():
        record = _load(_safe_path(root, relative))
        _require(
            _valid_request_target(
                record.get("url"),
                base_url=base_url,
                method=record.get("method"),
                expected_method=method,
                endpoint=endpoint,
                query_keys=query_keys,
            ),
            "e2e1_recorded_request_origin_mismatch",
        )


def _request_session_id(record: dict[str, Any]) -> str:
    url = record.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    match = (
        re.fullmatch(
            r"/api/sessions/(session_[0-9a-f]{32})/.+",
            parsed.path,
        )
        if parsed is not None
        else None
    )
    _require(match is not None, "e2e1_request_session_id_invalid")
    return match.group(1)


def _map_markers(value: dict[str, Any]) -> set[str]:
    markers = value.get("rendered_marker_ids")
    _require(isinstance(markers, list) and len(markers) == len(set(markers)), "e2e1_markers_invalid")
    return set(markers)


def _verify_registered(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    base = root / "registered_replacement"
    copilot_request = read(base / "copilot_message_request.json")
    copilot_body = json.loads(copilot_request["body_utf8"])
    copilot_events = json.loads(
        (base / "copilot_stream_events.json").read_text(encoding="utf-8")
    )
    draft_request = read(base / "draft_request.json")
    draft_body = json.loads(draft_request["body_utf8"])
    draft_response = read(base / "draft_response.json")
    preview_response = read(base / "preview_response.json")
    response = preview_response["proposal"]
    trace = read(base / "trace.json")
    diff = read(base / "plan_diff.json")
    evaluation = read(base / "evaluation_summary.json")
    map_ack = read(base / "map_render_ack.json")
    browser = read(base / "browser_assertions.json")
    child = response["child_plan_id"]
    content = response["child_plan_content_hash"]
    operation = {
        "type": "replace_nearby",
        "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
    }
    _require(
        copilot_request["headers"]["Accept"] == "text/event-stream"
        and copilot_request["headers"]["X-Session-Token"] == "OMITTED_SECRET"
        and copilot_body["message"]
        == "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park",
        "e2e1_registered_copilot_request_invalid",
    )
    _require(
        _verify_copilot_stream(
            copilot_events,
            expected_client_message_id=copilot_body.get("client_message_id"),
        ) == trace["copilot_stream_request_id"]
        and copilot_body["client_message_id"] == trace["copilot_client_message_id"]
        and copilot_events[-1]["payload"]["turn"]["interpretation"]["state"] == "proposal_ready",
        "e2e1_registered_copilot_stream_invalid",
    )
    preview_posts = [
        url for url in browser["post_requests"] if url.endswith("/preview")
    ]
    base_url = _load(root / MANIFEST_NAME)["base_url"]
    _require(
        len(preview_posts) == 1
        and _valid_request_target(
            preview_posts[0],
            base_url=base_url,
            method="POST",
            expected_method="POST",
            endpoint="/preview",
        )
        and _request_session_id(copilot_request)
        == _request_session_id(draft_request)
        == _request_session_id({"url": preview_posts[0]})
        == copilot_events[-1]["payload"]["session"]["session_id"]
        == draft_response["session"]["session_id"]
        == preview_response["session"]["session_id"],
        "e2e1_registered_session_identity_mismatch",
    )
    _require(
        draft_request["headers"]["X-Session-Token"] == "OMITTED_SECRET"
        and {
            field: draft_body.get(field)
            for field in ("type", "target", "parameters")
        }
        == operation
        and draft_body.get("source") == "copilot"
        and isinstance(draft_body.get("expected_revision"), int)
        and draft_response["session"]["draft"][0]["type"] == operation["type"]
        and draft_response["session"]["draft"][0]["target"] == operation["target"]
        and draft_response["session"]["draft"][0]["parameters"] == operation["parameters"],
        "e2e1_registered_draft_invalid",
    )
    _require(
        child
        == trace["child_plan_id"]
        == diff["child_plan_id"]
        == evaluation["plan_id"]
        == map_ack["active_plan_id"]
        == map_ack["rendered_map_plan_id"],
        "e2e1_registered_plan_identity_mismatch",
    )
    _require(
        content
        == trace["child_plan_content_hash"]
        == evaluation["plan_content_hash"]
        == map_ack["active_plan_content_hash"],
        "e2e1_registered_content_identity_mismatch",
    )
    _require(response["diff_id"] == diff["diff_id"] == trace["diff_id"], "e2e1_registered_diff_mismatch")
    _require(
        response["certificate_id"] == evaluation["certificate_id"] == trace["certificate_id"],
        "e2e1_registered_certificate_mismatch",
    )
    route = evaluation["route_validation"]
    _require(
        evaluation["state"] == "eligible"
        and route["required_leg_count"] == 15
        and route["road_validated_leg_count"] == 15
        and route["fallback_leg_count"] == 0,
        "e2e1_registered_evaluation_invalid",
    )
    _require(_map_markers(map_ack) == CHILD_MARKERS, "e2e1_registered_markers_invalid")
    _require(browser["initial"] == browser["staged"], "e2e1_staging_changed_map")
    _require(browser["rendered"] == map_ack, "e2e1_browser_map_projection_mismatch")
    _require(
        sum(url.endswith("/copilot/messages") for url in browser["post_requests"]) == 1
        and sum(url.endswith("/draft/operations") for url in browser["post_requests"]) == 1
        and sum(url.endswith("/preview") for url in browser["post_requests"]) == 1
        and not any(url.endswith("/accept") for url in browser["post_requests"]),
        "e2e1_registered_browser_request_sequence_invalid",
    )
    _require(browser["console_errors"] == [] and browser["request_failures"] == [], "e2e1_browser_errors")
    _require(
        browser["initial"]["rendered_geometry_hash"] != map_ack["rendered_geometry_hash"],
        "e2e1_geometry_did_not_change",
    )
    _verify_one_click_registered(base, read)


def _verify_one_click_registered(
    base: Path,
    read: Callable[[Path], dict[str, Any]],
) -> None:
    copilot_request = read(base / "one_click_copilot_request.json")
    copilot_body = json.loads(copilot_request["body_utf8"])
    stored_events = json.loads(
        (base / "one_click_copilot_stream_events.json").read_text(encoding="utf-8")
    )
    raw_events = _sse_events(
        (base / "one_click_copilot_stream_raw.txt").read_text(encoding="utf-8")
    )
    draft_request = read(base / "one_click_draft_request.json")
    draft_body = json.loads(draft_request["body_utf8"])
    draft_response = read(base / "one_click_draft_response.json")
    preview_request = read(base / "one_click_preview_request.json")
    preview_body = json.loads(preview_request["body_utf8"])
    preview_response = read(base / "one_click_preview_response.json")
    proposal = preview_response["session"]["proposal"]
    map_ack = read(base / "one_click_map_render_ack.json")
    trace = read(base / "one_click_trace.json")
    browser = read(base / "one_click_browser_assertions.json")
    diff = read(base / "one_click_plan_diff.json")
    evaluation = read(base / "one_click_evaluation_summary.json")
    operation = {
        "type": "replace_nearby",
        "target": TARGET,
        "parameters": {"candidate_id": CANDIDATE},
        "source": "copilot",
    }
    _require(
        copilot_request["method"] == "POST"
        and copilot_request["headers"]["Accept"] == "text/event-stream"
        and copilot_request["headers"]["X-Session-Token"] == "OMITTED_SECRET"
        and copilot_body["message"]
        == "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
        and stored_events == raw_events,
        "e2e1_one_click_copilot_request_invalid",
    )
    _verify_copilot_stream(
        stored_events,
        expected_client_message_id=copilot_body.get("client_message_id"),
    )
    _require(
        _request_session_id(copilot_request)
        == _request_session_id(draft_request)
        == _request_session_id(preview_request)
        == stored_events[-1]["payload"]["session"]["session_id"]
        == draft_response["session"]["session_id"]
        == preview_response["session"]["session_id"],
        "e2e1_one_click_session_identity_mismatch",
    )
    _require(
        draft_request["method"] == "POST"
        and draft_request["headers"]["X-Session-Token"] == "OMITTED_SECRET"
        and {key: draft_body.get(key) for key in operation} == operation
        and isinstance(draft_body.get("expected_revision"), int)
        and draft_response["session"]["draft"] == [draft_response["operation"]]
        and {
            key: draft_response["operation"].get(key)
            for key in ("type", "target", "parameters", "source")
        } == operation,
        "e2e1_one_click_draft_invalid",
    )
    canonical_operation = {
        key: value
        for key, value in draft_response["operation"].items()
        if key != "created_at"
    }
    _require(
        preview_request["method"] == "POST"
        and preview_request["headers"]["X-Session-Token"] == "OMITTED_SECRET"
        and set(preview_body) == {"expected_revision"}
        and isinstance(preview_body["expected_revision"], int)
        and proposal["compiled_request"]["operations"] == [canonical_operation]
        and proposal["state"] == "eligible",
        "e2e1_one_click_preview_invalid",
    )
    child = proposal["child_plan_id"]
    content = proposal["child_plan_content_hash"]
    _require(
        child
        == trace["child_plan_id"]
        == diff["child_plan_id"]
        == evaluation["plan_id"]
        == map_ack["active_plan_id"]
        == map_ack["rendered_map_plan_id"],
        "e2e1_one_click_plan_identity_mismatch",
    )
    _require(
        content
        == trace["child_plan_content_hash"]
        == evaluation["plan_content_hash"]
        == map_ack["active_plan_content_hash"],
        "e2e1_one_click_content_identity_mismatch",
    )
    _require(
        proposal["diff_id"] == trace["diff_id"] == diff["diff_id"]
        and proposal["certificate_id"]
        == trace["certificate_id"]
        == evaluation["certificate_id"],
        "e2e1_one_click_evaluation_identity_mismatch",
    )
    parent_identity_fields = {
        "active_plan_id",
        "active_plan_content_hash",
        "rendered_map_plan_id",
        "rendered_geometry_hash",
        "rendered_marker_ids",
    }
    _require(
        all(
            browser["initial"].get(field)
            == browser["parent_while_preview_runs"].get(field)
            for field in parent_identity_fields
        )
        and browser["initial"].get("preview_state") == "PARENT_RENDERED"
        and browser["parent_while_preview_runs"].get("preview_state") == "PREVIEW_RUNNING"
        and browser["rendered"] == map_ack
        and browser["repair_post_sequence"] == ["draft", "preview"]
        and trace["repair_post_sequence"] == ["draft", "preview"]
        and trace["accept_or_keep_invoked"] is False
        and trace["assertions_passed"] is True
        and browser["console_errors"] == []
        and browser["request_failures"] == [],
        "e2e1_one_click_browser_sequence_invalid",
    )
    _require(
        browser["initial"]["rendered_map_plan_id"] == proposal["parent_plan_id"]
        and browser["initial"]["rendered_geometry_hash"] != map_ack["rendered_geometry_hash"]
        and _map_markers(map_ack) == CHILD_MARKERS,
        "e2e1_one_click_map_transition_invalid",
    )


def _verify_poi(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    from verify_e2e1_poi_controls import verify as verify_poi_controls

    base = root / "poi_discovery"
    trace = read(base / "trace.json")
    map_ack = read(base / "map_render_ack.json")
    browser = read(base / "browser_assertions.json")
    verify_poi_controls(base,browser["parent_map"])
    response = read(base / "browser_preview_response.json")["proposal"]
    _require(trace["assertions_passed"] is True, "e2e1_poi_trace_failed")
    _require(
        trace["child_plan_id"] == response["child_plan_id"] == map_ack["rendered_map_plan_id"],
        "e2e1_poi_plan_identity_mismatch",
    )
    _require(trace["geometry_hash"] == map_ack["rendered_geometry_hash"], "e2e1_poi_geometry_mismatch")
    _require(set(browser["candidate_ids"]) == {
        "pfeiffer_big_sur_state_park",
        "point_lobos_state_natural_reserve",
    }, "e2e1_poi_candidate_universe_invalid")
    _require(browser["parent_map"] == browser["staged_map"], "e2e1_poi_staging_changed_map")
    _require(browser["rendered_map"] == map_ack, "e2e1_poi_browser_map_mismatch")
    _require(
        browser["console_errors"] == []
        and browser["request_failures"] == []
        and browser["http_errors"] == []
        and browser["llm_call_made"] is False,
        "e2e1_poi_browser_errors",
    )


def _verify_unsupported(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    base = root / "unsupported_removal"
    request = read(base / "copilot_removal_request.json")
    body = json.loads(request["body_utf8"])
    response = read(base / "copilot_removal_response.json")
    stored_events = json.loads(
        (base / "copilot_stream_events.json").read_text(encoding="utf-8")
    )
    raw_events = _sse_events(
        (base / "copilot_stream_raw.txt").read_text(encoding="utf-8")
    )
    map_ack = read(base / "map_render_ack.json")
    turn = response["turn"]
    intents = turn["interpretation"]["intents"]
    _require(body["message"] == "I do not want to go to Stearns Wharf.", "e2e1_unsupported_input_mismatch")
    _require(
        request["headers"]["Accept"] == "text/event-stream"
        and request["headers"]["X-Session-Token"] == "OMITTED_SECRET",
        "e2e1_unsupported_secret_not_redacted",
    )
    _require(
        isinstance(stored_events, list)
        and stored_events == raw_events
        and _verify_copilot_stream(
            stored_events,
            expected_client_message_id=body.get("client_message_id"),
            expected_response=response,
        ) == stored_events[-1]["request_id"],
        "e2e1_unsupported_copilot_stream_invalid",
    )
    _require(
        turn["user_message"] == body["message"]
        and turn["interpretation"]["state"] == "unsupported"
        and len(intents) == 1
        and intents[0]["type"] == "unsupported_request"
        and response["session"]["draft"] == [],
        "e2e1_unsupported_semantics_invalid",
    )
    _require(
        map_ack["preview_state"] == "PARENT_RENDERED"
        and _map_markers(map_ack) == PARENT_MARKERS,
        "e2e1_unsupported_map_changed",
    )


def _verify_ineligible(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    base = root / "ineligible_duration"
    response = read(base / "preview_response.json")["proposal"]
    diff = read(base / "plan_diff.json")
    evaluation = read(base / "evaluation_summary.json")
    map_ack = read(base / "map_render_ack.json")
    _require(
        response["child_plan_id"] == diff["child_plan_id"] == evaluation["plan_id"],
        "e2e1_ineligible_child_identity_mismatch",
    )
    _require(
        evaluation["state"] == "ineligible"
        and evaluation["acceptance_eligible"] is False
        and "day_time_exceeded" in {item["code"] for item in evaluation["certificate"]["failures"]},
        "e2e1_ineligible_evaluation_invalid",
    )
    _require(
        map_ack["preview_state"] == "PREVIEW_INELIGIBLE"
        and map_ack["rendered_map_plan_id"] == diff["parent_plan_id"]
        and map_ack["rendered_map_plan_id"] != diff["child_plan_id"]
        and _map_markers(map_ack) == PARENT_MARKERS,
        "e2e1_ineligible_parent_not_preserved",
    )
    _require(
        len(diff["duration_changes"]) == 2
        and all(item["to_minutes"] == 480.0 for item in diff["duration_changes"]),
        "e2e1_ineligible_diff_invalid",
    )


def _verify_stale(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    base = root / "stale_response"
    trace = read(base / "trace.json")
    older = read(base / "preview_a_response.json")
    newer = read(base / "preview_b_response.json")["proposal"]
    map_ack = read(base / "map_render_ack.json")
    _require(
        trace["assertions_passed"] is True
        and trace["older_response_ignored"] is True
        and trace["older_response_status"] == 409
        and older == {"detail": "stale_session_revision"},
        "e2e1_stale_response_not_rejected",
    )
    _require(
        map_ack["rendered_map_plan_id"] == newer["child_plan_id"]
        and map_ack["rendered_geometry_hash"] == trace["rendered_geometry_hash"],
        "e2e1_stale_response_overwrote_newer",
    )


def _verify_accepted(root: Path, read: Callable[[Path], dict[str, Any]]) -> None:
    base = root / "accepted_reload"
    trace = read(base / "trace.json")
    preview = read(base / "preview_response.json")["proposal"]
    accepted = read(base / "accept_response.json")["session"]
    map_ack = read(base / "map_render_ack.json")
    restored = read(base / "reload_response.json")["session"]
    before_conversation = read(base / "conversation_before_response.json")["conversation"]
    after_conversation = read(base / "conversation_after_response.json")["conversation"]
    reload_assertions = read(base / "session_reload_assertions.json")
    session_id = accepted["session_id"]
    base_url = read(root / MANIFEST_NAME)["base_url"]
    chat_http = read(base / "copilot_message_http_io.json")
    chat_request = json.loads(chat_http["request_body_utf8"])
    chat_events = _sse_events(chat_http["response_body_utf8"])
    _verify_copilot_stream(chat_events, expected_client_message_id=chat_request["client_message_id"])
    chat_turn = chat_events[-1]["payload"]["turn"]
    _require(
        chat_http["method"] == "POST"
        and chat_http["url"] == f"{base_url}/api/sessions/{session_id}/copilot/messages"
        and chat_http["response_status_code"] == 200
        and chat_turn["user_message"] == chat_request["message"]
        == "Review the current itinerary without changing it."
        and chat_turn["provider"] == "deterministic"
        and chat_turn in before_conversation["turns"]
        and chat_turn in after_conversation["turns"],
        "e2e1_accepted_transcript_retention_invalid",
    )
    for label, method, endpoint in (
        ("preview", "POST", "/preview"), ("accept", "POST", "/accept"),
        ("reload", "GET", ""), ("conversation_before", "GET", "/conversation"),
        ("conversation_after", "GET", "/conversation"),
    ):
        http = read(base / f"{label}_http_io.json")
        response = read(base / f"{label}_response.json")
        _require(
            http["method"] == method
            and http["url"] == f"{base_url}/api/sessions/{session_id}{endpoint}"
            and http["response_status_code"] == 200
            and json.loads(http["response_body_utf8"]) == response
            and (method != "GET" or http["request_body_utf8"] == "")
            and http["headers_persisted"] is False,
            "e2e1_accepted_reload_http_invalid",
        )
    _require(
        restored["session_id"] == session_id == reload_assertions["session_id"]
        and restored["accepted_plan_id"] == accepted["accepted_plan_id"]
        and restored["conversation_id"] == accepted["conversation_id"]
        == before_conversation["conversation_id"] == reload_assertions["conversation_id"]
        and re.fullmatch(r"conversation_[0-9a-f]{32}", str(restored["conversation_id"])) is not None
        and before_conversation == after_conversation
        and restored["selected_stop_id"] == accepted["selected_stop_id"]
        == reload_assertions["selected_stop_id"] == "pfeiffer_big_sur_state_park"
        and restored["selected_day"] == reload_assertions["selected_day"] == 5
        and reload_assertions["session_creation_count_before_reload"]
        == reload_assertions["session_creation_count_after_reload"] == 1
        and reload_assertions["assertions_passed"] is True
        and reload_assertions["console_errors"] == [],
        "e2e1_accepted_session_continuity_invalid",
    )
    for label in ("accept_rendered", "reload_rendered"):
        diagnostics = read(base / f"{label}_diagnostics.json")
        _require(diagnostics["console_errors"] == [], "e2e1_accepted_browser_errors")
    _require(
        trace["assertions_passed"] is True
        and trace["survived_reload"] is True
        and accepted["accepted_plan_id"] == preview["child_plan_id"]
        and map_ack["rendered_map_plan_id"] == accepted["accepted_plan_id"],
        "e2e1_accepted_reload_identity_mismatch",
    )
    _require(
        map_ack["rendered_geometry_hash"] == trace["geometry_hash"]
        and _map_markers(map_ack) == CHILD_MARKERS
        and accepted["draft"] == [],
        "e2e1_accepted_reload_projection_invalid",
    )


def _verify_scenarios(
    root: Path,
    read: Callable[[Path], dict[str, Any]] = _load,
) -> None:
    _verify_registered(root, read)
    _verify_poi(root, read)
    _verify_unsupported(root, read)
    _verify_ineligible(root, read)
    _verify_stale(root, read)
    _verify_accepted(root, read)
    output = (root / "pytest_output.txt").read_text(encoding="utf-8")
    _require(re.search(r"\b7 passed in \d+\.\d+s", output) is not None, "e2e1_pytest_output_invalid")
    for path in root.rglob("*.png"):
        _require(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "e2e1_screenshot_invalid")


def _verify_map_recovery_case(base: Path, base_url: str, context: str) -> None:
    """Bind one original or accepted Retry journey to its actual canonical plan."""
    trace = _load(base / "trace.json")
    initial_response = _load(base / "initial_response.json")
    active_response = _load(base / "active_session_response.json")
    session = active_response["session"]
    _require(
        trace["schema_version"] == "map-config-recovery-browser-v2" and trace["context"] == context
        and trace["assertions_passed"] is True
        and trace["normal_unmodified_reloads"] == 3
        and trace["injected_config_responses"] == 4
        and trace["maximum_attempts_per_cycle"] == 3
        and trace["new_sessions_after_initial"] == 0
        and trace["chat_messages_submitted"] == 0
        and trace["screenshots_require_stable_map_ready"] is True
        and all(trace[name] == [] for name in ("console_errors", "request_failures", "http_errors")),
        "e2e1_recovery_trace_invalid",
    )
    if context == "accepted_child":
        preview = _load(base / "preview_response.json")["session"]["proposal"]
        accepted = _load(base / "accept_response.json")
        execution = preview["runtime_execution"]
        certificate = execution["evaluation_certificate"]
        geography_source = accepted
        _require(active_response == accepted
            and trace["child_plan_id"] == preview["child_plan_id"] == execution["child_plan_id"] == certificate["plan_id"]
            and trace["child_plan_content_hash"] == preview["child_plan_content_hash"]
                == execution["child_plan_content_hash"] == certificate["plan_content_hash"]
            and trace["certificate_id"] == preview["certificate_id"] == certificate["certificate_id"]
            and execution["evaluation_executed"] is True
            and certificate["comparison_eligibility"] == preview["state"] == "eligible"
            and execution == trace["runtime_execution"], "e2e1_recovery_child_identity_invalid")
    else:
        geography_source = initial_response
        _require(context == "original" and all(trace[key] is None for key in (
            "child_plan_id", "child_plan_content_hash", "certificate_id", "runtime_execution"))
            and not (base / "preview_response.json").exists() and not (base / "accept_response.json").exists(),
            "e2e1_recovery_original_fabricated_acceptance")
    plans = [plan for plan in geography_source["workspace"]["geography"]["plans"]
        if plan["plan_id"] == session["accepted_plan_id"]]
    _require(len(plans) == 1, "e2e1_recovery_active_plan_ambiguous")
    geography = plans[0]
    child_id, child_hash = geography["plan_id"], geography["content_hash"]
    _require(
        trace["session_id"] == session["session_id"]
        and all(session[field] == initial_response["session"][field] for field in ("session_id", "trip_id", "run_id"))
        and trace["active_plan_id"] == session["accepted_plan_id"] == child_id
        and trace["active_plan_content_hash"] == child_hash
        and trace["rendered_map_plan_id"] == child_id
        and trace["rendered_geometry_hash"] == _completed_race_geometry_hash(geography)
        and (trace["child_plan_id"] == child_id if context == "accepted_child"
             else child_id == initial_response["session"]["accepted_plan_id"]),
        "e2e1_recovery_active_identity_invalid",
    )
    markers = set(trace["rendered_marker_ids"])
    _require(markers == {feature["properties"]["stop_id"] for feature in geography["stops"]["features"]}
        == (CHILD_MARKERS if context == "accepted_child" else PARENT_MARKERS), "e2e1_recovery_markers_invalid")
    for label in ("normal_reload_1", "normal_reload_2", "normal_reload_3",
                  "recovered_after_one_nonready_config", "recovered_after_explicit_retry"):
        row = _load(base / f"{label}_diagnostics.json")
        stage = row["map_stage"]
        _require(
            stage.get("renderedMapPlanId") == stage.get("activePlanId") == child_id
            and stage.get("activePlanContentHash") == child_hash
            and stage.get("renderedGeometryHash") == trace["rendered_geometry_hash"]
            and set(json.loads(stage.get("renderedMarkerIds", "[]"))) == markers,
            "e2e1_recovery_reload_identity_invalid",
        )
    blocked = _load(base / "persistent_failure_no_ready_claim_diagnostics.json")
    _require(blocked["retry_visible"] is True
             and blocked["map_stage"].get("renderedMapPlanId") is None
             and blocked["geographic_map"].get("mapReady") != "true",
             "e2e1_recovery_false_ready_claim")
    injections = json.loads((base / "fault_injections.json").read_text(encoding="utf-8"))
    _require(isinstance(injections, list) and len(injections) == 4, "e2e1_recovery_injection_count_invalid")
    _require([row["phase"] for row in injections] == ["fault_injected_once"] + ["fault_injected_persistent"] * 3,
             "e2e1_recovery_injection_sequence_invalid")
    for row in injections:
        original = json.loads(row["upstream_response_body_utf8"])
        injected = json.loads(row["injected_response_body_utf8"])
        _require(row["classification"] == "explicit_deterministic_http_fault_injection"
                 and row["method"] == "GET" and row["url"] == f"{base_url}/api/map/config"
                 and row["upstream_response_status"] == row["injected_response_status"] == 200
                 and original["status"] == "ready" and injected == {**original, "status": "failed"},
                 "e2e1_recovery_injection_scope_invalid")
    requests = json.loads((base / "browser_requests.json").read_text(encoding="utf-8"))
    creations = [row for row in requests if row["method"] == "POST" and row["url"] == f"{base_url}/api/sessions"]
    _require(len(creations) == 1 and not any("/copilot/messages" in row["url"] for row in requests),
             "e2e1_recovery_session_or_chat_request_invalid")
    _require(not any(row["method"] != "GET" and row["phase"] == "explicit_retry_after_fault_removed" for row in requests),
             "e2e1_recovery_retry_mutation")
    index = json.loads((base / "browser_http_index.json").read_text(encoding="utf-8"))
    _require(len(index) == len({item["file"] for item in index})
        and {item["file"] for item in index} == {path.name for path in base.glob("http_*.json")},
        "e2e1_recovery_http_inventory_invalid")
    observed_sessions, records = [], []
    for item in index:
        row = _load(_safe_path(base, item["file"]))
        _verify_product_http_capture(row)
        _require(row["url"].startswith(base_url + "/api/")
                 and all(row[field] == item[field] for field in ("url", "method", "phase"))
                 and row["response_status_code"] == 200
                 and row["request_body_sha256"] == hashlib.sha256(row["request_body_utf8"].encode()).hexdigest()
                 and row["response_body_sha256"] == hashlib.sha256(row["response_body_utf8"].encode()).hexdigest(),
                 "e2e1_recovery_http_binding_invalid")
        records.append(row)
        if row["method"] == "GET" and row["url"] == f"{base_url}/api/sessions/{session['session_id']}":
            restored_payload = json.loads(row["response_body_utf8"])
            restored = restored_payload["session"]
            _require(all(restored[field] == session[field] for field in (
                "session_id", "trip_id", "run_id", "accepted_plan_id", "conversation_id", "selected_stop_id", "selected_day",
            )), "e2e1_recovery_session_identity_invalid")
            restored_plans = [plan for plan in restored_payload["workspace"]["geography"]["plans"] if plan["plan_id"] == child_id]
            _require(restored_plans == [geography], "e2e1_recovery_restored_geography_mismatch")
            observed_sessions.append(row["phase"])
    created_records = [row for row in records if row["method"] == "POST" and row["url"] == base_url + "/api/sessions"]
    active_path = "/accept" if context == "accepted_child" else "/selection"
    active_records = [row for row in records if row["method"] == "POST" and row["url"].endswith(active_path)
        and json.loads(row["response_body_utf8"]) == active_response]
    _require(len(created_records) == len(active_records) == 1
        and json.loads(created_records[0]["response_body_utf8"]) == initial_response, "e2e1_recovery_active_response_unbound")
    if context == "original":
        _require(not any(row["url"].endswith(("/preview", "/accept", "/draft/operations")) for row in records),
            "e2e1_recovery_original_unexpected_mutation")
    else:
        previews = [row for row in records if row["url"].endswith("/preview")]
        _require(len(previews) == 1 and json.loads(previews[0]["response_body_utf8"])
            == _load(base / "preview_response.json"), "e2e1_recovery_preview_response_unbound")
    retry_inputs = json.loads((base / "retry_control_input.json").read_text(encoding="utf-8"))
    retry_response = _load(base / "retry_response.json")
    _verify_product_http_capture(retry_response)
    retries = [row for row in records if row["phase"] == "explicit_retry_after_fault_removed"
        and row["method"] == "GET" and row["url"] == base_url + "/api/map/config"]
    _require(len(retry_inputs) == len(retries) == 1
        and all(retry_response[key] == retries[0][key] for key in retry_response)
        and json.loads(retry_response["response_body_utf8"])["status"] == "ready",
        "e2e1_recovery_retry_response_unbound")
    retry_input = retry_inputs[0]
    ready = _load(base / "recovered_after_explicit_retry.readiness.json")
    _require(retry_input["trusted"] is True and retry_input["selector"] == "#retry-map-runtime"
        and retry_input["map_ready"] != "true" and not retry_input["map_stage"].get("renderedMapPlanId")
        and retry_input["map_stage"].get("activePlanId") == child_id
        and retry_input["map_stage"].get("activePlanContentHash") == child_hash
        and _isolation_timestamp(blocked["observed_at"]) <= _isolation_timestamp(retry_input["clicked_at"])
            <= _isolation_timestamp(retries[0]["observed_at"]) <= _isolation_timestamp(ready["screenshot_started_at"]),
        "e2e1_recovery_retry_input_unbound")
    _require(observed_sessions == ["normal_reload_1", "normal_reload_2", "normal_reload_3",
                                    "fault_injected_once", "fault_injected_persistent"],
             "e2e1_recovery_reload_sequence_invalid")
    expected_captures = {"initial_parent", "active_before_reload", "normal_reload_1", "normal_reload_2",
                         "normal_reload_3", "recovered_after_one_nonready_config", "recovered_after_explicit_retry"}
    _require({path.stem for path in base.glob("*.png")} == expected_captures, "e2e1_recovery_capture_universe_invalid")
    for name in expected_captures:
        readiness = _load(base / f"{name}.readiness.json")
        _require((base / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                 and readiness["stable_readiness_before_and_after"] is True
                 and readiness["map_ready_at"] <= readiness["screenshot_started_at"] <= readiness["screenshot_completed_at"]
                 and 0.95 <= readiness["visible_tile_area_coverage_ratio"] <= 1
                 and readiness["coverage_method"] == "loaded_tile_projected_union_area_v1",
                 "e2e1_recovery_capture_before_ready")


def _verify_map_recovery(root: Path, base_url: str) -> None:
    """Require both canonical plan contexts and the separate visible-tile fault."""
    base = root / "map_initialization_recovery"
    _require({path.name for path in base.iterdir() if path.is_dir()} == {"original", "accepted_child", "tile_coverage"},
        "e2e1_recovery_context_universe_invalid")
    for context in ("original", "accepted_child"):
        _verify_map_recovery_case(base / context, base_url, context)
    tree = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(tree.iter("testcase"))
    expected_names = {"test_active_map_reload_recovers_without_replacing_session_or_plan[" + context + "]"
        for context in ("original", "accepted_child")} | {"test_loaded_source_cannot_acknowledge_a_viewport_without_renderable_tiles"}
    _require(len(cases) == 3 and {case.get("name") for case in cases} == expected_names
        and all(case.get("classname") == "tests.product_app.browser.test_map_initialization_recovery_live" for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in tree.iter()), "e2e1_recovery_junit_universe_invalid")
    output = (root / "pytest_output.txt").read_text(encoding="utf-8")
    _require(re.search(r"\b3 passed\b", output) is not None, "e2e1_recovery_pytest_output_invalid")
    tiles = _load(base / "tile_coverage/trace.json")
    _require(tiles["assertions_passed"] is True and tiles["backend_mutations_after_load"] == 0
             and tiles["blocked_readiness"]["tileCoverageRatio"] == 0
             and tiles["blocked_readiness"]["tilesLoaded"] is False
             and tiles["before"]["rendered_map_plan_id"] == tiles["after"]["rendered_map_plan_id"]
             and tiles["before"]["rendered_geometry_hash"] == tiles["after"]["rendered_geometry_hash"]
             and tiles["before"]["rendered_marker_ids"] == tiles["after"]["rendered_marker_ids"]
             and tiles["console_errors"] == tiles["request_failures"] == [], "e2e1_visible_tile_coverage_invalid")
    for name in ("before_tile_fault", "after_tile_recovery"):
        readiness = _load(base / f"tile_coverage/{name}.readiness.json")
        _require(readiness["stable_readiness_before_and_after"] is True
                 and 0.95 <= readiness["visible_tile_area_coverage_ratio"] <= 1
                 and readiness["coverage_method"] == "loaded_tile_projected_union_area_v1"
                 and readiness["map_ready_at"] <= readiness["screenshot_started_at"] <= readiness["screenshot_completed_at"],
                 "e2e1_visible_tile_capture_invalid")


def _completed_race_geometry_hash(geography: dict[str, Any]) -> str:
    legs = json.loads(json.dumps(geography["validated_legs"]))
    for feature in legs["features"]:
        feature["properties"].pop("role", None)

    def numbers(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [numbers(item) for item in value]
        return value

    encoded = json.dumps(numbers(legs), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _verify_completed_race(root: Path, base_url: str) -> None:
    base = root / "completed_preview_race"
    trace = _load(base / "trace.json")
    a = _load(base / "preview_a_response.json")
    b = _load(base / "preview_b_response.json")
    acknowledgement = _load(base / "map_render_ack.json")
    failures = json.loads((base / "network_failures.json").read_text(encoding="utf-8"))
    _require(isinstance(failures, list), "e2e1_completed_race_network_failures_invalid")
    rendered_style = _load(base / "rendered_map_style.json")
    from verify_e2e1_network_observations import verify as verify_network
    network = verify_network(base, base_url, failures, rendered_style)
    _require(trace["schema_version"] == "completed-preview-response-race-v1"
             and trace["assertions_passed"] is True
             and trace["older_successful_response_ignored"] is True
             and trace["preview_response_statuses"] == [200, 200]
             and trace["undo_statuses"] == [409, 200]
             and trace["a_completed_at"] <= trace["b_rendered_at"] <= trace["a_released_at"]
             and trace["unexpected_console_errors"] == []
             and trace["request_failures"] == failures
             and trace["blocking_request_failure_indexes"] == []
             and network["blocking_network_failures"] == 0,
             "e2e1_completed_race_trace_invalid")
    _require(a["session"]["session_id"] == b["session"]["session_id"] == trace["session_id"],
             "e2e1_completed_race_session_mismatch")
    for label, response, candidate in (
        ("a", a, "pfeiffer_big_sur_state_park"), ("b", b, "point_lobos_state_natural_reserve"),
    ):
        proposal = response["proposal"]
        execution = proposal["runtime_execution"]
        certificate = execution["evaluation_certificate"]
        request = _load(base / f"preview_{label}_request.json")
        _require(response["session"]["proposal"] == proposal
                 and proposal["state"] == certificate["comparison_eligibility"] == "eligible"
                 and execution["evaluation_executed"] is True
                 and execution["child_plan_id"] == certificate["plan_id"] == proposal["geography_plan"]["plan_id"]
                 == proposal["child_plan_id"] == trace[f"{label}_child_plan_id"]
                 and execution["child_plan_content_hash"] == certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
                 and proposal["certificate_id"] == certificate["certificate_id"]
                 and proposal["parent_plan_id"] == trace["parent_plan_id"]
                 and execution["repair_request"]["confirmed_constraints"]["must_include"] == [candidate]
                 and execution["repair_request"]["confirmed_constraints"]["must_delete"] == [TARGET]
                 and proposal["expected_revision"] == request["expected_revision"]
                 and response["session"]["revision"] == request["expected_revision"] + 1
                 and _completed_race_geometry_hash(proposal["geography_plan"]) == trace[f"{label}_geometry_hash"],
                 "e2e1_completed_race_artifacts_invalid")
    _require(a["proposal"]["child_plan_id"] != b["proposal"]["child_plan_id"]
             and trace["a_geometry_hash"] != trace["b_geometry_hash"]
             and acknowledgement == trace["before_release"] == trace["after_release"]
             and acknowledgement["rendered_map_plan_id"] == acknowledgement["active_plan_id"] == b["proposal"]["child_plan_id"]
             and acknowledgement["active_plan_content_hash"] == b["proposal"]["child_plan_content_hash"]
             and acknowledgement["rendered_geometry_hash"] == trace["b_geometry_hash"]
             and _map_markers(acknowledgement) == (PARENT_MARKERS - {TARGET}) | {"point_lobos_state_natural_reserve"},
             "e2e1_completed_race_stale_overwrite")
    _require(trace["before_surfaces"] == trace["after_surfaces"], "e2e1_completed_race_surfaces_changed")
    surface = trace["after_surfaces"]
    _require(surface["metrics"] and all(row["plan_id"] == b["proposal"]["child_plan_id"]
                                        and row["content_hash"] == b["proposal"]["child_plan_content_hash"]
                                        for row in surface["metrics"])
             and surface["evidence"]["evidencePlanId"] == b["proposal"]["child_plan_id"]
             and surface["evidence"]["planContentHash"] == b["proposal"]["child_plan_content_hash"]
             and surface["evidence"]["evaluationId"] == b["proposal"]["certificate_id"]
             and surface["diff_id"] == b["proposal"]["diff_id"], "e2e1_completed_race_surface_identity_invalid")
    records = []
    for filename in trace["http_records"]:
        row = _load(_safe_path(base, filename))
        _verify_product_http_capture(row)
        _require(row["url"].startswith(base_url + "/api/")
                 and hashlib.sha256(row["request_body_utf8"].encode()).hexdigest() == row["request_body_sha256"]
                 and hashlib.sha256(row["response_body_utf8"].encode()).hexdigest() == row["response_body_sha256"],
                 "e2e1_completed_race_http_invalid")
        records.append(row)
    previews = [row for row in records if row["url"].endswith("/preview")]
    undos = [row for row in records if row["url"].endswith("/draft/undo")]
    _require(len(previews) == 2 and [row["response_status_code"] for row in previews] == [200, 200]
             and [json.loads(row["response_body_utf8"]) for row in previews] == [a, b]
             and [row["response_status_code"] for row in undos] == [409, 200]
             and json.loads(undos[0]["response_body_utf8"]) == {"detail": "stale_session_revision"}
             and json.loads(undos[0]["request_body_utf8"])["expected_revision"] == a["proposal"]["expected_revision"]
             and json.loads(undos[1]["request_body_utf8"])["expected_revision"] == a["session"]["revision"]
             and not any("/copilot/messages" in row["url"] for row in records),
             "e2e1_completed_race_reconciliation_invalid")
    session_reads = [row for row in records if row["method"] == "GET"
                     and row["url"] == f"{base_url}/api/sessions/{trace['session_id']}"]
    _require(len(session_reads) == 1
             and json.loads(session_reads[0]["response_body_utf8"])["session"] == a["session"],
             "e2e1_completed_race_reconciliation_snapshot_invalid")
    for name in ("screenshot_before", "screenshot_b_before_release", "screenshot_after"):
        ready = _load(base / f"{name}.readiness.json")
        _require(ready["stable_readiness_before_and_after"] is True
                 and 0.95 <= ready["visible_tile_area_coverage_ratio"] <= 1
                 and ready["map_ready_at"] <= ready["screenshot_started_at"] <= ready["screenshot_completed_at"]
                 and ready["coverage_method"] == "loaded_tile_projected_union_area_v1"
                 and (base / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
                 "e2e1_completed_race_screenshot_not_ready")
    output = (root / "pytest_output.txt").read_text(encoding="utf-8")
    _require(re.search(r"\b1 passed in \d+\.\d+s", output) is not None, "e2e1_completed_race_pytest_invalid")


def _verify_workspace_visibility(value, checkpoint, completed_at, expected_text):
    """Recompute clipping from raw DOM geometry; do not trust a reported ratio."""
    code = "e2e1_workspace_visible_controls_invalid"
    _require(value["viewport"] == checkpoint["readiness"]["viewport"]
             and datetime.fromisoformat(checkpoint["captured_at"]) <= datetime.fromisoformat(value["at"])
             <= datetime.fromisoformat(completed_at)
             and [row["selector"] for row in value["elements"]] == list(expected_text), code)
    width, height = value["viewport"]["width"], value["viewport"]["height"]

    def finite(number):
        return type(number) in (int, float) and math.isfinite(number)

    def rectangle(row):
        _require(all(finite(row[key]) for key in ("left", "top", "right", "bottom", "width", "height"))
                 and row["width"] > 0 and row["height"] > 0
                 and abs(row["right"] - row["left"] - row["width"]) < .01
                 and abs(row["bottom"] - row["top"] - row["height"]) < .01, code)

    owners = value["scroll_owners"]
    _require(owners and all(row["tag"] and isinstance(row["class_name"], str)
             and all(finite(row[key]) and row[key] >= 0 for key in
                     ("scroll_top", "scroll_left", "client_height", "scroll_height"))
             for row in owners), code)
    for row in value["elements"]:
        selector = row["selector"]
        disabled = True if selector.startswith("#inline-") else (
            False if selector in {"#copilot-submit", "#close-copilot"} else None
        )
        _require(row["present"] is True and row["text"].strip() == expected_text[selector]
                 and row["disabled"] is disabled
                 and row["display"] != "none" and row["visibility"] == "visible"
                 and finite(row["opacity"]) and row["opacity"] > 0, code)
        rectangle(row["rect"])
        rect = row["rect"]
        clip = {"left": max(0, rect["left"]), "top": max(0, rect["top"]),
                "right": min(width, rect["right"]), "bottom": min(height, rect["bottom"])}
        _require(row["ancestors"], code)
        for parent in row["ancestors"]:
            rectangle(parent["rect"])
            _require(parent["display"] != "none" and parent["visibility"] == "visible"
                     and finite(parent["opacity"]) and parent["opacity"] > 0
                     and all(finite(parent[key]) and parent[key] >= 0 for key in
                             ("client_left", "client_top", "client_width", "client_height", "scroll_top", "scroll_left")), code)
            for axis, near, far, offset, extent in (("x", "left", "right", "client_left", "client_width"),
                                                   ("y", "top", "bottom", "client_top", "client_height")):
                _require(parent["overflow_" + axis] in {"visible", "hidden", "clip", "scroll", "auto", "overlay"}, code)
                if parent["overflow_" + axis] != "visible":
                    lower = parent["rect"][near] + parent[offset]
                    clip[near], clip[far] = max(clip[near], lower), min(clip[far], lower + parent[extent])
        _require(set(row["clip"]) == set(clip) and all(finite(row["clip"][key])
                 and abs(row["clip"][key] - number) < .01 for key, number in clip.items()), code)
        ratio = max(0, clip["right"] - clip["left"]) * max(0, clip["bottom"] - clip["top"]) / (rect["width"] * rect["height"])
        _require(finite(row["visible_ratio"]) and abs(row["visible_ratio"] - ratio) < .00001
                 and .98 <= ratio <= 1.00001 and len(row["points"]) == 5, code)
        for point, (dx, dy) in zip(row["points"], ((.5, .5), (.1, .1), (.9, .1), (.1, .9), (.9, .9)), strict=True):
            _require(finite(point["x"]) and finite(point["y"])
                     and abs(point["x"] - rect["left"] - rect["width"] * dx) < .01
                     and abs(point["y"] - rect["top"] - rect["height"] * dy) < .01
                     and clip["left"] <= point["x"] <= clip["right"]
                     and clip["top"] <= point["y"] <= clip["bottom"]
                     and point["unoccluded"] is True and isinstance(point["hit"], dict)
                     and bool(point["hit"].get("tag")), code)
            if selector in {"#inline-undo", "#inline-preview", "#map-context-title", "#copilot-submit", "#close-copilot"}:
                tag = "H3" if selector == "#map-context-title" else "BUTTON"
                _require(point["hit"].get("id") == selector[1:] and point["hit"]["tag"] == tag, code)


def _verify_workspace_text_selection(selection, impact, records, created, plan, checkpoint, before_at):
    code = "e2e1_workspace_text_selection_not_completed"
    positions = []
    for captured in (selection, impact):
        _verify_product_http_capture(captured)
        matches = [index for index, row in enumerate(records)
                   if all(row.get(key) == value for key, value in captured.items())]
        _require(len(matches) == 1, code)
        positions.append(matches[0])
    left, right = positions
    _require(left < right and all(not row["url"].endswith("/selection") for row in records[left + 1:right]), code)
    request = json.loads(selection["request_body_utf8"])
    session = json.loads(selection["response_body_utf8"])["session"]
    expected_id = created["session"]["session_id"]
    previous = [json.loads(row["response_body_utf8"]).get("session") for row in records[:left]]
    revisions = [item["revision"] for item in previous if isinstance(item, dict) and item.get("session_id") == expected_id]
    _require(revisions and request["expected_revision"] == max(revisions), code)
    _require(selection["method"] == "POST" and selection["response_status_code"] == 200
             and selection["url"].endswith("/api/sessions/" + expected_id + "/selection")
             and request["selected_stop_id"] == session["selected_stop_id"] == "bixby_creek_bridge_viewpoint"
             and request["selected_day"] == session["selected_day"] == 5
             and request["selected_segment_id"] is session["selected_segment_id"] is None
             and session["revision"] == request["expected_revision"] + 1
             and all(session[key] == created["session"][key] for key in ("session_id", "trip_id", "run_id"))
             and session["accepted_plan_id"] == plan["plan_id"] and session["draft"] == [], code)
    impact_request, response = json.loads(impact["request_body_utf8"]), json.loads(impact["response_body_utf8"])
    _require(impact["method"] == "POST" and impact["response_status_code"] == 200
             and impact["url"] == selection["url"].removesuffix("/selection") + "/draft/impact-preview"
             and impact_request == {"expected_revision": session["revision"]}
             and response["schema_version"] == "product-draft-impact-preview-v1"
             and response["session_id"] == expected_id and response["session_revision"] == session["revision"]
             and response["parent_plan_id"] == plan["plan_id"] and response["parent_plan_content_hash"] == plan["content_hash"]
             and response["operations"] == [] and response["is_certified"] is False
             and response["summary"]["operation_count"] == 0
             and response["summary"]["can_run_evaluated_preview"] is False
             and checkpoint["selected_stop"] == session["selected_stop_id"] and checkpoint["selected_day"] == "5"
             and checkpoint["text_scope"]["planId"] == plan["plan_id"]
             and checkpoint["text_scope"]["planContentHash"] == plan["content_hash"]
             and datetime.fromisoformat(before_at) <= datetime.fromisoformat(records[left]["observed_at"])
             <= datetime.fromisoformat(records[right]["observed_at"])
             <= datetime.fromisoformat(checkpoint["readiness"]["screenshot_started_at"]), code)


def _verify_workspace_metric_scroll(value, checkpoint, initial, plan, completed_at):
    """Bind native footer scrolling and restored layout to the initial plan."""
    code = "e2e1_workspace_metric_scroll_invalid"
    _require(value.get("schema_version") == "workspace-metric-scroll-v1"
             and value.get("setup") == "locator_scroll_into_view_if_needed"
             and value.get("restoration") == "native_reverse_wheel_then_Control+Home"
             and value.get("map_before") == value.get("map_after") == initial, code)
    before, onscreen, end, restored = (value[key] for key in ("before", "onscreen", "end", "restored"))
    times = [checkpoint["captured_at"], before["at"], onscreen["at"], end["at"],
             value["detail_visibility"]["at"], restored["at"], value["completed_at"], completed_at]
    stamps = [datetime.fromisoformat(item) for item in times]
    _require(stamps == sorted(stamps), code)
    scope = checkpoint["metric_scope"]
    _require(before["scope"] == scope and scope["planId"] == plan["plan_id"]
             and scope["planContentHash"] == plan["content_hash"]
             and [row["label"] for row in before["metrics"]]
             == ["Route time", "Contextual risk", "Preservation", "Edit cost", "Certificate"], code)
    for row in before["metrics"]:
        _require(row["scope"] == {"planId": plan["plan_id"], "planContentHash": plan["content_hash"]}
                 and all(isinstance(row[key], str) and row[key].strip() for key in ("label", "value", "note")), code)
    dimensions = ("client_height", "client_width", "scroll_height", "scroll_width")
    for snapshot in (before, onscreen, end, restored):
        _require(snapshot["scope"] == scope and snapshot["metrics"] == before["metrics"]
                 and all(type(snapshot[key]) in (int, float) and math.isfinite(snapshot[key])
                         and snapshot[key] >= 0 for key in (*dimensions, "scroll_top", "scroll_left"))
                 and all(snapshot[key] == before[key] for key in dimensions)
                 and snapshot["client_width"] > 0 and snapshot["client_height"] > 0
                 and snapshot["scroll_width"] >= snapshot["client_width"]
                 and snapshot["scroll_height"] >= snapshot["client_height"], code)
    axes = {"x":before["scroll_width"] > before["client_width"],
            "y":before["scroll_height"] > before["client_height"]}
    _require(value["overflow_axes"] == axes
             and all(before[key] == onscreen[key] == restored[key] == 0 for key in ("scroll_top", "scroll_left"))
             and abs(end["scroll_left"] - (end["scroll_width"] - end["client_width"])) <= 1
             and abs(end["scroll_top"] - (end["scroll_height"] - end["client_height"])) <= 1
             and restored["page_scroll"] == before["page_scroll"] == {"top":0,"left":0}
             and restored["workspace_scroll"] == before["workspace_scroll"] == {"top":0,"left":0}, code)
    for axis in ("x", "y"):
        if axes[axis]:
            _require(all(snapshot["overflow_" + axis] in {"auto", "scroll"}
                         for snapshot in (before, onscreen, end, restored)), code)
    owners = [row for row in checkpoint["scroll_owners"] if "metric-strip" in row["class_name"].split()]
    _require(len(owners) == 1 and all(owners[0][key] == before[key] for key in dimensions), code)
    commands, events = value["commands"], value["events"]
    expected_commands = ["wheel_to_end", "wheel_restore", "keyboard_home"] if any(axes.values()) else ["keyboard_home"]
    _require([row["action"] for row in commands] == expected_commands
             and commands[-1]["key"] == "Control+Home"
             and all(datetime.fromisoformat(onscreen["at"]) <= datetime.fromisoformat(row["at"])
                     <= datetime.fromisoformat(restored["at"]) for row in commands)
             and all(row["trusted"] is True and row["type"] in {"wheel", "scroll", "keydown"}
                     and datetime.fromisoformat(onscreen["at"]) <= datetime.fromisoformat(row["at"])
                     <= datetime.fromisoformat(restored["at"]) for row in events), code)
    keys = [row for row in events if row["type"] == "keydown"]
    _require(len(keys) == 1 and keys[0]["key"] == "Home" and keys[0]["ctrl_key"] is True, code)
    wheels = [row for row in events if row["type"] == "wheel"]
    _require(len(wheels) == (2 if any(axes.values()) else 0), code)
    if wheels:
        for command, event, sign in zip(commands[:2], wheels, (1, -1), strict=True):
            _require(all(command["delta_" + axis] == event["delta_" + axis]
                         == (sign * before["scroll_width" if axis == "x" else "scroll_height"] if axes[axis] else 0)
                         for axis in ("x", "y")), code)
        scrolls = [row for row in events if row["type"] == "scroll"]
        _require(any(abs(row["scroll_top"] - end["scroll_top"]) <= 1
                     and abs(row["scroll_left"] - end["scroll_left"]) <= 1 for row in scrolls)
                 and any(row["scroll_top"] == row["scroll_left"] == 0 for row in scrolls), code)
    _verify_workspace_visibility(value["detail_visibility"],
        {"captured_at":end["at"],"readiness":checkpoint["readiness"]}, value["completed_at"],
        {"#metric-strip .metric:last-child small":before["metrics"][-1]["note"]})


def _verify_workspace_selected_marker(value: dict, checkpoint: dict, plan: dict, completed_at: str) -> None:
    code = "e2e1_workspace_selected_marker_not_visible"
    feature = next((row for row in plan["stops"]["features"]
                    if row["properties"]["stop_id"] == checkpoint["selected_stop"]), None)
    container = checkpoint["readiness"]["container"]
    _require(feature is not None and value.get("feature_found") is True
             and value.get("stop_id") == checkpoint["selected_stop"]
             and value.get("plan_id") == plan["plan_id"]
             and value.get("coordinates") == feature["geometry"]["coordinates"]
             and all(type(value.get(key)) in {int, float} and math.isfinite(value[key])
                     for key in ("x", "y", "width", "height"))
             and abs(value["width"] - container["width"]) <= 1
             and abs(value["height"] - container["height"]) <= 1
             and 12 <= value["x"] <= value["width"] - 12
             and 12 <= value["y"] <= value["height"] - 12
             and isinstance(value.get("rendered_stop_ids"), list)
             and set(value["rendered_stop_ids"]) == {checkpoint["selected_stop"]}
             and checkpoint["captured_at"] <= value.get("observed_at", "") <= completed_at, code)
    probes = value.get("visibility_probes")
    _require(isinstance(probes, list) and len(probes) == 5, code)
    for probe, (dx, dy) in zip(probes, ((0,0),(-10,0),(10,0),(0,-10),(0,10)), strict=True):
        _require(isinstance(probe, dict) and probe.get("unoccluded") is True
                 and all(type(probe.get(key)) in {int, float} and math.isfinite(probe[key]) for key in ("x","y"))
                 and abs(probe["x"] - value["x"] - dx) < .01
                 and abs(probe["y"] - value["y"] - dy) < .01
                 and isinstance(probe.get("hit"), dict) and probe["hit"].get("tag") == "CANVAS"
                 and probe["hit"].get("id") == ""
                 and isinstance(probe["hit"].get("class_name"), str)
                 and "maplibregl-canvas" in probe["hit"]["class_name"].split(), code)


def _verify_workspace_stability(root: Path, base_url: str) -> None:
    evidence_root = root / "workspace_stability"
    _require({path.name for path in evidence_root.iterdir() if path.is_dir()}
             == {f"{width}x{height}" for width, height in WORKSPACE_VIEWPORTS},
             "e2e1_workspace_viewport_universe_invalid")
    for width, height in WORKSPACE_VIEWPORTS:
        base = evidence_root / f"{width}x{height}"
        trace = _load(base / "trace.json")
        _require(trace["schema_version"] == "workspace-stability-browser-v1"
                 and trace["assertions_passed"] is True
                 and trace["viewport"] == {"width": width, "height": height}
                 and trace["touch_enabled"] is (width == 390)
                 and trace["keyboard_viewport_simulation"] is (width == 390)
                 and trace["os_virtual_keyboard_tested"] is False,
                 "e2e1_workspace_trace_invalid")
        _require(json.loads((base / "console_errors.json").read_text(encoding="utf-8")) == [],
                 "e2e1_workspace_console_errors")
        network = json.loads((base / "network_events.json").read_text(encoding="utf-8"))
        _require(all(row["classification"] == "browser_cancelled" and row["failure"] == "net::ERR_ABORTED"
                     for row in network), "e2e1_workspace_network_failure")
        index = json.loads((base / "browser_http_index.json").read_text(encoding="utf-8"))
        records = []
        for item in index:
            row = _load(_safe_path(base, item["file"]))
            _require(row["url"] == item["url"] and row["method"] == item["method"]
                     and row["response_status_code"] == item["status"] < 400
                     and row["url"].startswith(base_url + "/api/")
                     and "/copilot/messages" not in row["url"]
                     and row["request_headers_persisted"] is False
                     and row["response_headers_persisted"] is False
                     and hashlib.sha256(row["request_body_utf8"].encode()).hexdigest() == row["request_body_sha256"]
                     and hashlib.sha256(row["response_body_utf8"].encode()).hexdigest() == row["response_body_sha256"],
                     "e2e1_workspace_http_invalid")
            _verify_product_http_capture(row)
            records.append(row)
        creations = [row for row in records if row["method"] == "POST" and row["url"] == base_url + "/api/sessions"]
        _require(len(creations) == 1, "e2e1_workspace_session_replaced")
        created = json.loads(creations[0]["response_body_utf8"])
        plans = {plan["plan_id"]: plan for plan in created["workspace"]["geography"]["plans"]}
        parent_id = created["session"]["accepted_plan_id"]
        _require(trace["initial"]["rendered_map_plan_id"] == parent_id,
                 "e2e1_workspace_initial_plan_invalid")
        checkpoints = json.loads((base / "checkpoints.json").read_text(encoding="utf-8"))
        _require(checkpoints == trace["checkpoints"], "e2e1_workspace_checkpoint_projection_invalid")
        expected_labels = ["initial_trip", "after_reload", "after_reload_2", "after_reload_3", "text_route_day5", "compare_recommendation",
                           "after_module_switching", "copilot_open", "copilot_closed"]
        if width == 390:
            expected_labels.insert(7, "reduced_keyboard_viewport")
        else:
            inserted = ["copilot_width_560", "copilot_width_320", "copilot_width_440"]
            if width == 1440:
                inserted += ["copilot_resized_1024", "copilot_resized_1440"]
            expected_labels[expected_labels.index("copilot_closed"):expected_labels.index("copilot_closed")] = inserted
        if width == 1440:
            expected_labels += ["resized_1024", "resized_1440"]
        _require([row["label"] for row in checkpoints] == expected_labels,
                 "e2e1_workspace_checkpoint_universe_invalid")
        compare_option = None
        for checkpoint in checkpoints:
            label = checkpoint["label"]
            ready = _load(base / f"{label}.readiness.json")
            expected_size = {"width": width, "height": height}
            if label == "reduced_keyboard_viewport":
                expected_size["height"] = 480
            if label.startswith("resized_"):
                expected_size["width"] = int(label.split("_")[1])
            if label.startswith("copilot_resized_"):
                expected_size["width"] = int(label.rsplit("_", 1)[1])
            _require(ready == checkpoint["readiness"]
                     and ready["viewport"] == expected_size
                     and ready["stable_readiness_before_and_after"] is True
                     and ready["coverage_method"] == "loaded_tile_projected_union_area_v1"
                     and 0.95 <= ready["visible_tile_area_coverage_ratio"] <= 1
                     and datetime.fromisoformat(ready["map_ready_at"]) <= datetime.fromisoformat(ready["screenshot_started_at"])
                     <= datetime.fromisoformat(ready["screenshot_completed_at"])
                     <= datetime.fromisoformat(checkpoint["captured_at"]) <= datetime.fromisoformat(trace["completed_at"])
                     and checkpoint["document_width"] <= checkpoint["viewport_width"] + 1
                     and (base / f"{label}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
                     "e2e1_workspace_capture_not_ready")
            visible = ready["visible_maps"]
            _require(visible and {item["pane"] for item in visible}
                     == (({"option"} if width == 390 else {"baseline", "option"})
                         if label == "compare_recommendation" else {"trip"}),
                     "e2e1_workspace_visible_map_universe_invalid")
            for item in visible:
                plan = plans.get(item["plan_id"])
                _require(plan is not None and item["plan_content_hash"] == plan["content_hash"]
                         and 0.95 <= item["visible_tile_area_coverage_ratio"] <= 1
                         and item["map_ready_at"] <= ready["screenshot_started_at"]
                         and item["container"]["width"] > 0 and item["container"]["height"] > 0
                         and item["geometry_hash"] == _completed_race_geometry_hash(plan)
                         and set(item["marker_ids"]) == {row["properties"]["stop_id"] for row in plan["stops"]["features"]},
                         "e2e1_workspace_visible_map_identity_invalid")
                if item["pane"] == "option":
                    compare_option = item
                    _require(item["plan_id"] != parent_id, "e2e1_workspace_compare_not_selected")
                else:
                    _require(item["plan_id"] == parent_id, "e2e1_workspace_parent_not_preserved")
            if label == "initial_trip" and width != 390:
                _require(500 <= ready["container"]["height"] <= 760, "e2e1_workspace_map_height_unbounded")
            if label == "text_route_day5":
                _require(checkpoint["text_scope"]["planId"] == parent_id
                         and checkpoint["text_scope"]["planContentHash"] == plans[parent_id]["content_hash"],
                         "e2e1_workspace_text_route_identity_invalid")
            if width != 390 and label.startswith("copilot_"):
                _verify_workspace_selected_marker(_load(base / f"{label}.selected_marker.json"),
                                                  checkpoint, plans[parent_id], trace["completed_at"])
                dock = checkpoint["dock"]
                _require(dock["composer_value"] == "Unsaved UI test text; do not send.",
                         "e2e1_workspace_composer_not_preserved")
                if label != "copilot_closed":
                    expected_width = int(label.removeprefix("copilot_width_")) if label.startswith("copilot_width_") else 440
                    _require(dock["open"] is True and abs(dock["width"] - expected_width) <= 1
                             and abs(dock["left"] + dock["width"] - expected_size["width"]) <= 1
                             and abs(dock["workspace_right"] - dock["left"]) <= 1
                             and dock["repair_panel_display"] == "none",
                             "e2e1_workspace_dock_overlaps_map")
                else:
                    _require(dock["open"] is False and dock["repair_panel_display"] != "none",
                             "e2e1_workspace_decision_panel_not_restored")
            if label in {"initial_trip", "text_route_day5", "after_module_switching"} or (
                width != 390 and label.startswith("copilot_")
            ):
                expected_text = {"#inline-undo": "Undo latest draft", "#inline-preview": "Preview repair"}
                if label.startswith("copilot_"):
                    selected_stop = next(row["properties"] for row in plans[parent_id]["stops"]["features"]
                                         if row["properties"]["stop_id"] == "bixby_creek_bridge_viewpoint")
                    expected_text["#map-context-title"] = selected_stop.get("name") or selected_stop["stop_id"]
                    if label != "copilot_closed":
                        expected_text.update({"#copilot-submit":"→", "#close-copilot":"×"})
                if label == "text_route_day5":
                    stops = sorted((row["properties"] for row in plans[parent_id]["stops"]["features"]
                                    if row["properties"]["day"] == 5), key=lambda row:
                                        row["stop_order"])
                    _require(stops, "e2e1_workspace_day_five_stops_missing")
                    expected_text.update({".textual-map-alternative > summary": "Text route description",
                        ".text-route-summary > h3:first-child": "Day 5 route",
                        ".text-route-summary dd:first-of-type": " → ".join(
                            f"{index}. {stop.get('name') or stop['stop_id']}" for index, stop in enumerate(stops, 1))})
                    before = _load(base / "text_route_before_selection.visibility.json")
                    description = _load(base / "text_route_description.visibility.json")
                    _require(before["viewport"] == expected_size and [row["selector"] for row in before["elements"]]
                             == [".textual-map-alternative > summary", ".text-route-summary > h3:first-child"]
                             and all(row.get("present") is True and row.get("text", "").strip() for row in before["elements"])
                             and description["viewport"] == expected_size
                             and [row["selector"] for row in description["elements"]] == [".textual-map-alternative > p"]
                             and description["elements"][0]["present"] is True
                             and bool(description["elements"][0]["text"].strip()),
                             "e2e1_workspace_text_observations_missing")
                    _verify_workspace_text_selection(_load(base / "text_route_selection_http.json"),
                        _load(base / "text_route_impact_http.json"), records, created, plans[parent_id], checkpoint, before["at"])
                _verify_workspace_visibility(_load(base / f"{label}.visibility.json"), checkpoint, trace["completed_at"], expected_text)
        if width != 390:
            commands = json.loads((base / "dock_resize_commands.json").read_text(encoding="utf-8"))
            expected_commands = [
                {"action":"keyboard_resize", "key":key, "count":count, "target_width":target}
                for target, key, count in ((560,"ArrowLeft",6),(320,"ArrowRight",12),(440,"ArrowLeft",6))
            ]
            if width == 1440:
                expected_commands += [{"action":"window_resize","width":size,"height":height} for size in (1024,1440)]
            dock_checkpoints = {row["label"]:row for row in checkpoints}
            _require(len(commands) == len(expected_commands), "e2e1_workspace_resize_commands_invalid")
            for command, expected in zip(commands, expected_commands, strict=True):
                label = (f"copilot_width_{expected['target_width']}" if expected["action"] == "keyboard_resize"
                         else f"copilot_resized_{expected['width']}")
                _require({key:value for key,value in command.items() if key != "at"} == expected
                         and dock_checkpoints["copilot_open"]["captured_at"] <= command["at"] <= dock_checkpoints[label]["captured_at"],
                         "e2e1_workspace_resize_commands_invalid")
            small = dock_checkpoints["copilot_width_560"]["readiness"]["container"]["width"]
            large = dock_checkpoints["copilot_width_320"]["readiness"]["container"]["width"]
            _require(abs(large - small - 240) <= 2, "e2e1_workspace_dock_resize_not_connected")
        _verify_workspace_metric_scroll(_load(base / "metric_footer_scroll.json"), checkpoints[0],
                                        trace["initial"], plans[parent_id], trace["completed_at"])
        selected_evidence = _load(base / "evidence_scope.json")
        _require(compare_option is not None and selected_evidence["plan_id"] == compare_option["plan_id"]
                 and selected_evidence["content_hash"] == compare_option["plan_content_hash"]
                 and selected_evidence["plan_id"] in selected_evidence["visible_text"],
                 "e2e1_workspace_compare_evidence_identity_invalid")
    output = (root / "pytest_output.txt").read_text(encoding="utf-8")
    _require(re.search(r"\b4 passed in \d+\.\d+s", output) is not None, "e2e1_workspace_pytest_invalid")


def _verify_runtime_environments(root: Path, manifest: dict[str, Any]) -> None:
    from e2e1_runtime_environment import validate_runtime_environment

    receipt = manifest["provider_isolation"]["record"]
    nodes = {case.get("classname").replace(".", "/") + ".py::" + case.get("name")
             for case in ET.parse(root / "pytest_junit.xml").getroot().iter("testcase")}
    observed, referenced = set(), set()
    screenshots = list(root.rglob("*.readiness.json"))
    _require(screenshots and nodes, "e2e1_runtime_environment_missing")
    for path in screenshots:
        ready = _load(path)
        binding = ready.get("runtime_environment") or {}
        node = binding.get("test_nodeid", "")
        expected_name = "runtime_environment_" + hashlib.sha256(node.encode()).hexdigest()[:16] + ".json"
        _require(node in nodes and binding.get("artifact") == expected_name, "e2e1_runtime_journey_binding_invalid")
        actual = _safe_path(path.parent, expected_name)
        record = _load(actual)
        validate_runtime_environment(record, run_id=receipt["run_id"], root=ROOT)
        _require(binding.get("sha256") == _sha256(actual) and binding.get("browser") == record["browser"]
                 and record["application_origin"] == manifest["base_url"]
                 and datetime.fromisoformat(receipt["installed_at"]) <= datetime.fromisoformat(record["observed_at"])
                 <= datetime.fromisoformat(record["completed_at"]) <= datetime.fromisoformat(ready["screenshot_started_at"])
                 <= datetime.fromisoformat(ready["screenshot_completed_at"]) <= datetime.fromisoformat(receipt["finalized_at"]),
                 "e2e1_runtime_receipt_binding_invalid")
        observed.add(node)
        referenced.add(actual)
    _require(observed == nodes and referenced == set(root.rglob("runtime_environment_*.json")),
             "e2e1_runtime_journey_universe_invalid")


def _artifact_content_hash(value):
    return hashlib.sha256(json.dumps({key: child for key, child in value.items() if key != "content_hash"},
        sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def _new_journey_http(base, base_url):
    index = json.loads((base / "http_index.json").read_text(encoding="utf-8"))
    _require(len(index) == len({row["file"] for row in index})
             and {row["file"] for row in index} == {path.name for path in base.glob("http_*.json") if path.name != "http_index.json"},
             "e2e1_journey_http_inventory_invalid")
    records = []
    for item in index:
        row = _load(_safe_path(base, item["file"]))
        _verify_product_http_capture(row)
        _require(row["url"] == item["url"] and row["url"].startswith(base_url + "/api/")
                 and row["response_status_code"] == item["status"] == 200 and "/copilot/messages" not in row["url"],
                 "e2e1_journey_http_invalid")
        records.append(row)
    return records


def _verify_plan_ack(ack, proposal):
    geography = proposal["geography_plan"]
    _require(ack["active_plan_id"] == ack["rendered_map_plan_id"] == geography["plan_id"] == proposal["child_plan_id"]
             and ack["active_plan_content_hash"] == geography["content_hash"] == proposal["child_plan_content_hash"]
             and ack["rendered_geometry_hash"] == _completed_race_geometry_hash(geography)
             and set(ack["rendered_marker_ids"]) == {row["properties"]["stop_id"] for row in geography["stops"]["features"]},
             "e2e1_journey_render_identity_invalid")


def _verify_journey_captures(base, labels, plan_ids):
    _require({path.stem.removesuffix(".readiness") for path in base.glob("*.readiness.json")} == set(labels),
             "e2e1_journey_capture_universe_invalid")
    for label in labels:
        ready = _load(base / f"{label}.readiness.json")
        _require(ready["stable_readiness_before_and_after"] is True
                 and ready["coverage_method"] == "loaded_tile_projected_union_area_v1"
                 and 0.95 <= ready["visible_tile_area_coverage_ratio"] <= 1
                 and ready["visible_maps"] and all(item["plan_id"] in plan_ids
                    and 0.95 <= item["visible_tile_area_coverage_ratio"] <= 1
                    and datetime.fromisoformat(item["map_ready_at"]) <= datetime.fromisoformat(ready["screenshot_started_at"])
                    for item in ready["visible_maps"])
                 and datetime.fromisoformat(ready["map_ready_at"]) <= datetime.fromisoformat(ready["screenshot_started_at"])
                 <= datetime.fromisoformat(ready["screenshot_completed_at"])
                 and (base / f"{label}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
                 "e2e1_journey_capture_not_ready")


def _verify_direct_edits(root, base_url):
    cases = {
        "combined": ("bixby_creek_bridge_viewpoint", {"candidate_id":"pfeiffer_big_sur_state_park"}),
        "duration": ("griffith_observatory", {"duration":{"mode":"exact","preferred_minutes":60,"minimum_minutes":60,"maximum_minutes":60}}),
        "time_window": ("stearns_wharf", {"earliest_arrival":"10:00","latest_departure":None}),
        "role": ("stearns_wharf", {"role":"meal"}),
        "order": ("kings_canyon_national_park", {"day":4,"sequence_index":1}),
    }
    _require({path.name for path in (root / "direct_edits").iterdir() if path.is_dir()} == set(cases),
             "e2e1_direct_edit_universe_invalid")
    previous_decision = None
    for kind, (target, parameters) in cases.items():
        base = root / "direct_edits" / kind
        trace = _load(base / "trace.json")
        payload = _load(base / "preview_response.json")
        proposal = payload["proposal"]
        operation_input = json.loads((base / "operation_input.json").read_text(encoding="utf-8"))
        combined = kind == "combined"
        operation = operation_input[0] if combined else operation_input
        _require(trace["schema_version"] == "direct-edit-browser-v1" and trace["kind"] == kind
                 and trace["assertions_passed"] is True and operation_input == trace["operation_input"]
                 and operation["type"] == ("replace_nearby" if combined else "set_stop_" + kind) and operation["target"] == target
                 and operation["parameters"] == parameters and proposal == payload["session"]["proposal"]
                 and proposal["state"] == "eligible" and proposal["execution_mode"] == ("atomic_combined_independent_evaluation" if combined else "typed_direct_edit_independent_evaluation")
                 and proposal["child_plan_id"] == trace["child_plan_id"]
                 and proposal["child_plan_content_hash"] == trace["child_plan_content_hash"], "e2e1_direct_edit_identity_invalid")
        execution = proposal.get("runtime_execution", {})
        certificate = execution.get("evaluation_certificate") or proposal.get("evaluation_certificate") or proposal.get("certificate_schedule_evidence", {}).get("certificate_record") or proposal.get("certificate_role_evidence", {}).get("certificate_record")
        diff = proposal.get("plan_diff") or execution.get("plan_diff")
        _require(certificate["comparison_eligibility"] == "eligible"
                 and certificate["plan_id"] == diff["child_plan_id"] == proposal["child_plan_id"]
                 and certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
                 and certificate["certificate_id"] == proposal["certificate_id"] == trace["evaluation_id"]
                 and certificate["content_hash"] == _artifact_content_hash(certificate) == proposal["certificate_content_hash"]
                 and diff["parent_plan_id"] == proposal["parent_plan_id"] == trace["parent"]["active_plan_id"]
                 and proposal["parent_plan_content_hash"] == trace["parent"]["active_plan_content_hash"]
                 and _artifact_content_hash(diff) == proposal["diff_content_hash"], "e2e1_direct_edit_evaluation_invalid")
        _verify_plan_ack(trace["rendered"], proposal)
        _require(trace["reloaded"] == trace["rendered"]
                 and all(trace["staged"][field] == trace["parent"][field] for field in (
                     "active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids")),
                 "e2e1_direct_edit_stage_or_reload_invalid")
        records = _new_journey_http(base, base_url)
        previews = [row for row in records if row["url"].endswith("/preview")]
        mutations = [row for row in records if row["url"].endswith("/draft/operations")]
        _require(json.loads(previews[-1]["response_body_utf8"]) == payload
                 and (list(map(lambda row: json.loads(row["request_body_utf8"]), mutations)) == operation_input if combined else json.loads(mutations[-1]["request_body_utf8"]) == operation)
                 and len([row for row in mutations if json.loads(row["request_body_utf8"])["type"].startswith("set_stop_")]) == (4 if combined else 1),
                 "e2e1_direct_edit_http_link_invalid")
        accepted = _load(base / "acceptance_response.json")
        decisions = [json.loads(row["response_body_utf8"]) for row in records if row["url"].endswith("/accept")]
        _require(decisions[-1] == accepted and len(decisions) == 1,
                 "e2e1_direct_edit_acceptance_http_invalid")
        decision = accepted["result"]["decision"]
        pointer = accepted["result"]["pointer"]
        _require(decision["content_hash"] == _canonical_sha256({key:value for key,value in decision.items() if key != "content_hash"})
                 and pointer["content_hash"] == _canonical_sha256({key:value for key,value in pointer.items() if key != "content_hash"})
                 and decision["decision_type"] == "accept_repair"
                 and accepted["session"]["accepted_plan_id"] == decision["selected_plan_id"] == proposal["child_plan_id"]
                 and decision["selected_plan_content_hash"] == proposal["child_plan_content_hash"]
                 and decision["previous_plan_id"] == proposal["parent_plan_id"]
                 and decision["previous_plan_content_hash"] == proposal["parent_plan_content_hash"]
                 and decision["certificate_id"] == proposal["certificate_id"]
                 and decision["certificate_content_hash"] == proposal["certificate_content_hash"]
                 and decision["diff_id"] == proposal["diff_id"] and decision["diff_content_hash"] == proposal["diff_content_hash"]
                 and decision["draft_content_hash"] == proposal["draft_content_hash"]
                 and pointer["decision_id"] == decision["decision_id"]
                 and pointer["accepted_plan_id"] == proposal["child_plan_id"]
                 and pointer["accepted_plan_content_hash"] == proposal["child_plan_content_hash"]
                 and pointer["revision"] == decision["pointer_revision_after"] == decision["pointer_revision_before"] + 1
                 and accepted["session"]["proposal"] is None and accepted["session"]["draft"] == [],
                 "e2e1_direct_edit_acceptance_binding_invalid")
        if previous_decision is not None:
            _require(previous_decision["selected_plan_id"] == proposal["parent_plan_id"]
                     and previous_decision["selected_plan_content_hash"] == proposal["parent_plan_content_hash"]
                     and previous_decision["pointer_revision_after"] == decision["pointer_revision_before"],
                     "e2e1_direct_edit_accepted_chain_invalid")
        if combined:
            child = execution["child_plan"]
            stops = {stop["stop_id"]: stop for stop in child["selected_stops"]}
            _require(execution["schema_version"] == "atomic-combined-repair-execution-v1"
                     and execution["evaluation_executed"] is True and execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
                     and child["content_hash"] == _artifact_content_hash(child) == proposal["child_plan_content_hash"]
                     and execution["request_id"] == child["planning_request_id"] == proposal["compiled_request"]["source_request_id"]
                     and len(execution["operations"]) == len(operation_input) == 5
                     and execution["optimizer_execution"]["planner_run"]["method_executed"] == "deterministic_context_aware_heuristic"
                     and diff["schema_version"] == "plan-diff-v6"
                     and all(diff[key] for key in ("duration_changes", "time_window_changes", "role_changes", "reorder_changes"))
                     and "bixby_creek_bridge_viewpoint" not in stops and "pfeiffer_big_sur_state_park" in stops
                     and stops["griffith_observatory"]["visit_duration_minutes"] == 90
                     and stops["golden_gate_bridge"]["time_window_constraint"]["earliest_arrival"] == "10:00"
                     and stops["golden_gate_bridge"]["itinerary_role"] == "scenic_stop"
                     and child["ordered_days"][3]["stop_ids"] == ["kings_canyon_national_park", "sequoia_national_park"]
                     and trace["parent"]["rendered_geometry_hash"] != trace["rendered"]["rendered_geometry_hash"],
                     "e2e1_combined_semantics_invalid")
        previous_decision = decision
        _verify_plan_ack(trace["accepted"], proposal)
        _require(trace["accepted"]["preview_state"] == "PARENT_RENDERED"
                 and trace["rendered"]["preview_state"] == "PREVIEW_RENDERED"
                 and {key:value for key,value in trace["accepted"].items() if key != "preview_state"}
                 == {key:value for key,value in trace["rendered"].items() if key != "preview_state"},
                 "e2e1_direct_edit_acceptance_state_invalid")
        restored = [json.loads(row["response_body_utf8"]) for row in records if row["method"] == "GET"
                    and row["url"].endswith("/api/sessions/" + accepted["session"]["session_id"])][-1]
        accepted_scope = _load(base / "accepted_evidence_panel.json")["scope"]
        _require(restored["session"]["accepted_plan_id"] == proposal["child_plan_id"]
                 and restored["session"]["proposal"] is None and restored["session"]["draft"] == []
                 and restored["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"]
                 and accepted_scope["evidenceScope"] == "accepted"
                 and accepted_scope["evidencePlanId"] == proposal["child_plan_id"]
                 and accepted_scope["planContentHash"] == proposal["child_plan_content_hash"]
                 and accepted_scope["evaluationId"] == proposal["certificate_id"],
                 "e2e1_direct_edit_accepted_reload_invalid")
        evidence = _load(base / "evidence_panel.json")
        _require(evidence["scope"]["evidencePlanId"] == proposal["child_plan_id"]
                 and evidence["scope"]["planContentHash"] == proposal["child_plan_content_hash"]
                 and proposal["certificate_id"] in evidence["text"], "e2e1_direct_edit_evidence_scope_invalid")
        _require(json.loads((base / "console_errors.json").read_text(encoding="utf-8")) == []
                 and json.loads((base / "network_failures.json").read_text(encoding="utf-8")) == [], "e2e1_direct_edit_browser_error")
        _verify_journey_captures(base, ("screenshot_before", "screenshot_after", "compare", "after_reload", "accepted_after_reload"),
            {proposal["parent_plan_id"], proposal["child_plan_id"]})
    _require(re.search(r"\b5 passed in \d+\.\d+s", (root / "pytest_output.txt").read_text(encoding="utf-8")) is not None,
             "e2e1_direct_edit_pytest_invalid")


def _verify_addition_continuation(root, base_url):
    base = root / "addition_continuation"
    assertions = _load(base / "browser_assertions.json")
    trace = _load(base / "journey_trace.json")
    _require(assertions["schema_version"] == "addition-continuation-browser-v1"
             and assertions["assertions_passed"] is True and assertions["mocked_api_responses"] is False
             and assertions["direct_api_setup_calls"] == assertions["copilot_submissions"] == 0
             and assertions["session_creation_count"] == 1 and assertions["acceptance_count"] == 2
             and all(assertions[name] == trace[name] == [] for name in (
                 "console_errors", "request_failures", "http_errors", "http_capture_errors")), "e2e1_addition_assertions_invalid")
    records = _new_journey_http(base, base_url)
    from verify_e2e1_control_actions import decision_controls
    decision_controls(json.loads((base/'decision_control_journeys.json').read_text(encoding='utf-8')),
        json.loads((base/'decision_control_events.json').read_text(encoding='utf-8')),records,assertions)
    previews = [row for row in records if row["url"].endswith("/preview")]
    accepts = [row for row in records if row["url"].endswith("/accept")]
    mutations = [row for row in records if row["url"].endswith("/draft/operations")]
    _require(len(previews) == len(accepts) == len(mutations) == 2
             and len([row for row in records if row["url"] == base_url + "/api/sessions" and row["method"] == "POST"]) == 1,
             "e2e1_addition_http_universe_invalid")
    decisions = json.loads((base / "acceptance_decisions.json").read_text(encoding="utf-8"))
    _require(decisions == assertions["decisions"] == trace["decisions"] and len(decisions) == 2,
             "e2e1_addition_decision_projection_invalid")
    parent = assertions["a"]
    plan_ids = {parent["active_plan_id"]}
    for index, label in enumerate(("b", "c")):
        saved = _load(base / f"{label}_canonical_artifacts.json")
        payload = json.loads(previews[index]["response_body_utf8"])
        proposal = saved["proposal"]
        execution = proposal["runtime_execution"]
        child, certificate, diff = execution["child_plan"], execution["evaluation_certificate"], execution["plan_diff"]
        _require(proposal == payload["proposal"] == payload["session"]["proposal"]
                 and proposal["state"] == "eligible" and execution["evaluation_executed"] is True
                 and execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
                 and child["plan_id"] == certificate["plan_id"] == diff["child_plan_id"] == proposal["child_plan_id"]
                 and child["content_hash"] == _artifact_content_hash(child) == certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
                 and certificate["content_hash"] == _artifact_content_hash(certificate) == proposal["certificate_content_hash"]
                 and certificate["comparison_eligibility"] == "eligible"
                 and _artifact_content_hash(diff) == proposal["diff_content_hash"]
                 and child["parent_plan_id"] == diff["parent_plan_id"] == proposal["parent_plan_id"] == parent["active_plan_id"]
                 and proposal["parent_plan_content_hash"] == parent["active_plan_content_hash"]
                 and child["planning_request_id"] == execution["request_id"] == proposal["compiled_request"]["source_request_id"],
                 "e2e1_addition_canonical_identity_invalid")
        ack = assertions[label]
        _verify_plan_ack(ack, proposal)
        _require(ack == saved["map_ack"] and ack["rendered_geometry_hash"] != parent["rendered_geometry_hash"],
                 "e2e1_addition_geometry_not_updated")
        expected = ({"type":"replace_nearby","target":TARGET,"parameters":{"candidate_id":CANDIDATE}}
                    if index == 0 else {"type":"add_candidate","target":"point_lobos_state_natural_reserve","parameters":{"day":5}})
        operation = json.loads(mutations[index]["request_body_utf8"])
        _require(all(operation[name] == value for name, value in expected.items())
                 and {row["stop_id"] for row in diff["added_stops"]} == {CANDIDATE if index == 0 else "point_lobos_state_natural_reserve"}
                 and {row["stop_id"] for row in diff["deleted_stops"]} == ({TARGET} if index == 0 else set()),
                 "e2e1_addition_semantics_invalid")
        accepted = json.loads(accepts[index]["response_body_utf8"])
        decision = decisions[index]
        _require(json.loads(accepts[index]["request_body_utf8"])["confirmation"] == "accept_evaluated_repair"
                 and accepted["result"]["decision"] == decision
                 and accepted["session"]["accepted_plan_id"] == decision["selected_plan_id"] == proposal["child_plan_id"]
                 and accepted["session"]["draft"] == [] and accepted["session"]["proposal"] is None
                 and decision["previous_plan_id"] == parent["active_plan_id"]
                 and decision["selected_plan_content_hash"] == proposal["child_plan_content_hash"]
                 and decision["certificate_id"] == proposal["certificate_id"]
                 and decision["diff_id"] == proposal["diff_id"]
                 and decision["content_hash"] == hashlib.sha256(json.dumps({key:value for key,value in decision.items() if key != "content_hash"},
                     sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest(), "e2e1_addition_acceptance_invalid")
        for suffix in ("preview", "after_evidence", "accepted", "after_reload"):
            recorded = trace["checkpoints"][f"{label}_{suffix}"]
            _require(all(recorded[field] == ack[field] for field in (
                "active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids")),
                "e2e1_addition_reload_identity_invalid")
        metrics = _load(base / f"{label}_metrics_dom.json")["attributes"]
        evidence = _load(base / f"{label}_evidence_dom.json")
        _require(metrics["planId"] == evidence["attributes"]["evidencePlanId"] == proposal["child_plan_id"]
                 and metrics["planContentHash"] == evidence["attributes"]["planContentHash"] == proposal["child_plan_content_hash"]
                 and metrics["evaluationId"] == evidence["attributes"]["evaluationId"] == proposal["certificate_id"]
                 and proposal["diff_id"] in evidence["visible_text"], "e2e1_addition_visible_scope_invalid")
        parent = ack
        plan_ids.add(ack["active_plan_id"])
    _require(decisions[1]["previous_plan_id"] == decisions[0]["selected_plan_id"]
             and decisions[1]["pointer_revision_before"] == decisions[0]["pointer_revision_after"]
             and set(assertions["c"]["rendered_marker_ids"]) == set(assertions["b"]["rendered_marker_ids"]) | {"point_lobos_state_natural_reserve"},
             "e2e1_addition_lineage_invalid")
    labels = ["a_initial", "b_staged_parent_unchanged", "c_staged_parent_unchanged", "c_final_after_evidence"]
    labels += [f"{label}_{suffix}" for label in ("b", "c") for suffix in ("preview", "compare", "after_evidence", "accepted", "after_reload")]
    labels += [f'keep_original_{context}_{suffix}' for context in ('accepted_child','original') for suffix in ('rendered','reloaded')]
    _verify_journey_captures(base, labels, plan_ids)
    _require(re.search(r"\b1 passed in \d+\.\d+s", (root / "pytest_output.txt").read_text(encoding="utf-8")) is not None,
             "e2e1_addition_pytest_invalid")


def _verify_mobile_keyboard(observed, prompts):
    _require(observed.get("schema_version") == "mobile-keyboard-browser-v1"
             and observed.get("os_virtual_keyboard_tested") is False
             and observed.get("message_request_count") == len(prompts), "e2e1_mobile_keyboard_scope_invalid")
    _require(observed.get("focus") == {"opened":"copilot-input","tab":"copilot-submit",
             "shift_tab":"copilot-input","escape":"open-copilot"}, "e2e1_mobile_keyboard_focus_invalid")
    newline = observed.get("newline",{})
    _require(newline.get("value") == "\n" and newline.get("request_count") == 0
             and any(event.get("key") == "Enter" and event.get("shift") is True
                     and event.get("trusted") is True and event.get("target") == "copilot-input"
                     for event in newline.get("events",[])), "e2e1_mobile_keyboard_newline_invalid")
    typing = observed.get("typing",[])
    _require(len(typing) == len(prompts), "e2e1_mobile_keyboard_typing_invalid")
    for number, (row, prompt) in enumerate(zip(typing,prompts,strict=True)):
        events = row.get("events",[])
        _require(row.get("input") == row.get("value_before_submit") == prompt
                 and row.get("focused_before_submit") == "copilot-input"
                 and "".join(event.get("key","") for event in events) == prompt
                 and all(event.get("trusted") is True and event.get("target") == "copilot-input"
                         and event.get("shift") is False and event.get("value") == prompt[:index]
                         for index, event in enumerate(events)), "e2e1_mobile_keyboard_typing_invalid")
        submitted = row.get("events_through_submit",[])
        expected = [*events,{"key":"Enter","shift":False,"trusted":True,"target":"copilot-input","value":prompt}]
        _require(row.get("submit_method") == ("enter" if number == len(prompts)-1 else "touch")
                 and submitted == (expected if number == len(prompts)-1 else events)
                 and row.get("requests_before") == number and row.get("requests_after") == number+1,
                 "e2e1_mobile_keyboard_submission_invalid")
    viewport = observed.get("reduced_viewport",{})
    controls = viewport.get("controls",[])
    _require(viewport.get("width") == 390 and viewport.get("height") == 480
             and [box.get("selector") for box in controls] == ["#copilot-input","#copilot-submit","#copilot-form"],
             "e2e1_mobile_keyboard_viewport_invalid")
    for box in controls:
        _require(all(type(box.get(key)) in {int,float} for key in ("x","y","width","height"))
                 and box["x"] >= 0 and box["y"] >= 0 and box["width"] > 0 and box["height"] > 0
                 and box["x"] + box["width"] <= 391 and box["y"] + box["height"] <= 481
                 and box.get("center_unoccluded") is True, "e2e1_mobile_keyboard_controls_obscured")
    scroll = observed.get("conversation_scroll",{})
    before, after = scroll.get("before",{}), scroll.get("after",{})
    _require(scroll.get("input_method") == "browser_wheel"
             and all(type(row.get(key)) in {int,float}
                     for row in (before,after) for key in ("top","height","content_height","page_x","page_y"))
             and before["content_height"] == after["content_height"] > before["height"] == after["height"] > 0
             and 0 <= after["top"] < before["top"] <= before["content_height"]-before["height"]+1
             and (before["page_x"],before["page_y"]) == (after["page_x"],after["page_y"]),
             "e2e1_mobile_keyboard_scroll_invalid")
    boundary = observed.get("boundary_scroll",{})
    before, after = boundary.get("before",{}), boundary.get("after",{})
    _require(all(type(before.get(key)) in {int,float} for key in ("top","height","content_height","page_x","page_y"))
             and before["content_height"] > before["height"] > 0
             and abs(before["top"] - (before["content_height"]-before["height"])) <= 1
             and before == after, "e2e1_mobile_keyboard_scroll_chaining")


def _verify_mobile_chat(root, base_url):
    base = root / "mobile_chat"
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    _require(len(cases) == 1 and cases[0].get("classname") == "tests.product_app.browser.test_mobile_chat_journey_live"
             and cases[0].get("name") == "test_mobile_chat_preserves_semantics_and_accepts_exact_rendered_child"
             and not any(node.tag in {"failure","error","skipped"} for node in junit.iter()), "e2e1_mobile_test_inventory_invalid")
    trace = _load(base / "trace.json")
    payload = _load(base / "preview_response.json")
    proposal = payload["proposal"]
    _require(trace["schema_version"] == "mobile-chat-browser-v1" and trace["assertions_passed"] is True
             and trace["touch_enabled"] is True and trace["os_virtual_keyboard_tested"] is False
             and trace["chat_turn_count"] == 3 and proposal["state"] == "eligible", "e2e1_mobile_trace_invalid")
    index = json.loads((base / "http_index.json").read_text(encoding="utf-8"))
    _require(isinstance(index,list), "e2e1_mobile_http_inventory_invalid")
    _require(len(index) == len({row["file"] for row in index})
             and {row["file"] for row in index} == {path.name for path in base.glob("http_*.json") if path.name != "http_index.json"},
             "e2e1_mobile_http_inventory_invalid")
    records = []
    for item in index:
        row = _load(_safe_path(base,item["file"]))
        _verify_product_http_capture(row)
        _require(row["url"] == item["url"] and row["method"] == item["method"]
                 and row["response_status_code"] == item["status"] == 200
                 and row["url"].startswith(base_url + "/api/"), "e2e1_mobile_http_invalid")
        records.append(row)
    chats = [row for row in records if row["url"].endswith("/copilot/messages")]
    turns = json.loads((base / "exact_chat_io.json").read_text(encoding="utf-8"))
    _require(isinstance(turns,list), "e2e1_mobile_chat_count_invalid")
    prompts = ["Keep Stearns Wharf.", "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park and remove Stearns Wharf.",
               "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park."]
    _verify_mobile_keyboard(_load(base / "mobile_keyboard_assertions.json"),prompts)
    _require(len(chats) == len(turns) == 3, "e2e1_mobile_chat_count_invalid")
    for row, turn, prompt in zip(chats,turns,prompts,strict=True):
        events = _sse_events(row["response_body_utf8"])
        _require(json.loads(row["request_body_utf8"]) == turn["input"] and events == turn["sse_events"]
                 and events[-1]["payload"] == turn["output"] and turn["output"]["turn"]["user_message"] == prompt,
                 "e2e1_mobile_chat_io_invalid")
        _verify_copilot_stream(events, expected_client_message_id=turn["input"]["client_message_id"])
    _require([(intent["type"],intent["target_stop_id"]) for intent in turns[0]["output"]["turn"]["interpretation"]["intents"]] == [("keep_stop","stearns_wharf")]
             and turns[1]["output"]["turn"]["interpretation"]["state"] == "unsupported"
             and all(turn["output"]["session"]["draft"] == [] for turn in turns)
             and turns[2]["output"]["turn"]["interpretation"]["state"] == "proposal_ready", "e2e1_mobile_semantics_invalid")
    stages = [row for row in records if row["url"].endswith("/draft/operations")]
    previews = [row for row in records if row["url"].endswith("/preview")]
    accepts = [row for row in records if row["url"].endswith("/accept")]
    _require(len(stages) == len(previews) == len(accepts) == 1
             and json.loads(stages[0]["request_body_utf8"]) == trace["operation"]
             and json.loads(previews[0]["response_body_utf8"]) == payload, "e2e1_mobile_mutation_flow_invalid")
    _require({key:trace["operation"][key] for key in ("type","target","parameters")} == {
        "type":"replace_nearby","target":"bixby_creek_bridge_viewpoint",
        "parameters":{"candidate_id":"pfeiffer_big_sur_state_park"}}, "e2e1_mobile_operation_invalid")
    _require(proposal.get("execution_mode") == "deterministic_repair_pipeline"
             and isinstance(proposal.get("runtime_execution"),dict), "e2e1_mobile_execution_invalid")
    execution = proposal["runtime_execution"]
    child, certificate, diff = execution["child_plan"], execution["evaluation_certificate"], execution["plan_diff"]
    _require(execution["evaluation_executed"] is True
             and child["content_hash"] == _artifact_content_hash(child) == proposal["child_plan_content_hash"]
             and certificate["content_hash"] == _artifact_content_hash(certificate) == proposal["certificate_content_hash"]
             and certificate["plan_id"] == diff["child_plan_id"] == child["plan_id"] == proposal["child_plan_id"]
             and certificate["plan_content_hash"] == child["content_hash"]
             and certificate["comparison_eligibility"] == "eligible"
             and _artifact_content_hash(diff) == proposal["diff_content_hash"], "e2e1_mobile_evaluation_invalid")
    _verify_plan_ack(trace["rendered"],proposal)
    _verify_plan_ack(trace["reloaded"],proposal)
    stop_ids = [stop["stop_id"] for stop in child["selected_stops"]]
    _require(len(stop_ids) == len(set(stop_ids)) and stop_ids == child["sequence"]
             and set(trace["rendered"]["rendered_marker_ids"]) == set(stop_ids),
             "e2e1_mobile_marker_identity_invalid")
    _require(trace["initial"]["active_plan_id"] == proposal["parent_plan_id"] == child["parent_plan_id"] == diff["parent_plan_id"]
             and execution["request_id"] == execution["repair_request"]["request_id"] == child["planning_request_id"]
             == proposal["compiled_request"]["source_request_id"]
             and [stop["stop_id"] for stop in diff["added_stops"]] == ["pfeiffer_big_sur_state_park"]
             and [stop["stop_id"] for stop in diff["deleted_stops"]] == ["bixby_creek_bridge_viewpoint"],
             "e2e1_mobile_lineage_invalid")
    surfaces = _load(base / "preview_surfaces.json")
    _require(surfaces["metrics"]["planId"] == proposal["child_plan_id"]
             and surfaces["metrics"]["planContentHash"] == proposal["child_plan_content_hash"]
             and surfaces["metrics"]["evaluationId"] == proposal["certificate_id"]
             and "Pfeiffer Big Sur State Park" in surfaces["timeline"] and "Bixby Creek Bridge" not in surfaces["timeline"],
             "e2e1_mobile_surfaces_invalid")
    _require(trace["rendered"]["preview_state"] == "PREVIEW_RENDERED" and trace["reloaded"]["preview_state"] == "PARENT_RENDERED",
             "e2e1_mobile_acceptance_state_invalid")
    accepted = _load(base / "acceptance_response.json")
    decision = accepted["result"]["decision"]
    _require(json.loads(accepts[0]["response_body_utf8"]) == accepted and decision == trace["decision"]
             and decision["content_hash"] == _canonical_sha256({key:value for key,value in decision.items() if key != "content_hash"})
             and decision["selected_plan_id"] == proposal["child_plan_id"]
             and decision["certificate_content_hash"] == certificate["content_hash"]
             and accepted["session"]["accepted_plan_id"] == proposal["child_plan_id"], "e2e1_mobile_acceptance_invalid")
    panel = _load(base / "accepted_evidence.json")["scope"]
    _require(panel["evidenceScope"] == "accepted" and panel["evidencePlanId"] == proposal["child_plan_id"]
             and panel["planContentHash"] == proposal["child_plan_content_hash"]
             and panel["evaluationId"] == certificate["certificate_id"], "e2e1_mobile_evidence_invalid")
    text_route = _load(base / "mobile_text_route.json")
    _require(text_route["visible"] is True and bool(text_route["text"])
             and text_route["attributes"]["planId"] == proposal["child_plan_id"]
             and text_route["attributes"]["planContentHash"] == proposal["child_plan_content_hash"],
             "e2e1_mobile_text_route_invalid")
    labels = ("initial_mobile","chat_proposal_mobile","preview_mobile","text_route_mobile","accepted_reload_mobile")
    _verify_journey_captures(base,labels,{trace["initial"]["active_plan_id"],proposal["child_plan_id"]})
    for label in labels:
        _require(_load(base / f"{label}.readiness.json")["viewport"] == {"width":390,"height":844}, "e2e1_mobile_viewport_invalid")
    _require(json.loads((base / "console_errors.json").read_text(encoding="utf-8"))
             == json.loads((base / "network_failures.json").read_text(encoding="utf-8")) == [], "e2e1_mobile_browser_errors")


def verify(output_dir: Path) -> dict[str, Any]:
    root = output_dir.resolve()
    _require(root.is_dir(), "e2e1_output_directory_missing")
    manifest = _load(root / MANIFEST_NAME)
    if manifest.get("schema_version") == "e2e1-self-managed-browser-manifest-v1":
        from verify_e2e1_self_managed import verify as verify_self_managed

        return verify_self_managed(root)
    generated_at = _verify_root(manifest)
    _verify_files(root, manifest)
    _verify_provider_isolation(root, manifest)
    _verify_sources(manifest, generated_at)
    _verify_llm_records(root, manifest)
    _verify_report(root, manifest)
    _verify_text_security(root)
    _verify_runtime_environments(root, manifest)
    if manifest["scenarios"] == ["map_initialization_recovery"]:
        _verify_map_recovery(root, manifest["base_url"])
    elif manifest["scenarios"] == ["completed_preview_race"]:
        _verify_completed_race(root, manifest["base_url"])
    elif manifest["scenarios"] == ["workspace_stability"]:
        _verify_workspace_stability(root, manifest["base_url"])
    elif manifest["scenarios"] == ["direct_edits"]:
        _verify_direct_edits(root, manifest["base_url"])
    elif manifest["scenarios"] == ["addition_continuation"]:
        _verify_addition_continuation(root, manifest["base_url"])
    elif manifest["scenarios"] == ["mobile_chat"]:
        _verify_mobile_chat(root, manifest["base_url"])
    elif manifest["scenarios"] == ["waypoint_pointer"]:
        from verify_e2e1_waypoint_pointer import verify as verify_waypoint

        verify_waypoint(root, manifest["base_url"])
    elif manifest["scenarios"] == ["capability_controls"]:
        from verify_e2e1_capability_controls import verify as verify_controls

        verify_controls(root, manifest["base_url"])
    elif manifest["scenarios"] == ["evaluated_variants"]:
        from verify_e2e1_evaluated_variants import verify as verify_evaluated

        verify_evaluated(root, manifest["base_url"])
    elif manifest["scenarios"] == ["webgl_recovery"]:
        from verify_e2e1_webgl_recovery import verify as verify_webgl

        verify_webgl(root, manifest["base_url"])
    elif manifest["scenarios"] == ["chat_closure"]:
        from verify_e2e1_chat_closure import verify as verify_chat_closure

        verify_chat_closure(root, manifest["base_url"])
    elif manifest["scenarios"] == ["same_plan_remount"]:
        from verify_e2e1_same_plan_remount import verify as verify_remount

        verify_remount(root, manifest["base_url"])
    elif manifest["scenarios"] == ["waypoint_contexts"]:
        from verify_e2e1_waypoint_contexts import verify as verify_waypoint_contexts

        verify_waypoint_contexts(root, manifest["base_url"])
    elif manifest["scenarios"] == ["candidate_day_matrix"]:
        from verify_e2e1_candidate_day_matrix import verify as verify_candidate_days

        verify_candidate_days(root, manifest["base_url"])
    else:
        _verify_recorded_request_origins(root, manifest["base_url"])
        _verify_scenarios(root)
    return {
        "verified": True,
        "manifest_content_hash": manifest["content_hash"],
        "manifest_sha256": _sha256(root / MANIFEST_NAME),
        "report_sha256": _sha256(root / REPORT_NAME),
        "artifact_files": len(manifest["files"]),
        "source_inputs": len(manifest["source_inputs"]),
        "llm_records": len(manifest["llm_records"]),
        "provider_calls": manifest["provider_calls"],
        "scenarios": len(manifest["scenarios"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(verify(parse_args().output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
