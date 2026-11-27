"""Run the complete deterministic E2E-1 browser gate from one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from e2e1_provider_isolation import deterministic_environment  # noqa: E402 - script bootstrap above
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402 - script bootstrap above

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
BROWSER_TEST = "tests/product_app/browser/test_e2e1_vertical_slice_live.py"
MAP_RECOVERY_BROWSER_TEST = "tests/product_app/browser/test_map_initialization_recovery_live.py"
COMPLETED_RACE_BROWSER_TEST = "tests/product_app/browser/test_completed_preview_race_live.py"
COMPLETED_RACE_SCENARIOS = (("PRODUCT_E2E_COMPLETED_RACE_EVIDENCE_DIR", "completed_preview_race"),)
WORKSPACE_STABILITY_BROWSER_TEST = "tests/product_app/browser/test_workspace_stability_live.py"
WORKSPACE_STABILITY_SCENARIOS = (("PRODUCT_E2E_STABILITY_EVIDENCE_DIR", "workspace_stability"),)
DIRECT_EDITS_BROWSER_TEST = "tests/product_app/browser/test_direct_edit_journeys_live.py"
DIRECT_EDITS_SCENARIOS = (("PRODUCT_E2E_DIRECT_EDITS_EVIDENCE_DIR", "direct_edits"),)
ADDITION_BROWSER_TEST = "tests/product_app/browser/test_addition_continuation_live.py"
ADDITION_SCENARIOS = (("PRODUCT_E2E_ADDITION_CONTINUATION_EVIDENCE_DIR", "addition_continuation"),)
MOBILE_CHAT_BROWSER_TEST = "tests/product_app/browser/test_mobile_chat_journey_live.py"
MOBILE_CHAT_SCENARIOS = (("PRODUCT_E2E_MOBILE_CHAT_EVIDENCE_DIR", "mobile_chat"),)
WAYPOINT_BROWSER_TEST = "tests/product_app/browser/test_waypoint_pointer_live.py"
WAYPOINT_SCENARIOS = (("PRODUCT_E2E_WAYPOINT_POINTER_EVIDENCE_DIR", "waypoint_pointer"),)
CONTROLS_BROWSER_TEST = "tests/product_app/browser/test_capability_controls_live.py"
CONTROLS_SCENARIOS = (("PRODUCT_E2E_CAPABILITY_CONTROLS_EVIDENCE_DIR", "capability_controls"),)
EVALUATED_BROWSER_TEST = "tests/product_app/browser/test_evaluated_variants_live.py"
EVALUATED_SCENARIOS = (("PRODUCT_E2E_EVALUATED_VARIANTS_EVIDENCE_DIR", "evaluated_variants"),)
WEBGL_BROWSER_TEST = "tests/product_app/browser/test_webgl_recovery_live.py"
WEBGL_SCENARIOS = (("PRODUCT_E2E_WEBGL_RECOVERY_EVIDENCE_DIR", "webgl_recovery"),)
CHAT_CLOSURE_BROWSER_TEST = "tests/product_app/browser/test_chat_runtime_closure_live.py"
CHAT_CLOSURE_SCENARIOS = (("PRODUCT_E2E_CHAT_CLOSURE_EVIDENCE_DIR", "chat_closure"),)
REMOUNT_BROWSER_TEST = "tests/product_app/browser/test_same_plan_remount_live.py"
REMOUNT_SCENARIOS = (("PRODUCT_E2E_REMOUNT_EVIDENCE_DIR", "same_plan_remount"),)
WAYPOINT_CONTEXTS_BROWSER_TEST = "tests/product_app/browser/test_waypoint_contexts_live.py"
WAYPOINT_CONTEXTS_SCENARIOS = (("PRODUCT_E2E_WAYPOINT_CONTEXTS_EVIDENCE_DIR", "waypoint_contexts"),)
CANDIDATE_DAYS_BROWSER_TEST = "tests/product_app/browser/test_candidate_day_matrix_live.py"
CANDIDATE_DAYS_SCENARIOS = (("PRODUCT_E2E_CANDIDATE_DAY_MATRIX_EVIDENCE_DIR", "candidate_day_matrix"),)
MAP_RECOVERY_SCENARIOS = (("PRODUCT_E2E_MAP_RECOVERY_EVIDENCE_DIR", "map_initialization_recovery"),)
MANIFEST_NAME = "file_io_manifest.json"
REPORT_NAME = "exact_io_report.md"
SCENARIOS = (
    ("PRODUCT_E2E_EVIDENCE_DIR", "registered_replacement"),
    ("PRODUCT_E2E_POI_EVIDENCE_DIR", "poi_discovery"),
    ("PRODUCT_E2E_GUARD_EVIDENCE_DIR", "unsupported_removal"),
    ("PRODUCT_E2E_INELIGIBLE_EVIDENCE_DIR", "ineligible_duration"),
    ("PRODUCT_E2E_STALE_EVIDENCE_DIR", "stale_response"),
    ("PRODUCT_E2E_ACCEPT_EVIDENCE_DIR", "accepted_reload"),
)
SOURCE_INPUTS = deterministic_source_inputs(BROWSER_TEST)


class E2E1RunnerError(RuntimeError):
    """Stable runner failure."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _allocate_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((HOST, 0))
        return int(listener.getsockname()[1])


def _port_is_available(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((HOST, port)) != 0


def _health(base_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as response:
            value = json.loads(response.read(65_537).decode())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _wait_until_ready(
    process: subprocess.Popen[str],
    base_url: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise E2E1RunnerError("e2e1_server_exited_before_ready")
        health = _health(base_url)
        components = health.get("components", {}) if health else {}
        if not isinstance(components, dict):
            raise E2E1RunnerError("e2e1_provider_isolation_preflight_failed")
        copilot = components.get("openai", {})
        if health and (
            not isinstance(copilot, dict)
            or copilot.get("status") != "disabled"
            or copilot.get("code") != "deterministic_adapter_selected"
        ):
            raise E2E1RunnerError("e2e1_provider_isolation_preflight_failed")
        if any(not isinstance(components.get(name, {}), dict) for name in ("map", "routing")):
            raise E2E1RunnerError("e2e1_server_health_invalid")
        if (
            health
            and health.get("ready") is True
            and components.get("map", {}).get("status") == "ready"
            and components.get("routing", {}).get("status") == "ready"
        ):
            return health
        time.sleep(0.25)
    raise E2E1RunnerError("e2e1_server_not_map_ready")


def _role(path: Path) -> str:
    if path.name.endswith("llm_io.json"):
        return "exact_llm_input_output"
    return "exact_input" if "request" in path.name else "exact_output"


def _source_rows(browser_test: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "size_bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha256(ROOT / relative),
        }
        for relative in (deterministic_source_inputs(browser_test) if browser_test else SOURCE_INPUTS)
    ]


def _isolation_passed(isolation: dict[str, Any]) -> bool:
    record = isolation.get("record")
    if not isinstance(record, dict):
        return False
    return bool(
        isolation.get("artifact") == "provider_isolation.json"
        and record.get("schema_version") == "e2e1-provider-isolation-v1"
        and isinstance(record.get("run_id"), str)
        and bool(record["run_id"].strip())
        and record.get("run_id") == isolation.get("run_id")
        and type(record.get("process_id")) is int
        and record["process_id"] > 0
        and type(isolation.get("server_pid")) is int
        and record["process_id"] == isolation.get("server_pid")
        and record.get("scope") == "application_server"
        and record.get("guard_boundaries") == ["httpx.Client.send", "httpx.AsyncClient.send", "socket.connect"]
        and all(
            record.get(name) is True
            for name in (
                "finalized",
                "api_key_disabled",
                "app_initialized",
                "server_started",
                "server_stopped",
            )
        )
        and record.get("installed_at")
        and record.get("finalized_at")
        and record.get("effective_adapter") == "deterministic"
        and record.get("actual_adapter") == "DeterministicCopilotAdapter"
        and type(record.get("sent_provider_calls")) is int
        and record["sent_provider_calls"] == 0
        and type(record.get("allowed_local_connections")) is int
        and record["allowed_local_connections"] > 0
        and record.get("blocked_attempts") == []
        and record.get("failure_code") is None
    )


def build_manifest(
    output_dir: Path,
    *,
    command: list[str],
    pytest_exit_code: int,
    health: dict[str, Any],
    base_url: str,
    source_inputs: list[dict[str, Any]] | None = None,
    run_id: str = "unrecorded",
    server_pid: int | None = None,
    scenarios: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    isolation_path = output_dir / "provider_isolation.json"
    try:
        isolation_record = json.loads(isolation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        isolation_record = None
    isolation = {
        "artifact": "provider_isolation.json",
        "run_id": run_id,
        "server_pid": server_pid,
        "record": isolation_record,
    }
    isolation_passed = _isolation_passed(isolation)
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in {MANIFEST_NAME, REPORT_NAME}:
            continue
        files.append(
            {
                "path": relative,
                "role": _role(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "exact_content_location": relative,
            }
        )
    sources = source_inputs if source_inputs is not None else _source_rows()
    llm_records = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "record": json.loads(path.read_text(encoding="utf-8")),
        }
        for path in sorted(output_dir.rglob("*llm_io.json"))
    ]
    manifest: dict[str, Any] = {
        "schema_version": "e2e1-file-io-manifest-v3",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "command": command,
        "pytest_exit_code": pytest_exit_code,
        "assertions_passed": pytest_exit_code == 0
        and isolation_passed
        and all((ROOT / row["path"]).is_file() and _sha256(ROOT / row["path"]) == row["sha256"] for row in sources),
        "health": {
            "build_id": health.get("build_id"),
            "map": health.get("components", {}).get("map", {}).get("code"),
            "routing": health.get("components", {}).get("routing", {}).get("code"),
            "copilot": health.get("components", {}).get("openai", {}).get("code"),
        },
        "scenarios": list(scenarios) if scenarios is not None else [name for _, name in SCENARIOS],
        "provider_calls": isolation_record["sent_provider_calls"] if isolation_passed else None,
        "provider_isolation": isolation,
        "llm_input_output_policy": (
            "Scenario LLM records describe deterministic interpretation. "
            "Measured-zero claims require finalized application-server provider_isolation.json; "
            "blocked attempts or incomplete telemetry fail the gate."
        ),
        "files": files,
        "source_inputs": sources,
        "llm_records": llm_records,
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    return manifest


def verify_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    seed = dict(manifest)
    expected = seed.pop("content_hash", None)
    if expected != _canonical_sha256(seed):
        raise E2E1RunnerError("e2e1_manifest_content_hash_mismatch")
    for row in manifest.get("files", []):
        path = output_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or _sha256(path) != row["sha256"]
            or row["exact_content_location"] != row["path"]
        ):
            raise E2E1RunnerError("e2e1_artifact_binding_mismatch")
    for row in manifest.get("source_inputs", []):
        path = ROOT / str(row["path"])
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or _sha256(path) != row["sha256"]:
            raise E2E1RunnerError("e2e1_source_binding_mismatch")
    if not require_complete:
        return
    if not _isolation_passed(manifest.get("provider_isolation", {})):
        raise E2E1RunnerError("e2e1_provider_isolation_unverified")
    if manifest.get("provider_calls") != 0 or manifest.get("assertions_passed") is not True:
        raise E2E1RunnerError("e2e1_provider_isolation_unverified")
    llm_records = manifest.get("llm_records")
    scenario_paths = {
        tuple(name for _, name in SCENARIOS): sorted(
            [f"{name}/llm_io.json" for _, name in SCENARIOS] + ["registered_replacement/one_click_llm_io.json"]
        ),
        ("map_initialization_recovery",): [
            f"map_initialization_recovery/{context}/llm_io.json" for context in ("accepted_child", "original")
        ],
        ("completed_preview_race",): ["completed_preview_race/llm_io.json"],
        ("workspace_stability",): sorted(
            f"workspace_stability/{size}/llm_io.json" for size in ("1440x900", "1280x800", "1024x768", "390x844")
        ),
        ("direct_edits",): sorted(
            f"direct_edits/{kind}/llm_io.json" for kind in ("combined", "duration", "time_window", "role", "order")
        ),
        ("addition_continuation",): ["addition_continuation/llm_io.json"],
        ("mobile_chat",): ["mobile_chat/llm_io.json"],
        ("waypoint_pointer",): ["waypoint_pointer/llm_io.json"],
        ("capability_controls",): [
            "capability_controls/accepted_child/llm_io.json",
            "capability_controls/original/llm_io.json",
        ],
        ("evaluated_variants",): [
            "evaluated_variants/accepted_child/llm_io.json",
            "evaluated_variants/original/llm_io.json",
        ],
        ("webgl_recovery",): ["webgl_recovery/llm_io.json"],
        ("chat_closure",): ["chat_closure/accepted_child/llm_io.json", "chat_closure/original/llm_io.json"],
        ("same_plan_remount",): ["same_plan_remount/llm_io.json"],
        ("waypoint_contexts",): [
            "waypoint_contexts/accepted_child/llm_io.json",
            "waypoint_contexts/original/llm_io.json",
        ],
        ("candidate_day_matrix",): ["candidate_day_matrix/llm_io.json"],
    }
    scenarios = manifest.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or not all(isinstance(name, str) for name in scenarios)
        or tuple(scenarios) not in scenario_paths
    ):
        raise E2E1RunnerError("e2e1_scenario_universe_invalid")
    if (
        not isinstance(llm_records, list)
        or not all(isinstance(item, dict) for item in llm_records)
        or [item.get("path") for item in llm_records] != scenario_paths[tuple(scenarios)]
    ):
        raise E2E1RunnerError("e2e1_llm_record_universe_mismatch")
    for item in llm_records:
        record = item.get("record", {})
        expected_keys = {"llm_input", "llm_output", "provider_call_count", "reason"}
        if item.get("path") == "unsupported_removal/llm_io.json":
            expected_keys.add("verification_test")
        if item.get("path") in {"chat_closure/original/llm_io.json", "chat_closure/accepted_child/llm_io.json"}:
            expected_keys.add("actual_openai_connectivity_claimed")
        if (
            set(record) != expected_keys
            or record.get("provider_call_count") != 0
            or record.get("llm_input") is not None
            or record.get("llm_output") is not None
            or not isinstance(record.get("reason"), str)
            or not record["reason"].strip()
            or (
                "actual_openai_connectivity_claimed" in expected_keys
                and record.get("actual_openai_connectivity_claimed") is not False
            )
            or (
                "verification_test" in expected_keys
                and record.get("verification_test")
                != "tests/product_app/test_openai_copilot.py::test_removal_language_is_rejected_before_provider_call"
            )
        ):
            raise E2E1RunnerError("e2e1_unexpected_llm_io")


def render_report(manifest: dict[str, Any]) -> str:
    record = manifest["provider_isolation"]["record"]
    lines = [
        "# E2E-1 Exact File and LLM Input/Output Report",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Command: {' '.join(manifest['command'])}",
        f"- Assertions passed: {str(manifest['assertions_passed']).lower()}",
        f"- Provider calls: {manifest['provider_calls']}",
        f"- Blocked provider attempts: {len(record['blocked_attempts']) if isinstance(record, dict) and isinstance(record.get('blocked_attempts'), list) else 'unknown'}",
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
            f"| {row['role']} | {row['path']} | {row['size_bytes']} | {row['sha256']} | [{location}]({location}) |"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Loopback product port; defaults to 8127 because the local map asset "
            "service intentionally pins CORS to that product origin."
        ),
    )
    parser.add_argument("--live-copilot-chat", action="store_true")
    parser.add_argument("--confirm-billed-synthetic-context", action="store_true")
    parser.add_argument(
        "--browser-test",
        choices=(
            "vertical-slice",
            "map-recovery",
            "completed-response-race",
            "workspace-stability",
            "direct-edits",
            "addition-continuation",
            "mobile-chat",
            "waypoint-pointer",
            "capability-controls",
            "evaluated-variants",
            "webgl-recovery",
            "chat-closure",
            "same-plan-remount",
            "waypoint-contexts",
            "candidate-day-matrix",
            "conversation",
            "combined-conflicts",
            "owned-edits",
            "build-coherence",
            "bootstrap-recovery",
            "optional-evidence-fault",
            "preview-geography-fault",
            "queued-preview",
            "aggregate",
        ),
        default="vertical-slice",
        help="A repository-known deterministic suite; map-recovery covers original and accepted-child journeys.",
    )
    return parser.parse_args()


def _persist_manifest(output_dir: Path, **inputs: Any) -> dict[str, Any]:
    """Bind the final result, including a retained semantic-verification failure."""
    manifest = build_manifest(output_dir, **inputs)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_NAME).write_text(render_report(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    if getattr(args, "browser_test", "") in {
        "conversation",
        "combined-conflicts",
        "owned-edits",
        "build-coherence",
        "bootstrap-recovery",
        "optional-evidence-fault",
        "preview-geography-fault",
        "queued-preview",
    }:
        if args.live_copilot_chat or args.confirm_billed_synthetic_context:
            raise E2E1RunnerError("e2e1_self_managed_is_non_live_only")
        from run_e2e1_self_managed import run

        return run(mode=args.browser_test, output_dir=args.output_dir, port=args.port)
    if getattr(args, "browser_test", "vertical-slice") == "aggregate":
        if args.live_copilot_chat or args.confirm_billed_synthetic_context:
            raise E2E1RunnerError("e2e1_aggregate_is_deterministic_only")
        from run_e2e1_aggregate import run_aggregate

        return run_aggregate(output_dir=args.output_dir, port=args.port)
    recovery = getattr(args, "browser_test", "vertical-slice") == "map-recovery"
    completed_race = getattr(args, "browser_test", "vertical-slice") == "completed-response-race"
    stability = getattr(args, "browser_test", "vertical-slice") == "workspace-stability"
    direct_edits = getattr(args, "browser_test", "vertical-slice") == "direct-edits"
    addition = getattr(args, "browser_test", "vertical-slice") == "addition-continuation"
    mobile_chat = getattr(args, "browser_test", "vertical-slice") == "mobile-chat"
    waypoint = getattr(args, "browser_test", "vertical-slice") == "waypoint-pointer"
    controls = getattr(args, "browser_test", "vertical-slice") == "capability-controls"
    evaluated = getattr(args, "browser_test", "vertical-slice") == "evaluated-variants"
    webgl = getattr(args, "browser_test", "vertical-slice") == "webgl-recovery"
    chat_closure = getattr(args, "browser_test", "vertical-slice") == "chat-closure"
    remount = getattr(args, "browser_test", "vertical-slice") == "same-plan-remount"
    waypoint_contexts = getattr(args, "browser_test", "vertical-slice") == "waypoint-contexts"
    candidate_days = getattr(args, "browser_test", "vertical-slice") == "candidate-day-matrix"
    if (
        direct_edits
        or addition
        or mobile_chat
        or waypoint
        or controls
        or evaluated
        or webgl
        or chat_closure
        or remount
        or waypoint_contexts
        or candidate_days
    ) and (args.live_copilot_chat or args.confirm_billed_synthetic_context):
        raise E2E1RunnerError("e2e1_edit_journeys_are_deterministic_only")
    if stability and (args.live_copilot_chat or args.confirm_billed_synthetic_context):
        raise E2E1RunnerError("e2e1_workspace_stability_is_deterministic_only")
    if completed_race and (args.live_copilot_chat or args.confirm_billed_synthetic_context):
        raise E2E1RunnerError("e2e1_completed_race_is_deterministic_only")
    if recovery and (args.live_copilot_chat or args.confirm_billed_synthetic_context):
        raise E2E1RunnerError("e2e1_recovery_is_deterministic_only")
    if args.live_copilot_chat:
        from run_e2e1_live_copilot_chat import run_live

        return run_live(
            output_dir=args.output_dir,
            confirmed=args.confirm_billed_synthetic_context,
            port=args.port,
        )
    if args.confirm_billed_synthetic_context:
        raise E2E1RunnerError("e2e1_live_copilot_flag_required")
    port = args.port if args.port is not None else 8127
    if not 1 <= port <= 65_535:
        raise E2E1RunnerError("e2e1_port_invalid")
    base_url = f"http://{HOST}:{port}"
    run_id = datetime.now(UTC).strftime("e2e1_vertical_slice_%Y%m%dT%H%M%S%fZ")
    output_dir = (args.output_dir or ROOT / "results" / "e2e" / run_id).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2E1RunnerError("e2e1_output_directory_not_empty")
    if not _port_is_available(port):
        raise E2E1RunnerError(f"e2e1_port_{port}_occupied")
    output_dir.mkdir(parents=True, exist_ok=True)
    browser_test = (
        COMPLETED_RACE_BROWSER_TEST if completed_race else MAP_RECOVERY_BROWSER_TEST if recovery else BROWSER_TEST
    )
    scenarios = COMPLETED_RACE_SCENARIOS if completed_race else MAP_RECOVERY_SCENARIOS if recovery else SCENARIOS
    if stability:
        browser_test, scenarios = WORKSPACE_STABILITY_BROWSER_TEST, WORKSPACE_STABILITY_SCENARIOS
    if direct_edits:
        browser_test, scenarios = DIRECT_EDITS_BROWSER_TEST, DIRECT_EDITS_SCENARIOS
    if addition:
        browser_test, scenarios = ADDITION_BROWSER_TEST, ADDITION_SCENARIOS
    if mobile_chat:
        browser_test, scenarios = MOBILE_CHAT_BROWSER_TEST, MOBILE_CHAT_SCENARIOS
    if waypoint:
        browser_test, scenarios = WAYPOINT_BROWSER_TEST, WAYPOINT_SCENARIOS
    if controls:
        browser_test, scenarios = CONTROLS_BROWSER_TEST, CONTROLS_SCENARIOS
    if evaluated:
        browser_test, scenarios = EVALUATED_BROWSER_TEST, EVALUATED_SCENARIOS
    if webgl:
        browser_test, scenarios = WEBGL_BROWSER_TEST, WEBGL_SCENARIOS
    if chat_closure:
        browser_test, scenarios = CHAT_CLOSURE_BROWSER_TEST, CHAT_CLOSURE_SCENARIOS
    if remount:
        browser_test, scenarios = REMOUNT_BROWSER_TEST, REMOUNT_SCENARIOS
    if waypoint_contexts:
        browser_test, scenarios = WAYPOINT_CONTEXTS_BROWSER_TEST, WAYPOINT_CONTEXTS_SCENARIOS
    if candidate_days:
        browser_test, scenarios = CANDIDATE_DAYS_BROWSER_TEST, CANDIDATE_DAYS_SCENARIOS
    source_inputs = _source_rows(browser_test) if browser_test != BROWSER_TEST else _source_rows()
    command = [
        sys.executable,
        "-m",
        "pytest",
        browser_test,
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        str(output_dir / "pytest_junit.xml"),
    ]
    health: dict[str, Any] = {}
    exit_code = 1
    server: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="e2e1-run-") as temporary:
        temp_root = Path(temporary)
        environment = deterministic_environment(os.environ)
        environment["PRODUCT_E2E_BASE_URL"] = base_url
        environment["PRODUCT_E2E_RUN_ID"] = run_id
        environment["PRODUCT_APP_ORIGIN"] = base_url
        if recovery:
            environment["PRODUCT_E2E_MAP_RECOVERY_DISPOSABLE"] = "1"
        if direct_edits:
            environment["PRODUCT_E2E_DIRECT_EDITS_DISPOSABLE"] = "1"
        if addition:
            environment["PRODUCT_E2E_ADDITION_CONTINUATION_DISPOSABLE"] = "1"
        if mobile_chat:
            environment["PRODUCT_E2E_MOBILE_CHAT_DISPOSABLE"] = "1"
        if waypoint:
            environment["PRODUCT_E2E_WAYPOINT_POINTER_DISPOSABLE"] = "1"
        if controls:
            environment["PRODUCT_E2E_CAPABILITY_CONTROLS_DISPOSABLE"] = "1"
        if evaluated:
            environment["PRODUCT_E2E_EVALUATED_VARIANTS_DISPOSABLE"] = "1"
        if webgl:
            environment["PRODUCT_E2E_WEBGL_RECOVERY_DISPOSABLE"] = "1"
        if chat_closure:
            environment["PRODUCT_E2E_CHAT_CLOSURE_DISPOSABLE"] = "1"
            environment["PRODUCT_E2E_CHAT_CLOSURE_STATE_ROOT"] = str(temp_root / "state")
        if remount:
            environment["PRODUCT_E2E_REMOUNT_DISPOSABLE"] = "1"
        if waypoint_contexts:
            environment["PRODUCT_E2E_WAYPOINT_CONTEXTS_DISPOSABLE"] = "1"
        if candidate_days:
            environment["PRODUCT_E2E_CANDIDATE_DAY_MATRIX_DISPOSABLE"] = "1"
        for variable, scenario in scenarios:
            directory = output_dir / scenario
            directory.mkdir(parents=True, exist_ok=True)
            environment[variable] = str(directory)
        command += ["--basetemp", str(temp_root / "pytest")]
        try:
            with (
                (output_dir / "server.stdout.log").open("w", encoding="utf-8") as stdout,
                (output_dir / "server.stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/e2e1_provider_isolation.py",
                        "--port",
                        str(port),
                        "--state-root",
                        str(temp_root / "state"),
                        "--output",
                        str(output_dir / "provider_isolation.json"),
                        "--run-id",
                        run_id,
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    stdin=subprocess.PIPE,
                )
                health = _wait_until_ready(server, base_url)
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                exit_code = completed.returncode
                (output_dir / "pytest_output.txt").write_text(
                    completed.stdout + completed.stderr,
                    encoding="utf-8",
                )
        except (E2E1RunnerError, OSError) as error:
            prior_exit_code = exit_code
            exit_code = exit_code or 1
            (output_dir / "runner_failure.json").write_text(
                json.dumps(
                    {
                        "code": str(error) if isinstance(error, E2E1RunnerError) else "e2e1_runner_os_error",
                        "pytest_exit_code": prior_exit_code,
                        "runner_exit_code": exit_code,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            if server is not None and server.poll() is None:
                try:
                    server.communicate("stop\n", timeout=20)
                except (subprocess.TimeoutExpired, OSError):
                    server.kill()
                    server.wait(timeout=5)
                if server.returncode != 0:
                    exit_code = exit_code or 1
    if recovery and exit_code == 0:
        from verify_e2e1_vertical_slice import E2E1VerificationError, _verify_map_recovery

        try:
            _verify_map_recovery(output_dir, base_url)
        except (E2E1VerificationError, OSError, ValueError, KeyError) as error:
            exit_code = 1
            (output_dir / "recovery_verification_failure.json").write_text(
                json.dumps(
                    {
                        "assertions_passed": False,
                        "error_type": type(error).__name__,
                        "code": str(error),
                    }
                ),
                encoding="utf-8",
            )
    manifest_inputs = dict(
        command=command,
        pytest_exit_code=exit_code,
        health=health,
        base_url=base_url,
        source_inputs=source_inputs,
        run_id=run_id,
        server_pid=server.pid if server is not None else None,
        scenarios=tuple(name for _, name in scenarios),
    )
    manifest = _persist_manifest(output_dir, **manifest_inputs)
    verify_manifest(output_dir, manifest, require_complete=manifest["assertions_passed"])
    exit_code = exit_code or (0 if manifest["assertions_passed"] else 1)
    if exit_code == 0:
        from verify_e2e1_vertical_slice import E2E1VerificationError, verify

        try:
            verify(output_dir)
        except (E2E1VerificationError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            exit_code = 1
            (output_dir / "semantic_verification_failure.json").write_text(
                json.dumps(
                    {
                        "assertions_passed": False,
                        "error_type": type(error).__name__,
                        "code": str(error),
                        "pytest_exit_code": manifest_inputs["pytest_exit_code"],
                        "runner_exit_code": exit_code,
                    }
                ),
                encoding="utf-8",
            )
            manifest = _persist_manifest(output_dir, **dict(manifest_inputs, pytest_exit_code=exit_code))
            verify_manifest(output_dir, manifest, require_complete=False)
    print(
        json.dumps(
            {
                "verified": exit_code == 0,
                "output_dir": str(output_dir),
                "manifest_content_hash": manifest["content_hash"],
                "artifact_files": len(manifest["files"]),
                "source_inputs": len(manifest["source_inputs"]),
                "llm_records": len(manifest["llm_records"]),
                "provider_calls": manifest["provider_calls"],
                "provider_isolation_passed": _isolation_passed(manifest["provider_isolation"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
