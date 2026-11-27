"""Run browser suites which own their app lifecycles; never launch an outer server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from e2e1_provider_isolation import deterministic_environment
from run_e2e1_vertical_slice import (
    ROOT,
    E2E1RunnerError,
    _canonical_sha256,
    _port_is_available,
    _role,
    _sha256,
    _source_rows,
)

MODES = {
    "queued-preview": {
        "browser_test": "tests/product_app/browser/test_copilot_queued_preview_live.py",
        "scenario": "conversation",
        "environment": "PRODUCT_E2E_CONVERSATION",
    },
    "owned-edits": {
        "browser_test": "tests/product_app/browser/test_owned_combined_journey_live.py",
        "scenario": "conversation",
        "environment": "PRODUCT_E2E_CONVERSATION",
    },
    "preview-geography-fault": {
        "browser_test": "tests/product_app/browser/test_preview_geography_fault_live.py",
        "scenario": "preview_geography_fault",
        "environment": "PRODUCT_E2E_PREVIEW_GEOGRAPHY_FAULT",
    },
    "optional-evidence-fault": {
        "browser_test": "tests/product_app/browser/test_optional_evidence_fault_live.py",
        "scenario": "optional_evidence_fault",
        "environment": "PRODUCT_E2E_OPTIONAL_EVIDENCE_FAULT",
    },
    "bootstrap-recovery": {
        "browser_test": "tests/product_app/browser/test_bootstrap_recovery_live.py",
        "scenario": "bootstrap_recovery",
        "environment": "PRODUCT_E2E_BOOTSTRAP_RECOVERY",
    },
    "build-coherence": {
        "browser_test": "tests/product_app/browser/test_build_coherence_live.py",
        "scenario": "build_coherence",
        "environment": "PRODUCT_E2E_BUILD_COHERENCE",
    },
    "conversation": {
        "browser_test": "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
        "scenario": "conversation",
        "environment": "PRODUCT_E2E_CONVERSATION",
    },
    "combined-conflicts": {
        "browser_test": "tests/product_app/browser/test_copilot_combined_conflicts_live.py",
        "scenario": "conversation",
        "environment": "PRODUCT_E2E_CONVERSATION",
    },
}
SCHEMA = "e2e1-self-managed-browser-manifest-v1"


def files(root):
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "exact_input" if "fixture_inputs" in path.relative_to(root).parts else _role(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "exact_content_location": path.relative_to(root).as_posix(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in {root / "file_io_manifest.json", root / "exact_io_report.md"}
    ]


def redact_conversation_state(root: Path) -> dict[str, object]:
    """Preserve runtime state as evidence without retaining credential verifiers."""
    state_root = root / "_state"
    rows: list[dict[str, object]] = []
    for path in sorted(state_root.rglob("session_*.json")) if state_root.is_dir() else ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "token_verifier" not in payload:
            raise E2E1RunnerError("e2e1_conversation_state_credential_missing")
        before = _sha256(path)
        payload["token_verifier"] = "OMITTED_SECRET"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "redacted_field_names": ["token_verifier"],
                "pre_redaction_sha256": before,
                "post_redaction_sha256": _sha256(path),
            }
        )
    result: dict[str, object] = {
        "schema_version": "conversation-state-credential-redaction-v1",
        "scope": "completed_runtime_state_evidence",
        "assertions_passed": bool(rows),
        "redacted_file_count": len(rows),
        "files": rows,
    }
    (root / "state_credential_redaction.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def render_report(manifest):
    lines = [
        "# Self-managed browser exact input and output",
        "",
        f"- Mode: {manifest['mode']}",
        f"- Command: {' '.join(manifest['command'])}",
        f"- Assertions passed: {str(manifest['assertions_passed']).lower()}",
        f"- Source unchanged: {str(manifest['source_unchanged']).lower()}",
        f"- Measured live provider calls: {manifest['provider_calls']}",
        "- Each app lifecycle has its own finalized isolation receipt. No unrelated outer app server is launched.",
        "- The conversation cancellation case uses the real OpenAI SDK over explicit non-network fake HTTP transport. It is not live OpenAI connectivity.",
        "- Interrupted chat captures preserve received prefixes and are explicitly marked incomplete.",
        "",
        "## Exact file inputs and outputs",
        "",
        "| Role | File | Bytes | SHA-256 |",
        "| --- | --- | ---: | --- |",
    ]
    lines += [
        f"| {row['role']} | [{row['path']}]({row['path']}) | {row['size_bytes']} | {row['sha256']} |"
        for row in manifest["files"]
    ]
    lines += ["", "## Exact source inputs", "", "| File | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| {row['path']} | {row['size_bytes']} | {row['sha256']} |" for row in manifest["source_inputs"]]
    lines += [
        "",
        "## Exact chat and model inputs and outputs",
        "",
        "Every exact_chat_io.json retains the exact submitted website input and parsed SSE output; each indexed HTTP record retains exact captured UTF-8 with explicit credential redactions.",
        "fake_sdk_io.json retains the exact real-SDK-generated request body and fake provider response (or cancelled-response null). Real API credentials are neither loaded nor recorded.",
        "provider_scope.json and llm_io.json distinguish deterministic interpretation, fake transport and zero real provider dispatch. Their full original bytes are linked above.",
        "Completed conversation state is retained for lineage checks, while token verifiers are replaced by OMITTED_SECRET and bound by state_credential_redaction.json.",
    ]
    return "\n".join(lines) + "\n"


def run(*, mode, output_dir=None, port=None):
    from verify_e2e1_self_managed import verify_scenarios

    if mode not in MODES or port not in {None, 8127}:
        raise E2E1RunnerError("e2e1_self_managed_mode_or_origin_invalid")
    if not _port_is_available(8127):
        raise E2E1RunnerError("e2e1_port_8127_occupied")
    spec = MODES[mode]
    run_id = datetime.now(UTC).strftime("e2e1_self_%Y%m%dT%H%M%S%fZ")
    root = (output_dir or ROOT / "results/e2e" / run_id).resolve()
    if mode == "owned-edits" and not root.is_relative_to(ROOT):
        raise E2E1RunnerError("e2e1_owned_fixture_requires_repository_output_directory")
    if root.exists() and any(root.iterdir()):
        raise E2E1RunnerError("e2e1_output_directory_not_empty")
    root.mkdir(parents=True, exist_ok=True)
    source_inputs = _source_rows(spec["browser_test"])
    base = root / spec["scenario"]
    base.mkdir()
    environment = deterministic_environment(os.environ)
    environment.update(
        PRODUCT_E2E_BASE_URL="http://127.0.0.1:8127",
        PRODUCT_APP_ORIGIN="http://127.0.0.1:8127",
        PRODUCT_E2E_RUN_ID=run_id,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment[spec["environment"] + "_DISPOSABLE"] = "1"
    environment[spec["environment"] + "_EVIDENCE_DIR"] = str(base)
    if mode == "queued-preview":
        environment["PRODUCT_E2E_QUEUED_PREVIEW_DISPOSABLE"] = "1"
    process_record = {
        "pid": None,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "exit_code": None,
        "outer_app_server_started": False,
    }
    # Registered product fixtures must stay repository-relative and survive the
    # run so the independent verifier can reload the exact input package.
    temporary_scope = (
        nullcontext(str(root / "fixture_inputs"))
        if mode == "owned-edits"
        else tempfile.TemporaryDirectory(prefix="e2e1-self-")
    )
    with temporary_scope as temporary:
        Path(temporary).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            spec["browser_test"],
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(root / "pytest_junit.xml"),
            "--basetemp",
            str(Path(temporary) / "pytest"),
        ]
        process_record["command"] = command
        with (root / "pytest_output.txt").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, text=True
            )
            process_record["pid"] = process.pid
            process_record["exit_code"] = process.wait()
    process_record["completed_at"] = datetime.now(UTC).isoformat()
    (root / "test_process.json").write_text(
        json.dumps(process_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mode in {"conversation", "combined-conflicts", "owned-edits", "queued-preview"}:
        redact_conversation_state(root)
    unchanged = source_inputs == _source_rows(spec["browser_test"])
    result, failure = None, None
    if process_record["exit_code"] == 0 and unchanged:
        try:
            result = verify_scenarios(root, mode=mode, run_id=run_id, process=process_record)
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            failure = {
                "code": str(error) if isinstance(error, RuntimeError) else "e2e1_self_managed_evidence_malformed",
                "error_type": type(error).__name__,
                "assertions_passed": False,
            }
            (root / "scenario_verification_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
    manifest = {
        "schema_version": SCHEMA,
        "mode": mode,
        "scenarios": [spec["scenario"]],
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": "http://127.0.0.1:8127",
        "command": command,
        "test_process": process_record,
        "pytest_exit_code": process_record["exit_code"],
        "assertions_passed": result is not None and process_record["exit_code"] == 0 and unchanged,
        "source_inputs": source_inputs,
        "source_unchanged": unchanged,
        "files": files(root),
        "provider_calls": result["provider_calls"] if result is not None else None,
        "verification": result,
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    (root / "file_io_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "exact_io_report.md").write_text(render_report(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "assertions_passed": manifest["assertions_passed"],
                "provider_calls": manifest["provider_calls"],
                "outer_app_server_started": False,
            },
            indent=2,
        )
    )
    return 0 if manifest["assertions_passed"] else 1
