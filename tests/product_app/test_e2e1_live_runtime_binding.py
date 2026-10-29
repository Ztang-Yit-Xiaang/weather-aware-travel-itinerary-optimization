"""Paid/replay lifecycle binding using synthetic receipts; never starts a server."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "live_runtime_verifier", ROOT / "scripts/verify_e2e1_live_copilot_chat.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
ENV_SPEC = importlib.util.spec_from_file_location(
    "runtime_receipt_fixtures", Path(__file__).with_name("test_e2e1_runtime_environment.py")
)
environment_tests = importlib.util.module_from_spec(ENV_SPEC)
ENV_SPEC.loader.exec_module(environment_tests)
captured = environment_tests.captured


@pytest.fixture
def bundle(tmp_path, captured):
    root = tmp_path / "synthetic_live_run"
    base = root / "live_copilot_chat"
    base.mkdir(parents=True)

    def stamp(seconds):
        return f"2026-09-08T01:00:{seconds:02d}+00:00"

    node = "tests/product_app/browser/test_e2e1_live_copilot_chat.py::test_live_website_copilot_stages_and_renders_exact_evaluated_child"
    receipt = {
        "schema_version": "e2e1-live-run-lifecycle-v1",
        "run_id": root.name,
        "execution_mode": "live_openai",
        "base_url": "http://127.0.0.1:8127",
        "process_id": 123,
        "observed_at": stamp(0),
        "server_ready_at": stamp(1),
        "pytest_started_at": stamp(2),
        "pytest_completed_at": stamp(7),
        "server_stopped_at": stamp(8),
        "finalized_at": stamp(9),
        "finalized": True,
        "pytest_exit_code": 0,
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/product_app/browser/test_e2e1_live_copilot_chat.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "pytest"),
            "--junitxml",
            str(root / "pytest_junit.xml"),
        ],
    }
    manifest = {
        "run_lifecycle": {"artifact": "run_lifecycle.json", "record": deepcopy(receipt)},
        "authorization": {"execution_mode": "live_openai"},
        "base_url": receipt["base_url"],
        "command": receipt["command"],
        "pytest_exit_code": 0,
        "generated_at": stamp(10),
    }
    witness = captured.capture()
    witness.update(run_id=root.name, observed_at=stamp(3), completed_at=stamp(4))
    witness["osrm"]["observed_at"] = stamp(3)
    runtime_path = base / ("runtime_environment_" + verifier._hash_text(node)[:16] + ".json")
    runtime_path.write_text(json.dumps(witness), encoding="utf-8")
    readiness = {
        "screenshot_started_at": stamp(5),
        "screenshot_completed_at": stamp(6),
        "runtime_environment": {
            "artifact": runtime_path.name,
            "sha256": verifier._sha256(runtime_path),
            "browser": witness["browser"],
            "test_nodeid": node,
        },
    }
    (root / "run_lifecycle.json").write_text(json.dumps(receipt), encoding="utf-8")
    (base / "before.readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
    (base / "before.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic_binding_fixture")
    (root / "pytest_junit.xml").write_text(
        '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.product_app.browser.test_e2e1_live_copilot_chat" '
        'name="test_live_website_copilot_stages_and_renders_exact_evaluated_child"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=root,
        base=base,
        manifest=manifest,
        receipt=receipt,
        witness=witness,
        runtime_path=runtime_path,
        readiness=readiness,
        stamp=stamp,
    )


@pytest.mark.parametrize("mode", ["live_openai", "recorded_exact_output_replay"])
def test_actual_journey_receipt_is_bound_to_finalized_live_or_replay_run(bundle, mode):
    bundle.manifest["authorization"]["execution_mode"] = mode
    bundle.receipt["execution_mode"] = mode
    bundle.manifest["run_lifecycle"]["record"] = deepcopy(bundle.receipt)
    (bundle.root / "run_lifecycle.json").write_text(json.dumps(bundle.receipt), encoding="utf-8")
    verifier._verify_command(bundle.manifest["command"])
    verifier._verify_runtime_environments(bundle.root, bundle.manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "pid_bool",
        "exit_bool",
        "unfinalized",
        "different_run",
        "different_mode",
        "late_screenshot",
        "early_runtime",
        "record_mismatch",
        "missing_lifecycle",
        "missing_runtime",
        "extra_runtime",
        "runtime_hash",
        "runtime_browser",
        "runtime_origin",
        "wrong_node",
        "failed_junit",
        "extra_junit",
        "missing_png",
    ],
)
def test_lifecycle_and_receipt_forgeries_fail_closed(bundle, mutation):
    if mutation == "pid_bool":
        bundle.receipt["process_id"] = True
    elif mutation == "exit_bool":
        bundle.receipt["pytest_exit_code"] = False
    elif mutation == "unfinalized":
        bundle.receipt["finalized"] = False
    elif mutation == "different_run":
        bundle.receipt["run_id"] = "another_run"
    elif mutation == "different_mode":
        bundle.receipt["execution_mode"] = "recorded_exact_output_replay"
    elif mutation == "record_mismatch":
        bundle.receipt["process_id"] = 456
    elif mutation == "late_screenshot":
        bundle.readiness["screenshot_completed_at"] = bundle.stamp(11)
    elif mutation == "early_runtime":
        bundle.witness["observed_at"] = bundle.stamp(0)
    elif mutation == "runtime_hash":
        bundle.readiness["runtime_environment"]["sha256"] = "0" * 64
    elif mutation == "runtime_browser":
        bundle.readiness["runtime_environment"]["browser"] = {"version": "0.0.0"}
    elif mutation == "runtime_origin":
        bundle.witness["application_origin"] = "http://127.0.0.1:8128"
    elif mutation == "wrong_node":
        bundle.readiness["runtime_environment"]["test_nodeid"] = "another_test"
    elif mutation in {"failed_junit", "extra_junit"}:
        path = bundle.root / "pytest_junit.xml"
        value = path.read_text(encoding="utf-8")
        value = (
            value.replace('failures="0"', 'failures="1"')
            if mutation == "failed_junit"
            else value.replace("</testsuite>", '<testcase classname="other" name="test"/></testsuite>')
        )
        path.write_text(value, encoding="utf-8")
    if mutation != "record_mismatch":
        bundle.manifest["run_lifecycle"]["record"] = deepcopy(bundle.receipt)
    (bundle.root / "run_lifecycle.json").write_text(json.dumps(bundle.receipt), encoding="utf-8")
    bundle.runtime_path.write_text(json.dumps(bundle.witness), encoding="utf-8")
    if mutation != "runtime_hash":
        bundle.readiness["runtime_environment"]["sha256"] = verifier._sha256(bundle.runtime_path)
    (bundle.base / "before.readiness.json").write_text(json.dumps(bundle.readiness), encoding="utf-8")
    if mutation == "missing_lifecycle":
        (bundle.root / "run_lifecycle.json").unlink()
    elif mutation == "missing_runtime":
        bundle.runtime_path.unlink()
    elif mutation == "extra_runtime":
        (bundle.base / "runtime_environment_unreferenced.json").write_text(json.dumps(bundle.witness), encoding="utf-8")
    elif mutation == "missing_png":
        (bundle.base / "before.png").unlink()
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verifier._verify_runtime_environments(bundle.root, bundle.manifest)
