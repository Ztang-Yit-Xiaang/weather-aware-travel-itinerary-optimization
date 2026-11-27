"""Explicit unit fixtures for actual-browser binding; no browser or server launch."""

import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import test_e2e1_runtime_environment as runtime_fixtures
import verify_e2e1_vertical_slice as verifier
from test_e2e1_runner import runner
from test_e2e1_runtime_environment import environment

captured = runtime_fixtures.captured

SPEC = importlib.util.spec_from_file_location(
    "receipt_browser_helpers", runner.ROOT / "tests/product_app/browser/test_e2e1_live_copilot_chat.py"
)
browser_helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(browser_helpers)


def test_runtime_receipt_uses_the_journey_browser_and_is_not_reprobed_per_screenshot(captured, tmp_path, monkeypatch):
    record = captured.capture()
    monkeypatch.setitem(sys.modules, "e2e1_runtime_environment", environment)
    monkeypatch.setenv("PRODUCT_E2E_RUN_ID", "synthetic_r24")
    monkeypatch.setenv("PRODUCT_E2E_BASE_URL", "http://127.0.0.1:8127")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/product_app/browser/test_unit.py::test_receipt (call)")
    observed = []

    def capture(**kwargs):
        observed.append(kwargs)
        return record

    monkeypatch.setattr(environment, "capture_runtime_environment", capture)
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        assert args == ["docker", "ps", "--filter", "publish=5000", "--filter", "status=running", "--format", "{{.ID}}"]
        return SimpleNamespace(stdout="c" * 64 + "\n")

    monkeypatch.setattr(browser_helpers.subprocess, "run", command)
    actual_browser = SimpleNamespace(version="140.0.1.2", browser_type=SimpleNamespace(name="chromium"))
    fake_page = SimpleNamespace(
        context=SimpleNamespace(browser=actual_browser),
        locator=lambda selector: SimpleNamespace(
            evaluate_all=lambda code: ["http://127.0.0.1:8080/maplibre/maplibre-gl.js"]
        ),
    )
    first = browser_helpers._runtime_receipt(fake_page, tmp_path)
    assert observed[0]["actual_browser"] == record["browser"]
    assert observed[0]["actual_browser"]["source"] == "actual_test_browser"
    assert browser_helpers._runtime_receipt(fake_page, tmp_path) == first
    assert len(observed) == len(commands) == 1
    actual_browser.version = "141.0.1.2"
    with pytest.raises(AssertionError, match="Actual browser differs"):
        browser_helpers._runtime_receipt(fake_page, tmp_path)


def _receipt_bundle(tmp_path, captured):
    record = captured.capture()
    node = "tests/product_app/browser/test_unit.py::test_receipt"
    name = "runtime_environment_" + hashlib.sha256(node.encode()).hexdigest()[:16] + ".json"
    target = tmp_path / name
    target.write_text(json.dumps(record), encoding="utf-8")
    complete = datetime.fromisoformat(record["completed_at"])
    ready = {
        "runtime_environment": {
            "artifact": name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "test_nodeid": node,
            "browser": deepcopy(record["browser"]),
        },
        "screenshot_started_at": (complete + timedelta(milliseconds=1)).isoformat(),
        "screenshot_completed_at": (complete + timedelta(milliseconds=2)).isoformat(),
    }
    (tmp_path / "screen.readiness.json").write_text(json.dumps(ready), encoding="utf-8")
    xml = ET.Element("testsuite")
    ET.SubElement(xml, "testcase", classname="tests.product_app.browser.test_unit", name="test_receipt")
    ET.ElementTree(xml).write(tmp_path / "pytest_junit.xml", encoding="utf-8")
    manifest = {
        "base_url": "http://127.0.0.1:8127",
        "provider_isolation": {
            "record": {
                "run_id": "synthetic_r24",
                "installed_at": (datetime.fromisoformat(record["observed_at"]) - timedelta(seconds=1)).isoformat(),
                "finalized_at": (complete + timedelta(seconds=1)).isoformat(),
            }
        },
    }
    return manifest, ready, target


@pytest.mark.parametrize(
    "fault", [None, "browser", "hash", "wrong_test", "screenshot_before_runtime", "different_run", "missing", "orphan"]
)
def test_runtime_receipt_gate_rejects_disconnected_or_early_evidence(captured, tmp_path, monkeypatch, fault):
    monkeypatch.setitem(sys.modules, "e2e1_runtime_environment", environment)
    manifest, ready, target = _receipt_bundle(tmp_path, captured)
    if fault == "browser":
        ready["runtime_environment"]["browser"]["version"] = "1.2.3"
    elif fault == "hash":
        ready["runtime_environment"]["sha256"] = "0" * 64
    elif fault == "wrong_test":
        ready["runtime_environment"]["test_nodeid"] += "_other"
    elif fault == "screenshot_before_runtime":
        ready["screenshot_started_at"] = "2000-01-01T00:00:00+00:00"
    elif fault == "different_run":
        manifest["provider_isolation"]["record"]["run_id"] = "other_run"
    elif fault == "missing":
        target.unlink()
    elif fault == "orphan":
        (tmp_path / "runtime_environment_orphan.json").write_bytes(target.read_bytes())
    (tmp_path / "screen.readiness.json").write_text(json.dumps(ready), encoding="utf-8")
    if fault:
        with pytest.raises((verifier.E2E1VerificationError, environment.RuntimeEnvironmentError, FileNotFoundError)):
            verifier._verify_runtime_environments(tmp_path, manifest)
    else:
        verifier._verify_runtime_environments(tmp_path, manifest)


@pytest.mark.parametrize(
    "mode,browser_test,flag,scenario",
    [
        ("direct-edits", runner.DIRECT_EDITS_BROWSER_TEST, "PRODUCT_E2E_DIRECT_EDITS_DISPOSABLE", "direct_edits"),
        (
            "addition-continuation",
            runner.ADDITION_BROWSER_TEST,
            "PRODUCT_E2E_ADDITION_CONTINUATION_DISPOSABLE",
            "addition_continuation",
        ),
    ],
)
def test_new_journey_dispatch_requires_disposable_deterministic_state(
    tmp_path, monkeypatch, mode, browser_test, flag, scenario
):
    output = tmp_path / scenario
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(
            browser_test=mode,
            output_dir=output,
            port=8127,
            live_copilot_chat=False,
            confirm_billed_synthetic_context=False,
        ),
    )
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    observed = []

    def stop_before_spawn(command, **kwargs):
        observed.append(kwargs["env"])
        raise runner.E2E1RunnerError("explicit_unit_fixture_stopped_before_process_creation")

    monkeypatch.setattr(runner.subprocess, "Popen", stop_before_spawn)
    assert runner.main() == 1
    assert observed[0][flag] == "1"
    assert observed[0]["OPENAI_API_KEY"] == ""
    assert observed[0]["PRODUCT_COPILOT_ADAPTER"] == "deterministic"
    assert observed[0]["PRODUCT_E2E_RUN_ID"]
    manifest = json.loads((output / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert browser_test in manifest["command"]
    assert manifest["scenarios"] == [scenario]
    assert manifest["assertions_passed"] is False
