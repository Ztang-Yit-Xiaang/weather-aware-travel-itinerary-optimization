from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_e2e1_vertical_slice.py"
SPEC = importlib.util.spec_from_file_location("run_e2e1_vertical_slice", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _evidence(tmp_path: Path) -> Path:
    (tmp_path / "provider_isolation.json").write_text(json.dumps({
        "schema_version": "e2e1-provider-isolation-v1", "run_id": "unit-fixture", "process_id": 123,
        "scope": "application_server",
        "guard_boundaries": ["httpx.Client.send", "httpx.AsyncClient.send", "socket.connect"],
        "installed_at": "2026-09-07T01:00:00+00:00", "finalized_at": "2026-09-07T01:01:00+00:00",
        "finalized": True, "effective_adapter": "deterministic",
        "actual_adapter": "DeterministicCopilotAdapter", "api_key_disabled": True,
        "app_initialized": True, "server_started": True, "server_stopped": True,
        "blocked_attempts": [], "sent_provider_calls": 0, "allowed_local_connections": 1,
        "failure_code": None,
    }), encoding="utf-8")
    for _, scenario in runner.SCENARIOS:
        directory = tmp_path / scenario
        directory.mkdir()
        record = {
            "provider_call_count": 0,
            "llm_input": None,
            "llm_output": None,
            "reason": "deterministic_e2e_no_provider_call",
        }
        if scenario == "unsupported_removal":
            record["verification_test"] = (
                "tests/product_app/test_openai_copilot.py::"
                "test_removal_language_is_rejected_before_provider_call"
            )
        (directory / "llm_io.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
        (directory / "preview_request.json").write_text("{}", encoding="utf-8")
        (directory / "preview_response.json").write_text("{}", encoding="utf-8")
        if scenario == "registered_replacement":
            (directory / "one_click_llm_io.json").write_text(
                json.dumps(
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "deterministic_e2e_no_provider_call",
                    }
                ),
                encoding="utf-8",
            )
    return tmp_path


def _manifest(output: Path) -> dict:
    return runner.build_manifest(
        output,
        command=["python", "-m", "pytest"],
        pytest_exit_code=0,
        health={
            "build_id": "build_test",
            "components": {
                "map": {"code": "maplibre_ready"},
                "routing": {"code": "runtime_osrm_ready"},
                "openai": {"code": "deterministic_adapter_selected"},
            },
        },
        base_url="http://127.0.0.1:49152",
        run_id="unit-fixture",
        server_pid=123,
    )


def test_manifest_binds_exact_files_sources_and_zero_llm_records(tmp_path: Path) -> None:
    output = _evidence(tmp_path)
    manifest = _manifest(output)
    runner.verify_manifest(output, manifest)
    assert manifest["assertions_passed"] is True
    assert manifest["base_url"] == "http://127.0.0.1:49152"
    assert manifest["provider_calls"] == 0
    assert len(manifest["llm_records"]) == 7
    report = runner.render_report(manifest)
    assert "## Exact file input and output" in report
    assert "## Exact LLM input and output" in report
    assert '"llm_input": null' in report
    assert '"llm_output": null' in report


def test_runner_binds_product_cors_origin_to_browser_origin() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'environment["PRODUCT_APP_ORIGIN"] = base_url' in source
    assert "env=environment," in source


def test_manifest_rejects_tampered_artifact(tmp_path: Path) -> None:
    output = _evidence(tmp_path)
    manifest = _manifest(output)
    (output / "registered_replacement" / "preview_response.json").write_text(
        '{"tampered":true}', encoding="utf-8"
    )
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_artifact_binding_mismatch"):
        runner.verify_manifest(output, manifest)


def test_manifest_rejects_nonnull_llm_content(tmp_path: Path) -> None:
    output = _evidence(tmp_path)
    record_path = output / "unsupported_removal" / "llm_io.json"
    record_path.write_text(
        json.dumps(
            {
                "provider_call_count": 0,
                "llm_input": {"forged": True},
                "llm_output": None,
                "reason": "forged content",
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(output)
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_unexpected_llm_io"):
        runner.verify_manifest(output, manifest)


def test_failed_manifest_preserves_the_primary_pytest_failure(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    output.mkdir()
    (output / "pytest_output.txt").write_text("browser assertion failed", encoding="utf-8")
    manifest = runner.build_manifest(
        output,
        command=["python", "-m", "pytest"],
        pytest_exit_code=3,
        health={},
        base_url="http://127.0.0.1:49153",
    )

    runner.verify_manifest(output, manifest, require_complete=False)

    assert manifest["assertions_passed"] is False
    assert manifest["pytest_exit_code"] == 3
    assert manifest["provider_calls"] is None
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_provider_isolation_unverified"):
        runner.verify_manifest(output, manifest)


def test_runner_refuses_an_occupied_explicit_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_port_is_available", lambda port: False)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output_dir": None,
                "port": 8127,
                "live_copilot_chat": False,
                "confirm_billed_synthetic_context": False,
            },
        )(),
    )
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_port_8127_occupied"):
        runner.main()


@pytest.mark.parametrize("change", [
    {"blocked_attempts": [{"sequence": 1, "boundary": "httpx.Client.send",
                            "reason": "non_loopback_target", "at": "2026-09-07T01:00:30+00:00"}]},
    {"finalized": False},
    {"server_stopped": False},
    {"sent_provider_calls": False},
    {"allowed_local_connections": True},
    {"process_id": 124},
    {"actual_adapter": "OpenAICopilotAdapter"},
])
def test_incomplete_or_blocked_isolation_never_claims_verified_zero(tmp_path: Path, change: dict) -> None:
    output = _evidence(tmp_path)
    path = output / "provider_isolation.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(change)
    path.write_text(json.dumps(record), encoding="utf-8")
    manifest = _manifest(output)
    assert manifest["assertions_passed"] is False
    assert manifest["provider_calls"] is None
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_provider_isolation_unverified"):
        runner.verify_manifest(output, manifest)


def test_deterministic_runner_neutralizes_inherited_openai_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    observed = {}
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key-not-a-credential")
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner, "_source_rows", lambda: [])
    monkeypatch.setattr(runner, "parse_args", lambda: type("Args", (), {
        "output_dir": tmp_path / "guarded-run", "port": 8127,
        "live_copilot_chat": False, "confirm_billed_synthetic_context": False,
    })())

    def inspect_spawn(*args, **kwargs):
        observed.update(kwargs["env"])
        raise runner.E2E1RunnerError("synthetic_stop_before_subprocess")

    monkeypatch.setattr(runner.subprocess, "Popen", inspect_spawn)
    try:
        runner.main()
    except runner.E2E1RunnerError as error:
        assert str(error) == "synthetic_stop_before_subprocess"
    assert observed["PRODUCT_COPILOT_ADAPTER"] == "deterministic"
    assert observed["OPENAI_API_KEY"] == ""
    assert runner.os.environ["PRODUCT_COPILOT_ADAPTER"] == "openai"
    assert runner.os.environ["OPENAI_API_KEY"] == "synthetic-test-key-not-a-credential"


def test_readiness_rejects_openai_before_running_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_health", lambda url: {
        "ready": True, "components": {
            "map": {"status": "ready"}, "routing": {"status": "ready"},
            "openai": {"status": "ready", "code": "openai_configured"},
        },
    })
    process = type("Process", (), {"poll": lambda self: None})()
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_provider_isolation_preflight_failed"):
        runner._wait_until_ready(process, "http://127.0.0.1:8127")


@pytest.mark.parametrize("flag", ["live_copilot_chat", "confirm_billed_synthetic_context"])
def test_mobile_journey_cannot_dispatch_billed_provider(monkeypatch, flag):
    values = {"browser_test":"mobile-chat","output_dir":None,"port":8127,
              "live_copilot_chat":False,"confirm_billed_synthetic_context":False}
    values[flag] = True
    monkeypatch.setattr(runner,"parse_args",lambda:type("Args",(),values)())
    with pytest.raises(runner.E2E1RunnerError,match="e2e1_edit_journeys_are_deterministic_only"):
        runner.main()


def test_mobile_journey_binds_real_browser_and_shared_capture_source():
    rows = runner._source_rows(runner.MOBILE_CHAT_BROWSER_TEST)
    paths = {row["path"] for row in rows}
    assert runner.MOBILE_CHAT_BROWSER_TEST in paths
    assert "tests/product_app/browser/test_e2e1_live_copilot_chat.py" in paths
    assert "scripts/e2e1_runtime_environment.py" in paths


@pytest.mark.parametrize("components", [None, [], {"openai": None}, {"openai": []}])
def test_malformed_health_fails_with_stable_preflight_reason(monkeypatch, components) -> None:
    monkeypatch.setattr(runner, "_health", lambda url: {"ready": True, "components": components})
    process = type("Process", (), {"poll": lambda self: None})()
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_provider_isolation_preflight_failed"):
        runner._wait_until_ready(process, "http://127.0.0.1:8127")


@pytest.mark.parametrize("write_failure,pytest_exit_code", [(True, 0), (False, 1)])
def test_runner_preserves_unicode_failures_without_reporting_success(monkeypatch, tmp_path, write_failure, pytest_exit_code) -> None:
    output = tmp_path / "failed-write"
    template = tmp_path / "witness-template"
    template.mkdir()
    _evidence(template)
    record = json.loads((template / "provider_isolation.json").read_text(encoding="utf-8"))
    original_popen = runner.subprocess.Popen
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner, "_source_rows", lambda: [])
    monkeypatch.setattr(runner, "parse_args", lambda: type("Args", (), {
        "output_dir": output, "port": 8127,
        "live_copilot_chat": False, "confirm_billed_synthetic_context": False,
    })())

    class Process:
        pid = 123
        returncode = None

        def __init__(self, command, **kwargs):
            record["run_id"] = command[command.index("--run-id") + 1]

        def poll(self):
            return self.returncode

        def communicate(self, command, timeout):
            assert command == "stop\n"
            (output / "provider_isolation.json").write_text(json.dumps(record), encoding="utf-8")
            self.returncode = 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_wait_until_ready", lambda *args: {})
    original_run = runner.subprocess.run
    exact_output = "Selected stop → Pfeiffer; × must stay visible.\n"

    def run_unicode_child(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        child = [runner.sys.executable, "-c",
                 f"import sys; print({exact_output.rstrip()!r}); sys.exit({pytest_exit_code})"]
        with monkeypatch.context() as child_patch:
            child_patch.setattr(runner.subprocess, "Popen", original_popen)
            result = original_run(child, **kwargs)
        assert result.stdout == exact_output
        (tmp_path / "utf8_runner_exact_io.json").write_text(json.dumps({
            "input":{"command":child,"output_encoding":kwargs["encoding"],"PYTHONIOENCODING":kwargs["env"]["PYTHONIOENCODING"]},
            "output":{"stdout":result.stdout,"stderr":result.stderr,"returncode":result.returncode},
            "scope":"real local child process with synthetic failure text; no app or provider call",
        }), encoding="utf-8")
        return result

    monkeypatch.setattr(runner.subprocess, "run", run_unicode_child)
    original_write = Path.write_text

    def fail_pytest_output(path, *args, **kwargs):
        if write_failure and path.name == "pytest_output.txt":
            raise OSError("synthetic output failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_pytest_output)
    assert runner.main() != 0
    manifest = json.loads((output / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["assertions_passed"] is False
    if write_failure:
        failure = json.loads((output / "runner_failure.json").read_text(encoding="utf-8"))
        assert failure == {"code": "e2e1_runner_os_error", "pytest_exit_code": 0, "runner_exit_code": 1}
    else:
        assert manifest["pytest_exit_code"] == 1
        assert (output / "pytest_output.txt").read_text(encoding="utf-8") == exact_output
        assert not (output / "runner_failure.json").exists()


@pytest.mark.parametrize("semantic_failure", [False, True])
def test_runner_success_requires_full_semantic_verification(monkeypatch, tmp_path, capsys, semantic_failure):
    import verify_e2e1_vertical_slice as verifier

    output = tmp_path / "run"
    template = tmp_path / "template"
    template.mkdir()
    _evidence(template)
    record = json.loads((template / "provider_isolation.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner, "_source_rows", lambda: [])
    monkeypatch.setattr(runner, "parse_args", lambda: type("Args", (), {
        "output_dir": output, "port": 8127,
        "live_copilot_chat": False, "confirm_billed_synthetic_context": False,
    })())

    class Process:
        pid = 123
        returncode = None

        def __init__(self, command, **kwargs):
            record["run_id"] = command[command.index("--run-id") + 1]

        def poll(self):
            return self.returncode

        def communicate(self, command, timeout):
            assert command == "stop\n"
            (output / "provider_isolation.json").write_text(json.dumps(record), encoding="utf-8")
            self.returncode = 0

    def synthetic_browser(*args, **kwargs):
        shutil.copytree(template, output, dirs_exist_ok=True)
        return type("Result", (), {"returncode": 0, "stdout": "synthetic browser unit fixture", "stderr": ""})()

    calls = []

    def semantic_check(path):
        calls.append(path)
        candidate = json.loads((path / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert candidate["assertions_passed"] is True
        assert candidate["pytest_exit_code"] == 0
        if semantic_failure:
            raise verifier.E2E1VerificationError("e2e1_waypoint_canonical_geometry_mismatch")
        return {"verified": True}

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_wait_until_ready", lambda *args: {})
    monkeypatch.setattr(runner.subprocess, "run", synthetic_browser)
    monkeypatch.setattr(verifier, "verify", semantic_check)
    assert runner.main() == int(semantic_failure)
    assert calls == [output]
    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert printed["verified"] is (not semantic_failure)
    assert manifest["assertions_passed"] is (not semantic_failure)
    runner.verify_manifest(output, manifest, require_complete=not semantic_failure)
    failure_path = output / "semantic_verification_failure.json"
    assert failure_path.exists() is semantic_failure
    if semantic_failure:
        assert json.loads(failure_path.read_text(encoding="utf-8")) == {
            "assertions_passed": False, "error_type": "E2E1VerificationError",
            "code": "e2e1_waypoint_canonical_geometry_mismatch", "pytest_exit_code": 0,
            "runner_exit_code": 1,
        }
        assert "semantic_verification_failure.json" in {row["path"] for row in manifest["files"]}
