"""Runner-selection and fail-closed checks; no browser or server is started."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_e2e1_runner import _evidence, runner

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as verifier  # noqa: E402


def test_only_repository_known_recovery_suite_can_be_selected(monkeypatch) -> None:
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--browser-test", "map-recovery"])
    assert runner.parse_args().browser_test == "map-recovery"
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--browser-test", "../arbitrary.py"])
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_default_suite_remains_the_original_vertical_slice(monkeypatch) -> None:
    monkeypatch.setattr(runner.sys, "argv", ["runner"])
    assert runner.parse_args().browser_test == "vertical-slice"
    assert len(runner.SCENARIOS) == 6  # Includes seven journeys: two share registered_replacement.
    assert runner.BROWSER_TEST.endswith("test_e2e1_vertical_slice_live.py")


@pytest.mark.parametrize("flag", ["live_copilot_chat", "confirm_billed_synthetic_context"])
def test_recovery_mode_cannot_dispatch_the_billed_runner(monkeypatch, flag: str) -> None:
    arguments = {
        "browser_test": "map-recovery",
        "output_dir": None,
        "port": 8127,
        "live_copilot_chat": False,
        "confirm_billed_synthetic_context": False,
    }
    arguments[flag] = True
    monkeypatch.setattr(runner, "parse_args", lambda: type("Args", (), arguments)())
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_recovery_is_deterministic_only"):
        runner.main()


def test_recovery_source_universe_includes_imported_capture_helpers_and_provider_guard() -> None:
    paths = [row["path"] for row in runner._source_rows(runner.MAP_RECOVERY_BROWSER_TEST)]
    assert runner.MAP_RECOVERY_BROWSER_TEST in paths
    assert "tests/product_app/browser/test_e2e1_live_copilot_chat.py" in paths
    assert "scripts/e2e1_provider_isolation.py" in paths
    assert "scripts/verify_e2e1_vertical_slice.py" in paths
    assert len(paths) == len(set(paths))


def test_source_change_during_run_never_produces_passing_manifest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _evidence(output)
    source = tmp_path / "app.js"
    source.write_text("before", encoding="utf-8")
    inputs = [{"path": "app.js", "size_bytes": 6, "sha256": hashlib.sha256(b"before").hexdigest()}]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source.write_text("after!", encoding="utf-8")
    manifest = runner.build_manifest(
        output,
        command=["python", "-m", "pytest"],
        pytest_exit_code=0,
        health={},
        base_url="http://127.0.0.1:8127",
        source_inputs=inputs,
        run_id="unit-fixture",
        server_pid=123,
        scenarios=("map_initialization_recovery",),
    )
    assert manifest["assertions_passed"] is False
    assert manifest["source_inputs"] == inputs
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_source_binding_mismatch"):
        runner.verify_manifest(output, manifest, require_complete=False)


def test_recovery_spawn_is_disposable_and_deterministic(monkeypatch, tmp_path: Path) -> None:
    observed = {}
    output = tmp_path / "focused"
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "browser_test": "map-recovery",
                "output_dir": output,
                "port": 8127,
                "live_copilot_chat": False,
                "confirm_billed_synthetic_context": False,
            },
        )(),
    )
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner, "_source_rows", lambda *args: [])
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-credential")

    def stop_before_spawn(command, **kwargs):
        observed.update(command=command, environment=kwargs["env"])
        raise runner.E2E1RunnerError("test_stopped_before_process_creation")

    monkeypatch.setattr(runner.subprocess, "Popen", stop_before_spawn)
    assert runner.main() != 0
    environment = observed["environment"]
    assert environment["PRODUCT_COPILOT_ADAPTER"] == "deterministic"
    assert environment["OPENAI_API_KEY"] == ""
    assert environment["PRODUCT_E2E_MAP_RECOVERY_DISPOSABLE"] == "1"
    assert Path(environment["PRODUCT_E2E_MAP_RECOVERY_EVIDENCE_DIR"]) == output / "map_initialization_recovery"
    assert "scripts/e2e1_provider_isolation.py" in observed["command"]
    manifest = json.loads((output / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["assertions_passed"] is False
    assert manifest["scenarios"] == ["map_initialization_recovery"]
    assert runner.MAP_RECOVERY_BROWSER_TEST in manifest["command"]
    assert runner.BROWSER_TEST not in manifest["command"]


@pytest.mark.parametrize("missing", ["original", "accepted_child", "tile_coverage"])
def test_recovery_requires_both_plan_contexts_and_tile_case(tmp_path, missing):
    for context in {"original", "accepted_child", "tile_coverage"} - {missing}:
        (tmp_path / "map_initialization_recovery" / context).mkdir(parents=True)
    with pytest.raises(verifier.E2E1VerificationError, match="recovery_context_universe_invalid"):
        verifier._verify_map_recovery(tmp_path, "http://127.0.0.1:8127")


def _retry_guard():
    # Execute only the actual Retry receipt guards, never full bundle validation.
    function = ast.parse(inspect.getsource(verifier._verify_map_recovery_case)).body[0]
    first = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "retry_inputs"
    )
    last = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and any(
            isinstance(value, ast.Constant) and value.value == "e2e1_recovery_retry_input_unbound"
            for value in node.value.args
        )
    )
    return compile(
        ast.Module(body=function.body[first : last + 1], type_ignores=[]), str(Path(verifier.__file__)), "exec"
    )


def _retry_fixture(tmp_path):
    # Small synthetic unit contract: no browser events, process, or provider claimed.
    body = '{"status":"ready"}'
    response = {
        "schema_version": "e2e1-product-http-io-v1",
        "url": "http://127.0.0.1:8127/api/map/config",
        "method": "GET",
        "response_status_code": 200,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
        "credential_redactions": {"request": [], "response": []},
        "body_capture": "exact_utf8",
        "request_body_utf8": "",
        "response_body_utf8": body,
    }
    for name in ("request", "response"):
        digest = hashlib.sha256(response[name + "_body_utf8"].encode()).hexdigest()
        response[name + "_body_sha256"] = response[name + "_original_body_sha256"] = digest
    inputs = [
        {
            "trusted": True,
            "selector": "#retry-map-runtime",
            "map_ready": "false",
            "map_stage": {"activePlanId": "unit-plan", "activePlanContentHash": "unit-hash"},
            "clicked_at": "2026-09-08T12:00:02Z",
        }
    ]
    record = {**response, "phase": "explicit_retry_after_fault_removed", "observed_at": "2026-09-08T12:00:03Z"}
    return (
        {
            "base": tmp_path,
            "base_url": "http://127.0.0.1:8127",
            "child_id": "unit-plan",
            "child_hash": "unit-hash",
            "records": [record],
            "blocked": {"observed_at": "2026-09-08T12:00:01Z"},
        },
        inputs,
        response,
    )


def _run_retry_guard(namespace, inputs, response):
    base = namespace["base"]
    for name, value in (
        ("retry_control_input.json", inputs),
        ("retry_response.json", response),
        ("recovered_after_explicit_retry.readiness.json", {"screenshot_started_at": "2026-09-08T12:00:04Z"}),
    ):
        (base / name).write_text(json.dumps(value), encoding="utf-8")
    exec(_retry_guard(), {**vars(verifier), **namespace})


def test_retry_receipt_guard_accepts_consistent_synthetic_unit_contract(tmp_path):
    namespace, inputs, response = _retry_fixture(tmp_path)
    _run_retry_guard(namespace, inputs, response)


@pytest.mark.parametrize(
    "fault",
    [
        "untrusted",
        "wrong_control",
        "wrong_plan",
        "wrong_hash",
        "old_ack",
        "false_ready",
        "early_click",
        "late_response",
        "duplicate_click",
        "duplicate_get",
        "failed_real_response",
    ],
)
def test_retry_receipt_counterexamples_reject(tmp_path, fault):
    namespace, inputs, response = _retry_fixture(tmp_path)
    if fault == "untrusted":
        inputs[0]["trusted"] = False
    elif fault == "wrong_control":
        inputs[0]["selector"] = "#other-button"
    elif fault == "wrong_plan":
        inputs[0]["map_stage"]["activePlanId"] = "foreign"
    elif fault == "wrong_hash":
        inputs[0]["map_stage"]["activePlanContentHash"] = "foreign"
    elif fault == "old_ack":
        inputs[0]["map_stage"]["renderedMapPlanId"] = "unit-plan"
    elif fault == "false_ready":
        inputs[0]["map_ready"] = "true"
    elif fault == "early_click":
        inputs[0]["clicked_at"] = "2026-09-08T12:00:00Z"
    elif fault == "late_response":
        namespace["records"][0]["observed_at"] = "2026-09-08T12:00:05Z"
    elif fault == "duplicate_click":
        inputs.append(deepcopy(inputs[0]))
    elif fault == "duplicate_get":
        namespace["records"].append(deepcopy(namespace["records"][0]))
    else:
        response["response_body_utf8"] = '{"status":"failed"}'
        response["response_body_sha256"] = response["response_original_body_sha256"] = hashlib.sha256(
            response["response_body_utf8"].encode()
        ).hexdigest()
        namespace["records"][0].update(response)
    with pytest.raises(verifier.E2E1VerificationError, match="recovery_retry_(input|response)_unbound"):
        _run_retry_guard(namespace, inputs, response)
