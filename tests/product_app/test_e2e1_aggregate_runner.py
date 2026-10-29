"""Orchestrator unit fixtures only: no server, browser, or provider is started."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_e2e1_aggregate as aggregate
from test_e2e1_runner import _evidence, _manifest, runner


def _junit(path, filename, names, outcome=None):
    suite = ET.Element("testsuite")
    for name in names:
        case = ET.SubElement(suite, "testcase", classname="tests.product_app.browser." + filename.removesuffix(".py"), name=name)
        if outcome:
            ET.SubElement(case, outcome)
    ET.ElementTree(suite).write(path / "pytest_junit.xml", encoding="utf-8")


def test_exact_inventory_includes_twenty_three_suites_and_seventy_two_cases():
    assert len(aggregate.SUITES) == 23
    assert sum(len(names) for _, _, names in aggregate.SUITES) == aggregate.EXPECTED_TEST_COUNT == 72
    paths = {row["path"] for row in aggregate.source_rows()}
    assert {"scripts/run_e2e1_aggregate.py", aggregate.REGISTRY} <= paths
    assert all(f"tests/product_app/browser/{filename}" in paths for _, filename, _ in aggregate.SUITES)


@pytest.mark.parametrize("fault", [None, "skipped", "failure", "error", "duplicate", "wrong_name"])
def test_junit_inventory_rejects_count_only_or_skipped_passes(tmp_path, fault):
    _, filename, names = aggregate.SUITES[0]
    actual = list(names)
    if fault == "duplicate":
        actual[-1] = actual[0]
    elif fault == "wrong_name":
        actual[-1] = "plausible_but_unexecuted_test"
    _junit(tmp_path, filename, actual, fault if fault in {"skipped", "failure", "error"} else None)
    if fault:
        with pytest.raises(aggregate.E2E1RunnerError):
            aggregate.test_inventory(tmp_path, filename, names)
    else:
        assert len(aggregate.test_inventory(tmp_path, filename, names)) == 7


@pytest.mark.parametrize("flag", ["live_copilot_chat", "confirm_billed_synthetic_context"])
def test_aggregate_does_not_dispatch_billed_calls(monkeypatch, flag):
    args = dict(browser_test="aggregate", output_dir=None, port=8127,
                live_copilot_chat=False, confirm_billed_synthetic_context=False)
    args[flag] = True
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(**args))
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_aggregate_is_deterministic_only"):
        runner.main()


def test_all_unit_fixture_passes_do_not_hide_missing_runtime_coverage(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    isolation = _manifest(_evidence(seed))["provider_isolation"]
    observed = []
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-key")
    monkeypatch.setattr(aggregate, "_port_is_available", lambda port: True)
    monkeypatch.setattr(aggregate, "verify_child", lambda path: {"unit_fixture_only": True})

    def run_fixture(command, **kwargs):
        observed.append((command, kwargs["env"]))
        mode = command[command.index("--browser-test") + 1]
        child = aggregate.Path(command[command.index("--output-dir") + 1])
        child.mkdir(parents=True)
        (_, filename, names), = [suite for suite in aggregate.SUITES if suite[0] == mode]
        _junit(child, filename, names)
        (child / "file_io_manifest.json").write_text(json.dumps({"provider_isolation": isolation}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="explicit mocked orchestration unit fixture", stderr="")

    monkeypatch.setattr(aggregate.subprocess, "run", run_fixture)
    output = tmp_path / "aggregate"
    assert aggregate.run_aggregate(output_dir=output, port=8127) == 1
    manifest = json.loads((output / aggregate.MANIFEST).read_text(encoding="utf-8"))
    coverage = json.loads((output / "interaction_coverage.json").read_text(encoding="utf-8"))
    assert manifest["passed_test_count"] == 72
    assert manifest["assertions_passed"] is False
    assert coverage["passed"] is False
    assert len(observed) == 23
    assert all(env["PRODUCT_COPILOT_ADAPTER"] == "deterministic" and env["OPENAI_API_KEY"] == "" for _, env in observed)
    assert len({command[command.index("--output-dir") + 1] for command, _ in observed}) == 23


def test_unverified_cleanup_stops_later_suites(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(aggregate, "_port_is_available", lambda port: True)
    monkeypatch.setattr(aggregate.subprocess, "run", lambda *args, **kwargs:
                        (calls.append(args), SimpleNamespace(returncode=1, stdout="failed unit fixture", stderr=""))[1])
    output = tmp_path / "stopped"
    assert aggregate.run_aggregate(output_dir=output, port=8127) == 1
    result = json.loads((output / aggregate.MANIFEST).read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert [row["status"] for row in result["suites"]] == ["failed"] + ["not_run"] * (len(aggregate.SUITES) - 1)


def test_enabled_unknown_operation_and_missing_accepted_context_fail_machine_gate():
    snapshots = [{"context": "original", "plan_id": "parent", "plan_content_hash": "hash", "artifact": "real_http.json",
        "capabilities": {"draft_capabilities": {"operations": {"unmapped_button": {"enabled": True, "preview_executable": True}}}}}]
    registry = {"unverified_requirements": [], "required_contexts": ["original", "accepted_child"], "rows": []}
    result = aggregate.coverage_report(snapshots, [], registry)
    assert result["passed"] is False
    assert {row["code"] for row in result["findings"]} == {
        "runtime_capability_context_missing", "enabled_interaction_without_complete_runtime_row"}


def test_disabled_operation_is_not_misreported_as_enabled_missing_coverage():
    snapshots = [{"context": "original", "plan_id": "parent", "plan_content_hash": "hash", "artifact": "real_http.json",
        "capabilities": {"draft_capabilities": {"operations": {"unsupported": {"enabled": False}}}}}]
    result = aggregate.coverage_report(snapshots, [], {"unverified_requirements": [], "required_contexts": ["original"], "rows": []})
    assert result["enabled_interactions"] == []


def test_http_capture_exact_and_disclosed_redaction_contract():
    import hashlib
    from copy import deepcopy

    from verify_e2e1_vertical_slice import E2E1VerificationError, _verify_product_http_capture

    body = '{"mutation_token":"OMITTED_SECRET"}'
    row = {"schema_version":"e2e1-product-http-io-v1", "body_capture":"credential_redacted",
        "request_headers_persisted":False, "response_headers_persisted":False,
        "authorization_persisted":False, "session_token_persisted":False,
        "credential_redactions":{"request":[], "response":["/mutation_token"]},
        "request_body_utf8":"", "response_body_utf8":body,
        "request_body_sha256":hashlib.sha256(b"").hexdigest(),
        "request_original_body_sha256":hashlib.sha256(b"").hexdigest(),
        "response_body_sha256":hashlib.sha256(body.encode()).hexdigest(),
        "response_original_body_sha256":"0"*64}
    _verify_product_http_capture(row)
    for key, value in [("response_headers_persisted",True), ("body_capture","exact_utf8"), ("response_original_body_sha256","")]:
        tampered = deepcopy(row)
        tampered[key] = value
        with pytest.raises(E2E1VerificationError):
            _verify_product_http_capture(tampered)


@pytest.mark.parametrize("mode,expected", [("conversation", 14), ("combined-conflicts", 2), ("owned-edits", 11), ("build-coherence", 4), ("optional-evidence-fault", 2), ("preview-geography-fault", 4), ("queued-preview", 4)])
def test_self_owned_suites_verify_every_lifecycle_without_outer_server(tmp_path, monkeypatch, mode, expected):
    import verify_e2e1_self_managed as self_managed

    (tmp_path / "test_process.json").write_text(json.dumps({"unit_fixture": True}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(self_managed, "_lifecycle", lambda base, run_id, process, **kwargs:
        (calls.append((base, run_id, process, kwargs)), {"run_id": run_id})[1])
    manifest = {"schema_version": "e2e1-self-managed-browser-manifest-v1", "mode": mode, "run_id": "unit"}
    assert len(aggregate.child_lifecycles(tmp_path, manifest)) == len(calls) == expected
    assert len({row[1] for row in calls}) == expected
    if mode == "conversation":
        assert sum(row[3].get("fake", False) for row in calls) == 2
        assert {row[1] for row in calls if row[3].get('fake',False)} == {
            'unit_fake_sdk_cancel', 'unit_fake_sdk_cancel_accepted_child'}
        assert calls[-1][1] == "unit_fake_sdk_cancel_accepted_child"
    elif mode in {"combined-conflicts", "owned-edits", "queued-preview"}:
        assert all(not row[3].get("fake", False) and not row[3].get("child_process", False) for row in calls)
    else:
        assert all(row[3]["child_process"] is True for row in calls)


def test_self_owned_cleanup_rejects_a_missing_process_witness(tmp_path):
    manifest = {"schema_version": "e2e1-self-managed-browser-manifest-v1", "mode": "conversation", "run_id": "unit"}
    with pytest.raises(OSError):
        aggregate.child_lifecycles(tmp_path, manifest)


def _passing_finalization_fixture(tmp_path, monkeypatch):
    """Mock orchestration only; the aggregate verifier itself remains real."""
    output = tmp_path / "aggregate"
    sources = [{"path": "unit-source", "size_bytes": 1, "sha256": "a" * 64}]
    receipts, inputs = [], []
    monkeypatch.setattr(aggregate, "SUITES", (("unit", "test_unit.py", ("test_observed",)),))
    monkeypatch.setattr(aggregate, "EXPECTED_TEST_COUNT", 1)
    monkeypatch.setattr(aggregate, "source_rows", lambda: sources)
    monkeypatch.setattr(aggregate, "_port_is_available", lambda port: True)
    monkeypatch.setattr(aggregate, "child_lifecycles", lambda *args: receipts)
    monkeypatch.setattr(aggregate, "verify_child", lambda path: {"unit_fixture_only": True})
    monkeypatch.setattr(aggregate, "capability_snapshots", lambda *args: [])
    monkeypatch.setattr(aggregate, "coverage_report", lambda *args: {
        "passed": True, "findings": [], "external_gates": [], "unit_fixture_only": True})

    def simulated_child(command, **kwargs):
        child = Path(command[command.index("--output-dir") + 1])
        child.mkdir(parents=True)
        receipts.append({"installed_at": aggregate._now(), "finalized_at": aggregate._now()})
        manifest = {"source_inputs": sources, "generated_at": aggregate._now(), "unit_fixture_only": True}
        aggregate._write(child / "file_io_manifest.json", manifest)
        _junit(child, "test_unit.py", ("test_observed",))
        inputs.append({"command": command, "child_manifest": manifest, "receipts": list(receipts),
            "adapter": kwargs["env"]["PRODUCT_COPILOT_ADAPTER"],
            "credential_disabled": kwargs["env"]["OPENAI_API_KEY"] == "", "provider_calls": 0})
        aggregate._write(tmp_path / "exact_unit_input.json", inputs)
        return SimpleNamespace(returncode=0, stdout="explicit orchestration unit fixture; no browser", stderr="")

    monkeypatch.setattr(aggregate.subprocess, "run", simulated_child)
    return output, receipts


@pytest.mark.parametrize("fault", ["occupied_port", "empty_lifecycle"])
def test_last_suite_cannot_pass_without_verified_cleanup(tmp_path, monkeypatch, fault):
    output, _ = _passing_finalization_fixture(tmp_path, monkeypatch)
    if fault == "occupied_port":
        ports = iter((True, False))
        monkeypatch.setattr(aggregate, "_port_is_available", lambda port: next(ports))
    else:
        monkeypatch.setattr(aggregate, "child_lifecycles", lambda *args: [])
    result = aggregate.run_aggregate(output_dir=output)
    manifest = json.loads((output / aggregate.MANIFEST).read_text(encoding="utf-8"))
    aggregate._write(tmp_path / "exact_unit_output.json", {"exit_code": result, "manifest": manifest})
    assert result == 1
    assert manifest["assertions_passed"] is False
    assert manifest["suites"][0]["status"] == "failed"
    assert manifest["suites"][0]["cleanup_verified"] is False
    assert manifest["suites"][0]["code"] == "e2e1_aggregate_cleanup_unverified"
    assert manifest["passed_test_count"] == 0


def test_success_invokes_full_aggregate_verifier_before_returning(tmp_path, monkeypatch):
    output, _ = _passing_finalization_fixture(tmp_path, monkeypatch)
    original = aggregate.verify_aggregate
    calls = []

    def verify(directory):
        calls.append(original(directory))
        return calls[-1]

    monkeypatch.setattr(aggregate, "verify_aggregate", verify)
    result = aggregate.run_aggregate(output_dir=output)
    aggregate._write(tmp_path / "exact_unit_output.json", {"exit_code": result, "verification_calls": calls})
    assert result == 0
    assert len(calls) == 1 and calls[0]["verified"] is True
    assert original(output) == calls[0]
    assert not (output / "aggregate_verification_failure.json").exists()


@pytest.mark.parametrize("fault", ["semantic", "artifact_tamper", "source_drift", "malformed"])
def test_final_verifier_failure_retains_and_binds_prior_evidence(tmp_path, monkeypatch, fault):
    output, _ = _passing_finalization_fixture(tmp_path, monkeypatch)
    original = aggregate.verify_aggregate
    observed = {}

    def verify(directory):
        observed["manifest"] = (directory / aggregate.MANIFEST).read_bytes()
        observed["report"] = (directory / aggregate.REPORT).read_bytes()
        observed["child"] = (directory / "suites/unit/file_io_manifest.json").read_bytes()
        if fault == "semantic":
            raise aggregate.E2E1VerificationError("unit_required_semantic_failure")
        if fault == "malformed":
            raise ValueError("unit malformed verifier evidence")
        if fault == "artifact_tamper":
            observed["child"] += b"\n"
            (directory / "suites/unit/file_io_manifest.json").write_bytes(observed["child"])
        else:
            monkeypatch.setattr(aggregate, "source_rows", lambda: [{
                "path": "unit-source", "size_bytes": 1, "sha256": "b" * 64}])
        return original(directory)

    monkeypatch.setattr(aggregate, "verify_aggregate", verify)
    result = aggregate.run_aggregate(output_dir=output)
    manifest = json.loads((output / aggregate.MANIFEST).read_text(encoding="utf-8"))
    aggregate._write(tmp_path / "exact_unit_output.json", {"exit_code": result, "manifest": manifest,
        "observed_before_verification": {key: value.decode() for key, value in observed.items()}})
    assert result == 1
    assert manifest["assertions_passed"] is False
    assert manifest["suites"][0]["status"] == "passed" and manifest["passed_test_count"] == 1
    assert manifest["source_unchanged"] is (fault != "source_drift")
    assert (output / "aggregate_pre_verification_manifest.json").read_bytes() == observed["manifest"]
    assert (output / "aggregate_pre_verification_report.md").read_bytes() == observed["report"]
    assert (output / "suites/unit/file_io_manifest.json").read_bytes() == observed["child"]
    failure = json.loads((output / "aggregate_verification_failure.json").read_text(encoding="utf-8"))
    assert failure["assertions_passed"] is False
    assert failure["code"] == {"semantic": "unit_required_semantic_failure",
        "artifact_tamper": "e2e1_aggregate_artifact_binding_mismatch",
        "source_drift": "e2e1_aggregate_source_changed", "malformed": "unit malformed verifier evidence"}[fault]
    assert failure["candidate_manifest_sha256"] == aggregate._sha256(output / "aggregate_pre_verification_manifest.json")
    assert failure["candidate_report_sha256"] == aggregate._sha256(output / "aggregate_pre_verification_report.md")
    assert manifest["files"] == aggregate._files(output)
    assert {"aggregate_pre_verification_manifest.json", "aggregate_pre_verification_report.md",
        "aggregate_verification_failure.json"} <= {row["path"] for row in manifest["files"]}
    seed = dict(manifest)
    assert seed.pop("content_hash") == aggregate._canonical_sha256(seed)
    assert (output / aggregate.REPORT).read_text(encoding="utf-8") == aggregate.render_report(manifest)
