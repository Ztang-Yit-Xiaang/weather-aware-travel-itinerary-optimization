"""Non-network fixtures for self-managed launch, lifecycle and privacy gates."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_e2e1_self_managed as runner  # noqa: E402
import run_e2e1_vertical_slice as dispatch  # noqa: E402
import verify_e2e1_self_managed as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs, live_source_inputs  # noqa: E402
from run_e2e1_aggregate import SUITES  # noqa: E402

SETUP_INPUTS = {"tests/conftest.py", "tests/product_app/conftest.py"}


@pytest.mark.parametrize("browser", ["tests/product_app/browser/" + filename for _, filename, _ in SUITES])
def test_every_registered_browser_mode_pins_actual_pytest_setup(browser):
    inputs = deterministic_source_inputs(browser)
    assert SETUP_INPUTS <= set(inputs)
    assert len(inputs) == len(set(inputs))
    rows = {row["path"]: row for row in dispatch._source_rows(browser)}
    for path in SETUP_INPUTS:
        assert rows[path]["sha256"] == hashlib.sha256((dispatch.ROOT / path).read_bytes()).hexdigest()


def test_paid_browser_mode_pins_the_same_actual_pytest_setup():
    from run_e2e1_live_copilot_chat import BROWSER_TEST, SOURCE_INPUTS, _source_rows

    assert SETUP_INPUTS <= set(live_source_inputs(BROWSER_TEST))
    assert len(SOURCE_INPUTS) == len(set(SOURCE_INPUTS))
    rows = {row["path"]: row for row in _source_rows()}
    for path in SETUP_INPUTS:
        assert rows[path]["sha256"] == hashlib.sha256((dispatch.ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("setup", sorted(SETUP_INPUTS))
@pytest.mark.parametrize("fault", ["missing_row", "changed_bytes", "missing_file"])
def test_independent_source_gate_rejects_changed_or_omitted_pytest_setup(tmp_path, monkeypatch, setup, fault):
    rows = dispatch._source_rows()
    assert setup in {row["path"] for row in rows}
    for row in rows:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((dispatch.ROOT / row["path"]).read_bytes())
    monkeypatch.setattr(core, "ROOT", tmp_path)
    generated = datetime.now(UTC) + timedelta(seconds=1)
    manifest = {"source_inputs": deepcopy(rows)}
    core._verify_sources(manifest, generated)
    target = tmp_path / setup
    if fault == "missing_row":
        manifest["source_inputs"] = [row for row in rows if row["path"] != setup]
        code = "source_universe_invalid"
    elif fault == "changed_bytes":
        data = target.read_bytes()
        target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        code = "source_hash_mismatch"
    else:
        target.unlink()
        code = "source_missing"
    with pytest.raises(core.E2E1VerificationError, match=code):
        core._verify_sources(manifest, generated)


def _receipt():
    return {"schema_version": "e2e1-provider-isolation-v1", "run_id": "unit_case", "process_id": 77,
        "scope": "application_server", "guard_boundaries": core.ISOLATION_BOUNDARIES,
        "installed_at": "2026-09-08T12:00:01+00:00", "finalized_at": "2026-09-08T12:00:03+00:00",
        "finalized": True, "api_key_disabled": True, "app_initialized": True, "server_started": True,
        "server_stopped": True, "failure_code": None, "sent_provider_calls": 0, "blocked_attempts": [],
        "allowed_local_connections": 2, "effective_adapter": "deterministic", "actual_adapter": "DeterministicCopilotAdapter"}


def test_context_observation_returns_the_exact_json_artifact_shape(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "browser"))
    from copilot_conversation_fixture import observe_context

    canonical = {"accepted_plan_id": "unit-child", "draft_operations": (),
        "conversation_memory": {"proposed_preferences": (), "pending_clarification": None}}
    session = SimpleNamespace(conversation_id=None)
    observed_dispatch = []

    def context(observed, *, dispatch_records):
        assert observed.conversation_id is None
        observed_dispatch.append(dispatch_records)
        return SimpleNamespace(as_dict=lambda: canonical)

    runtime = SimpleNamespace(service=SimpleNamespace(
        workspace=SimpleNamespace(get=lambda sid: session, _lock=RLock()),
        copilot_context=context))
    observed = observe_context(runtime, "unit-session", tmp_path, "unit")
    saved = json.loads((tmp_path / "context_unit.json").read_text(encoding="utf-8"))
    assert observed == saved
    assert observed["draft_operations"] == []
    assert observed["conversation_memory"]["proposed_preferences"] == []
    assert canonical["draft_operations"] == ()
    assert observed_dispatch == [()]


def _lifecycle(tmp_path, *, fake=False):
    receipt = _receipt()
    lifecycle = {"run_id": "unit_case", "pid": 77, "thread_alive_after_shutdown": False,
        "api_response_mocking": False, "live_provider_call_count": 0, "fake_sdk_attempt_count": 0,
        "external_transport_mocked_only": False}
    if fake:
        receipt.update(schema_version="conversation-browser-fake-sdk-isolation-v1",
            actual_adapter="OpenAICopilotAdapter", effective_adapter="openai",
            credential_mode="synthetic_unusable_fixture_value", external_transport="httpx.MockTransport",
            actual_sdk="openai.AsyncOpenAI", sdk_max_retries=0, live_openai_test=False,
            fake_provider_endpoint="http://127.0.0.1:9/v1/responses")
        lifecycle.update(fake_sdk_attempt_count=1, external_transport_mocked_only=True)
    (tmp_path / "provider_isolation.json").write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "server_lifecycle.json").write_text(json.dumps(lifecycle), encoding="utf-8")
    process = {"pid": 77, "started_at": "2026-09-08T12:00:00+00:00", "completed_at": "2026-09-08T12:00:04+00:00"}
    return receipt, lifecycle, process


@pytest.mark.parametrize("fake", [False, True])
def test_separate_real_server_and_fake_external_http_scopes_validate(tmp_path, fake):
    receipt, _, process = _lifecycle(tmp_path, fake=fake)
    assert verifier._lifecycle(tmp_path, "unit_case", process, fake=fake) == receipt


@pytest.mark.parametrize("fault", ["wrong_pid", "wrong_run", "reused_times", "not_finalized", "blocked_socket",
    "sent_live_call", "server_left_alive", "internal_api_mock", "fake_sdk_retry", "fake_as_live"])
def test_lifecycle_counterexamples_reject_without_browser(tmp_path, fault):
    receipt, lifecycle, process = _lifecycle(tmp_path, fake=fault.startswith("fake_"))
    if fault == "wrong_pid":
        receipt["process_id"] = 99
    elif fault == "wrong_run":
        receipt["run_id"] = "prior_run"
    elif fault == "reused_times":
        receipt["installed_at"] = "2026-09-07T12:00:01+00:00"
    elif fault == "not_finalized":
        receipt["finalized"] = False
    elif fault == "blocked_socket":
        receipt["blocked_attempts"] = [{"boundary": "socket.connect"}]
    elif fault == "sent_live_call":
        receipt["sent_provider_calls"] = 1
    elif fault == "server_left_alive":
        lifecycle["thread_alive_after_shutdown"] = True
    elif fault == "internal_api_mock":
        lifecycle["api_response_mocking"] = True
    elif fault == "fake_sdk_retry":
        receipt["sdk_max_retries"] = 2
    elif fault == "fake_as_live":
        receipt["live_openai_test"] = True
    (tmp_path / "provider_isolation.json").write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "server_lifecycle.json").write_text(json.dumps(lifecycle), encoding="utf-8")
    with pytest.raises(core.E2E1VerificationError):
        verifier._lifecycle(tmp_path, "unit_case", process, fake=fault.startswith("fake_"))


@pytest.mark.parametrize("mode", list(runner.MODES))
def test_dispatch_self_managed_before_any_outer_server_launch(monkeypatch, mode):
    args = SimpleNamespace(browser_test=mode, live_copilot_chat=False,
        confirm_billed_synthetic_context=False, output_dir=Path("unit-output"), port=8127)
    monkeypatch.setattr(dispatch, "parse_args", lambda: args)
    observed = []
    monkeypatch.setattr(runner, "run", lambda **kwargs: observed.append(kwargs) or 42)
    monkeypatch.setattr(dispatch.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No outer server may launch"))
    assert dispatch.main() == 42
    assert observed == [{"mode": mode, "output_dir": Path("unit-output"), "port": 8127}]


@pytest.mark.parametrize("flag", ["live_copilot_chat", "confirm_billed_synthetic_context"])
@pytest.mark.parametrize("mode", list(runner.MODES))
def test_self_managed_modes_never_consume_billed_authorization(monkeypatch, flag, mode):
    args = SimpleNamespace(browser_test=mode, live_copilot_chat=False,
        confirm_billed_synthetic_context=False, output_dir=None, port=8127)
    setattr(args, flag, True)
    monkeypatch.setattr(dispatch, "parse_args", lambda: args)
    with pytest.raises(dispatch.E2E1RunnerError, match="e2e1_self_managed_is_non_live_only"):
        dispatch.main()


def test_new_modes_bind_all_actual_fixture_and_verifier_inputs():
    expected = {
        "test_copilot_queued_preview_live.py": {"tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py", "scripts/run_e2e1_self_managed.py",
            "scripts/verify_e2e1_self_managed.py"},
        "test_owned_combined_journey_live.py": {"tests/product_app/owned_product_fixture.py",
            "scripts/build_coherent_product_demo.py", "tests/product_app/browser/control_action_fixture.py",
            "tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py", "scripts/run_e2e1_self_managed.py",
            "scripts/verify_e2e1_self_managed.py", "scripts/verify_e2e1_owned_edits.py"},
        "test_copilot_combined_conflicts_live.py": {"tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_copilot_conversation_journeys_live.py",
            "tests/product_app/browser/test_e2e1_live_copilot_chat.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py", "scripts/run_e2e1_self_managed.py",
            "scripts/verify_e2e1_self_managed.py"},
        "test_build_coherence_live.py": {"tests/product_app/browser/build_coherence_fixture.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"},
        "test_copilot_conversation_journeys_live.py": {"tests/product_app/browser/copilot_conversation_fixture.py",
            "tests/product_app/browser/test_selection_conversation_generation_live.py",
            "scripts/verify_e2e1_selection_generation.py",
            "scripts/run_e2e1_self_managed.py", "scripts/verify_e2e1_self_managed.py"},
        "test_waypoint_pointer_live.py": {"tests/product_app/browser/waypoint_ui_fixture.py",
            "scripts/verify_e2e1_waypoint_pointer.py"},
    }
    for filename, helpers in expected.items():
        inputs = deterministic_source_inputs("tests/product_app/browser/" + filename)
        assert helpers | {"tests/product_app/browser/test_e2e1_live_copilot_chat.py"} <= set(inputs)
        assert len(inputs) == len(set(inputs))
        assert all((dispatch.ROOT / path).is_file() for path in inputs)


def test_owned_fixture_output_cannot_leave_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "repository")
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No browser should launch"))
    with pytest.raises(dispatch.E2E1RunnerError, match="e2e1_owned_fixture_requires_repository_output_directory"):
        runner.run(mode="owned-edits", output_dir=tmp_path / "outside")


def test_nested_secret_capture_is_rejected_even_inside_an_exact_string(tmp_path):
    # This is deliberately synthetic unit input, not a real credential.
    (tmp_path / "unit.json").write_text(json.dumps({"body": json.dumps({"mutation_token": "unit-private-token"})}), encoding="utf-8")
    with pytest.raises(core.E2E1VerificationError, match="e2e1_structured_secret_persisted"):
        core._verify_text_security(tmp_path)


def test_completed_conversation_state_redacts_credential_verifier_and_binds_hashes(tmp_path):
    state = tmp_path / "_state" / "case" / "sessions"
    state.mkdir(parents=True)
    session = state / "session_unit.json"
    session.write_text(json.dumps({
        "schema_version": "unit",
        "session": {"session_id": "unit-session"},
        "token_verifier": {"algorithm": "sha256-salted-v1", "digest": "unit", "salt": "unit"},
    }), encoding="utf-8")

    result = runner.redact_conversation_state(tmp_path)

    saved = json.loads(session.read_text(encoding="utf-8"))
    row = result["files"][0]
    assert result["assertions_passed"] is True
    assert result["redacted_file_count"] == 1
    assert saved["token_verifier"] == "OMITTED_SECRET"
    assert row["path"] == "_state/case/sessions/session_unit.json"
    assert row["pre_redaction_sha256"] != row["post_redaction_sha256"]
    assert row["post_redaction_sha256"] == core._sha256(session)
    core._verify_text_security(tmp_path)


def test_declared_redaction_does_not_reveal_a_credential(tmp_path):
    (tmp_path / "unit.json").write_text(json.dumps({"body": json.dumps({"mutation_token": "OMITTED_SECRET"})}), encoding="utf-8")
    core._verify_text_security(tmp_path)


def test_missing_self_managed_bundle_rejects_with_stable_code(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_self_manifest_malformed"):
        verifier.verify(tmp_path)


def test_incomplete_stream_cannot_be_counted_as_a_complete_model_response():
    events = [{"schema_version": "copilot-stream-v1", "type": kind, "sequence": index,
        "request_id": "request_unit", "client_message_id": "message_unit"}
        for index, kind in enumerate(("message.accepted", "assistant.status"), 1)]
    record = {"request_body_utf8": '{"client_message_id":"message_unit"}',
        "response_body_utf8": "".join("data: " + json.dumps(event) + "\n\n" for event in events)}
    assert verifier._stream(record, complete=False)[1] == events
    with pytest.raises(core.E2E1VerificationError, match="e2e1_self_sse_terminal_invalid"):
        verifier._stream(record, complete=True)
    changed = deepcopy(record)
    events[1]["client_message_id"] = "other"
    changed["response_body_utf8"] = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    with pytest.raises(core.E2E1VerificationError, match="e2e1_self_sse_identity_invalid"):
        verifier._stream(changed, complete=False)


def test_private_copy_inventory_and_fingerprints_bind_exact_production_inputs():
    from itinerary_system.product_app.runtime import product_build_id

    paths = verifier._private_input_universe(core.ROOT)
    assert {"scripts/run_product_app.py", "configs/product_app_registry.json", "configs/product_route_delta.json"} <= paths
    rows = [{"path": path} for path in paths]
    registry = (core.ROOT / "configs/product_app_registry.json").read_bytes()
    assert verifier._private_build_id(core.ROOT, rows, registry) == product_build_id(core.ROOT)
    value = json.loads(registry)
    next(row for row in value["runs"] if row["default"])["label"] += " unit alternate generation"
    alternate = json.dumps(value).encode()
    assert verifier._private_build_id(core.ROOT, rows, alternate) != product_build_id(core.ROOT)
    incomplete = [row for row in rows if row["path"] != "scripts/run_product_app.py"]
    with pytest.raises(core.E2E1VerificationError, match="private_build_input_missing"):
        verifier._private_build_id(core.ROOT, incomplete, registry)


@pytest.mark.parametrize("contexts", [("original",), ("accepted_child",), ("original", "original"),
    ("original", "accepted_child", "extra")])
def test_build_junit_rejects_missing_duplicate_or_unregistered_context(tmp_path, contexts):
    rows = ''.join('<testcase classname="tests.product_app.browser.test_build_coherence_live" name="'
        + verifier.BUILD_TEST + '[' + context + ']"/>' for context in contexts)
    (tmp_path / "pytest_junit.xml").write_text('<testsuite>' + rows + '</testsuite>', encoding="utf-8")
    with pytest.raises(core.E2E1VerificationError, match="junit_inventory_invalid"):
        verifier._junit(tmp_path, "build-coherence")


def test_build_verifier_requires_both_actual_context_directories(tmp_path):
    (tmp_path / "build_coherence/accepted_child").mkdir(parents=True)
    with pytest.raises(core.E2E1VerificationError, match="build_context_universe_invalid"):
        verifier._build(tmp_path, "unit", {}, set())


def _conversation_accepted_fixture():
    """Synthetic verifier data only; no server, browser, evaluator or SDK run."""
    candidate = "pfeiffer_big_sur_state_park"
    child = {"plan_id": "child", "parent_plan_id": "parent", "planning_request_id": "request", "sequence": [candidate]}
    child["content_hash"] = core._artifact_content_hash(child)
    certificate = {"certificate_id": "certificate", "plan_id": "child", "plan_content_hash": child["content_hash"],
        "comparison_eligibility": "eligible", "failures": [], "route_validation": {
            "required_leg_count": 1, "road_validated_leg_count": 1, "missing_leg_count": 0, "fallback_leg_count": 0}}
    certificate["content_hash"] = core._artifact_content_hash(certificate)
    diff = {"diff_id": "diff", "parent_plan_id": "parent", "child_plan_id": "child"}
    diff["content_hash"] = core._artifact_content_hash(diff)
    geography = {"plan_id": "child", "content_hash": child["content_hash"],
        "validated_legs": {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"plan_id": "child"},
            "geometry": {"type": "LineString", "coordinates": [[-121, 36], [-121.1, 36.1]]}}]},
        "stops": {"type": "FeatureCollection", "features": [{"properties": {"stop_id": candidate}}]}}
    operation = {"type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint", "source": "map",
        "parameters": {"candidate_id": candidate}, "created_at": "unit"}
    original = {"active_plan_id": "parent", "active_plan_content_hash": "parent-hash", "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "parent-geometry", "rendered_marker_ids": ["bixby_creek_bridge_viewpoint"], "preview_state": "PARENT_RENDERED"}
    accepted_map = {"active_plan_id": "child", "active_plan_content_hash": child["content_hash"], "rendered_map_plan_id": "child",
        "rendered_geometry_hash": core._completed_race_geometry_hash(geography), "rendered_marker_ids": [candidate], "preview_state": "PARENT_RENDERED"}
    proposal = {"state": "eligible", "execution_mode": "deterministic_repair_pipeline", "parent_plan_id": "parent",
        "parent_plan_content_hash": "parent-hash", "child_plan_id": "child", "child_plan_content_hash": child["content_hash"],
        "certificate_id": "certificate", "certificate_content_hash": certificate["content_hash"], "diff_id": "diff",
        "diff_content_hash": diff["content_hash"], "geography_plan": geography,
        "compiled_request": {"operations": [{key: val for key, val in operation.items() if key != "created_at"}]},
        "runtime_execution": {"evaluation_executed": True, "evaluator": "PlanEvaluator.evaluate_final_plan", "child_plan": child,
            "evaluation_certificate": certificate, "plan_diff": diff, "request_id": "request", "repair_request": {"request_id": "request"}}}
    stage = {"operation": operation, "session": {"draft": [operation], "revision": 1, "accepted_plan_id": "parent"}}
    preview = {"proposal": proposal, "session": {"proposal": proposal, "draft": [operation], "revision": 2}}
    decision = {"decision_type": "accept_repair", "decision_id": "decision", "idempotency_key": "key", "previous_plan_id": "parent",
        "selected_plan_id": "child", "selected_plan_content_hash": child["content_hash"], "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"], "diff_id": "diff", "diff_content_hash": diff["content_hash"]}
    pointer = {"schema_version": "unit-pointer", "revision": 1, "accepted_plan_id": "child",
        "accepted_plan_content_hash": child["content_hash"], "original_plan_id": "parent", "decision_id": "decision"}
    for row in (decision, pointer):
        row["content_hash"] = core._canonical_sha256(row)
    accepted = {"result": {"decision": decision, "pointer": pointer}, "session": {"session_id": "session"}}
    restored = {"session": {"session_id": "session", "accepted_plan_id": "child", "draft": [], "proposal": None},
        "workspace": {"accepted_plan_pointer": {key: pointer[key] for key in (
            "schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")},
            "alternative_evidence": {"child": {"certificate": certificate}}}}
    records = []
    for method, suffix, request, response in (("POST", "/draft/operations", operation, stage),
        ("POST", "/preview", {"expected_revision": 1}, preview),
        ("POST", "/accept", {"confirmation": "accept_evaluated_repair", "expected_revision": 2, "idempotency_key": "key"}, accepted),
        ("GET", "", None, restored)):
        records.append({"method": method, "url": "http://127.0.0.1:8127/api/sessions/session" + suffix,
            "response_status_code": 200, "request_body_utf8": json.dumps(request) if request else "", "response_body_utf8": json.dumps(response)})
    trace = {"context": "accepted_child", "initial": deepcopy(accepted_map)}
    value = {"schema_version": "conversation-accepted-baseline-v1", "context": "accepted_child", "assertions_passed": True,
        "chat_request_count": 0, "fake_sdk_attempt_count": 0, "original": original, "accepted_map": accepted_map,
        "preview_map": {**accepted_map, "preview_state": "PREVIEW_RENDERED"}, "http_boundary_files": [f"http_{i}.json" for i in range(4)]}
    context = {"accepted_plan_id": "child", "accepted_plan_content_hash": child["content_hash"], "draft_operations": [],
        "conversation_memory": {"plan_id": "child", "plan_content_hash": child["content_hash"]}}
    files = {"accepted_baseline_artifacts.json": value, "context_accepted_baseline.json": context,
        "context_control_final.json": deepcopy(context), "accepted_setup_clicks.json": [
            {"trusted": True, "id": identity} for identity in ("draft-replace", "inline-preview", "accept-repair")]}
    return files, trace, records


@pytest.mark.parametrize("fault", [None, "no_accept", "duplicate_accept", "extra_stage", "keep_original", "rollback",
    "wrong_boundary", "chat_during_setup", "wrong_context", "stale_map", "changed_final_context", "changed_final_draft",
    "forged_child_hash", "unevaluated", "missing_road", "wrong_parent", "not_confirmed", "stale_reload", "untrusted_accept",
    "repeat_restore_after_boundary"])
def test_accepted_conversation_controls_require_real_bound_setup_and_no_later_mutation(tmp_path, fault):
    files, trace, records = _conversation_accepted_fixture()
    baseline = files["accepted_baseline_artifacts.json"]
    if fault == "no_accept":
        records[2]["url"] += "-not-accept"
    elif fault in {"duplicate_accept", "extra_stage", "keep_original", "rollback"}:
        extra = deepcopy(records[2] if fault == "duplicate_accept" else records[0])
        if fault in {"keep_original", "rollback"}:
            extra["url"] = "http://127.0.0.1:8127/api/sessions/session/" + fault.replace("_", "-")
        records.append(extra)
    elif fault == "wrong_boundary":
        baseline["http_boundary_files"].reverse()
    elif fault == "chat_during_setup":
        records[0]["url"] = "http://127.0.0.1:8127/api/sessions/session/copilot/messages"
    elif fault == "wrong_context":
        trace["context"] = "original"
    elif fault == "stale_map":
        trace["initial"]["rendered_geometry_hash"] = "old"
    elif fault == "changed_final_context":
        files["context_control_final.json"]["accepted_plan_id"] = "parent"
    elif fault == "changed_final_draft":
        files["context_control_final.json"]["draft_operations"] = ["unexpected"]
    elif fault in {"forged_child_hash", "unevaluated", "missing_road", "wrong_parent"}:
        body = json.loads(records[1]["response_body_utf8"])
        proposal = body["proposal"]
        if fault == "forged_child_hash":
            proposal["runtime_execution"]["child_plan"]["content_hash"] = "forged"
        elif fault == "unevaluated":
            proposal["runtime_execution"]["evaluation_executed"] = False
        elif fault == "missing_road":
            proposal["runtime_execution"]["evaluation_certificate"]["route_validation"]["missing_leg_count"] = 1
        else:
            proposal["parent_plan_content_hash"] = "wrong-parent-hash"
        body["session"]["proposal"] = proposal
        records[1]["response_body_utf8"] = json.dumps(body)
    elif fault == "not_confirmed":
        body = json.loads(records[2]["request_body_utf8"])
        body["confirmation"] = "preview_only"
        records[2]["request_body_utf8"] = json.dumps(body)
    elif fault == "stale_reload":
        body = json.loads(records[3]["response_body_utf8"])
        body["session"]["accepted_plan_id"] = "parent"
        records[3]["response_body_utf8"] = json.dumps(body)
    elif fault == "untrusted_accept":
        files["accepted_setup_clicks.json"][-1]["trusted"] = False
    elif fault == "repeat_restore_after_boundary":
        records.append(deepcopy(records[3]))
    for label, row in zip(("stage", "preview", "accept", "restored"), records[:4], strict=True):
        files[f"accepted_setup_{label}_http.json"] = row
    files["http_index.json"] = [{"file": f"http_{i}.json", "method": row["method"], "url": row["url"], "status": 200} for i, row in enumerate(records)]
    for filename, value in files.items():
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    if fault and fault != "repeat_restore_after_boundary":
        with pytest.raises(core.E2E1VerificationError):
            verifier._conversation_accepted_baseline(tmp_path, trace, records)
    else:
        verifier._conversation_accepted_baseline(tmp_path, trace, records)


@pytest.mark.parametrize("record_name,field", [("decision","proposal_id"),("pointer","updated_at")])
@pytest.mark.parametrize("fault", [None,"ascii_escaped_hash","changed_unicode","nonfinite"])
def test_acceptance_record_hashes_use_full_canonical_utf8_not_manifest_encoding(tmp_path,record_name,field,fault):
    from itinerary_system.product_app.acceptance import _canonical_json

    files,trace,records = _conversation_accepted_fixture()
    accepted = json.loads(records[2]["response_body_utf8"])
    row = accepted["result"][record_name]
    # Encoding-only synthetic record values, not a claim these are production timestamps.
    row[field] = "提案・海岸 café 🌲"
    unsigned = {key:value for key,value in row.items() if key!="content_hash"}
    canonical = json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
    assert canonical == _canonical_json(unsigned)
    row["content_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if fault == "ascii_escaped_hash":
        row["content_hash"] = hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    elif fault == "changed_unicode":
        row[field] += "已更改"
    elif fault == "nonfinite":
        row[field] = float("nan")
        row["content_hash"] = core._canonical_sha256({key:value for key,value in row.items() if key!="content_hash"})
    records[2]["response_body_utf8"] = json.dumps(accepted,ensure_ascii=False)
    for label,record in zip(("stage","preview","accept","restored"),records,strict=True):
        files[f"accepted_setup_{label}_http.json"] = record
    files["http_index.json"] = [{"file":f"http_{i}.json","method":record["method"],"url":record["url"],"status":200}
        for i,record in enumerate(records)]
    files["canonical_utf8_fixture_io.json"] = {"scope":"synthetic_encoding_regression_not_runtime",
        "canonical_unsigned_utf8":canonical,"record":record_name,"fault":fault,"input":accepted}
    for filename,value in files.items():
        (tmp_path/filename).write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")
    if fault:
        with pytest.raises((core.E2E1VerificationError,ValueError)):
            verifier._conversation_accepted_baseline(tmp_path,trace,records)
    else:
        verifier._conversation_accepted_baseline(tmp_path,trace,records)


@pytest.mark.parametrize("continued", [False, True])
@pytest.mark.parametrize("fault", [None, "changed_message", "partial_draft", "false_ready", "wrong_question",
    "foreign_owner", "changed_map", "changed_context", "untrusted_click"])
def test_combined_conflict_audit_rejects_forged_success_and_disconnected_state(continued, fault):
    name = "combined_conflicts" + ("_accepted_child" if continued else "")
    identity = {"active_plan_id":"child" if continued else "parent", "active_plan_content_hash":"hash",
        "rendered_map_plan_id":"child" if continued else "parent", "rendered_geometry_hash":"geometry",
        "rendered_marker_ids":["wharf"], "preview_state":"PARENT_RENDERED"}
    messages = [
        "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 120 minutes.",
        "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 90 minutes.",
        ("Set Stearns Wharf role to meal; set Stearns Wharf role to activity." if continued else
         "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Bixby Creek Bridge duration to 90 minutes."),
    ]
    questions = [
        "Different values were requested for the same stop attribute. Which value should I use?",
        "The same edit was requested more than once. Should I keep just one copy?",
        ("Different values were requested for the same stop attribute. Which value should I use?" if continued else
         "A stop cannot be preserved or edited and replaced in the same request. Which change should I keep?"),
    ]
    turns, visible = [], []
    for message, question in zip(messages, questions, strict=True):
        copy = question + " No part was staged or changed."
        session = {"session_id":"owner", "accepted_plan_id":identity["active_plan_id"],
            "revision":7, "draft":[], "proposal":None}
        turn = {"user_message":message, "context_revision":7, "assistant_message":copy,
            "interpretation":{"state":"clarification_required", "intents":[{
                "type":"request_clarification", "clarification_question":question}]}}
        turns.append({"input":{"message":message, "expected_revision":7}, "output":{"payload":{
            "session":session, "turn":turn}}})
        visible.append({"user_message":message, "assistant_message":copy, "proposal_card_count":0, "map":deepcopy(identity)})
    context = {"session_id":"owner", "accepted_plan_id":identity["active_plan_id"],
        "accepted_plan_content_hash":"hash", "draft_operations":[], "evaluated_proposal":None}
    artifacts = {"trace.json":{"scenario":name, "accepted":False, "chat_turn_count":3,
        "initial":deepcopy(identity), "final":deepcopy(identity)}, "exact_chat_io.json":turns,
        "blocked_visible_results.json":visible, "context_conflicts_before_reload.json":deepcopy(context),
        "context_conflicts_after_reload.json":deepcopy(context),
        "conflict_controls_before_reload.json":[{"trusted":True, "id":"copilot-submit"} for _ in messages]}
    first = turns[0]["output"]["payload"]
    if fault == "changed_message":
        visible[0]["user_message"] = "Different message"
    elif fault == "partial_draft":
        first["session"]["draft"] = [{"type":"set_stop_duration"}]
    elif fault == "false_ready":
        first["turn"]["interpretation"]["state"] = "proposal_ready"
    elif fault == "wrong_question":
        first["turn"]["interpretation"]["intents"][0]["clarification_question"] = "Generic question?"
    elif fault == "foreign_owner":
        first["session"]["session_id"] = "foreign"
    elif fault == "changed_map":
        visible[0]["map"]["rendered_geometry_hash"] = "different"
    elif fault == "changed_context":
        artifacts["context_conflicts_after_reload.json"]["accepted_plan_id"] = "different"
    elif fault == "untrusted_click":
        artifacts["conflict_controls_before_reload.json"][0]["trusted"] = False
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.combined_conflict_control_receipt(artifacts.__getitem__, name=name)
    else:
        verifier.combined_conflict_control_receipt(artifacts.__getitem__, name=name)


def test_scenario_reader_passes_list_artifacts_to_combined_receipt(tmp_path, monkeypatch):
    """Portable reader regression; mocked audit boundaries are not runtime proof."""
    name, node = "combined_chat", "tests.product_app.browser.test_unit::test_unit"
    case = tmp_path / "conversation" / name
    case.mkdir(parents=True)
    state = tmp_path / "_state" / hashlib.sha256(name.encode()).hexdigest()[:12]
    state.mkdir(parents=True)
    identity = {"active_plan_id":"parent"}
    artifacts = {
        "trace.json":{"schema_version":"conversation-browser-v1", "scenario":name,
            "assertions_passed":True, "initial":identity},
        "exact_chat_io.json":[], "console_errors.json":[], "network_failures.json":[],
        "server_lifecycle.json":{"state_directory":str(state.resolve())},
        "provider_scope.json":{"live_provider_call_count":0, "real_chat_endpoint":True,
            "internal_api_mocked":False, "live_openai_connectivity_proven":False, "fake_sdk_attempt_count":0},
    }
    for filename, value in artifacts.items():
        (case / filename).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "state_credential_redaction.json").write_text(json.dumps({
        "schema_version":"conversation-state-credential-redaction-v1", "scope":"completed_runtime_state_evidence",
        "assertions_passed":True, "redacted_file_count":0, "files":[]}), encoding="utf-8")
    observed = []

    def receipt(read, *, name):
        observed.append((name, read("exact_chat_io.json")))
        assert observed[-1] == ("combined_chat", [])

    monkeypatch.setattr(verifier, "CONVERSATION_CASES", {name:"test_unit"})
    monkeypatch.setattr(verifier, "_junit", lambda *args: {node})
    monkeypatch.setattr(verifier, "_lifecycle", lambda *args, **kwargs: {
        "installed_at":"2026-10-07T00:00:01+00:00", "finalized_at":"2026-10-07T00:00:02+00:00",
        "run_id":"unit_combined_chat", "process_id":77, "sent_provider_calls":0})
    monkeypatch.setattr(verifier, "_runtime", lambda *args: {node})
    monkeypatch.setattr(verifier, "_http", lambda *args, **kwargs: [])
    monkeypatch.setattr(verifier, "combined_chat_control_receipt", receipt)
    monkeypatch.setattr(verifier, "_conversation_control_closure", lambda *args: None)
    monkeypatch.setattr(core, "_verify_journey_captures", lambda *args: None)
    result = verifier.verify_scenarios(tmp_path, mode="conversation", run_id="unit", process={
        "pid":77, "exit_code":0, "outer_app_server_started":False, "started_at":"2026-10-07T00:00:00+00:00"})
    assert observed == [("combined_chat", [])] and result["verified"] and result["test_count"] == 1
