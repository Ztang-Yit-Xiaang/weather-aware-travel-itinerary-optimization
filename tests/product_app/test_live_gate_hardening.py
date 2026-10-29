"""Zero-provider counterexamples for the authorized website evidence runner."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest_plugins = ["recorded_runtime_fixture"]
SPEC = importlib.util.spec_from_file_location(
    "live_evidence_fixtures", Path(__file__).with_name("test_e2e1_live_copilot_evidence.py")
)
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
live = fixtures.live
browser_spec = importlib.util.spec_from_file_location(
    "live_browser_helpers", Path(__file__).parent / "browser/test_e2e1_live_copilot_chat.py"
)
browser_helpers = importlib.util.module_from_spec(browser_spec)
browser_spec.loader.exec_module(browser_helpers)


def actual_http_record(endpoint="selection", payload=None):
    body = json.dumps(payload or {"session": {"accepted_plan_id": "plan_parent"}})
    response = SimpleNamespace(
        request=SimpleNamespace(post_data="{}", method="POST"),
        url=f"http://127.0.0.1:49152/api/sessions/{fixtures.SESSION_ID}/{endpoint}",
        status=200,
        body=lambda: body.encode(),
    )
    return browser_helpers._http_record(response)


def test_actual_browser_http_serializer_is_accepted_by_independent_verifier():
    fixtures.verifier._verify_http_record(
        actual_http_record(), "current_http", base_url="http://127.0.0.1:49152", endpoint_suffix="/selection"
    )


@pytest.mark.parametrize("mutation", ["place_name", "extra_alias", "parent_identity"])
def test_changed_synthetic_context_never_reaches_transport(mutation):
    context = fixtures._context()
    if mutation == "place_name":
        context["known_places"][0]["name"] = "AUDIT_ONLY_NON_SYNTHETIC_TEXT"
    elif mutation == "extra_alias":
        context["known_places"][0]["aliases"].append("AUDIT_ONLY_EXTRA_CONTEXT")
    else:
        context["accepted_plan_id"] = "plan_" + "f" * 16
        context["conversation_memory"]["plan_id"] = context["accepted_plan_id"]
    forwarded = []

    def receive(request):
        forwarded.append(request)
        return httpx.Response(200, json={})

    transport = live.ExactIOTransport(
        httpx.MockTransport(receive), request_validator=live._one_provider_attempt_validator()
    )
    transport.begin_case(effort=live.EFFORT, case_id="e2e1_live_website_chat")

    async def submit():
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(live.LiveE2E1Error, match="context_hash_invalid"):
                await client.post(live.PROVIDER_ENDPOINT, json=fixtures._request(context))

    asyncio.run(submit())
    assert forwarded == []
    assert len(transport.records) == 1
    assert transport.records[0]["response_status_code"] is None


def test_failed_completed_response_with_missing_usage_keeps_exact_io(tmp_path):
    scenario = tmp_path / live.SCENARIO
    scenario.mkdir()
    record = fixtures._record()
    record["response_body_utf8"] = '{"status":"failed"}'
    live._preserve_failed_run(
        tmp_path,
        exit_code=1,
        command=[],
        health={},
        source_inputs=[],
        execution_mode="live_openai",
        current_records=[record],
    )
    trace = json.loads((tmp_path / "failure_trace.json").read_text())
    assert trace["assertions_passed"] is False
    assert trace["current_usage"] is None
    assert trace["current_billed_cost_usd"] is None


@pytest.mark.parametrize("phase", ["startup", "subprocess", "verification"])
def test_runner_exception_scope_preserves_failure_and_transport(tmp_path, phase):
    (tmp_path / live.SCENARIO).mkdir()
    records = [fixtures._record()] if phase != "startup" else []
    with pytest.raises(RuntimeError, match="original failure"):
        with live._failure_evidence(
            tmp_path, source_inputs=[], execution_mode="live_openai", current_records=records, health={}
        ):
            raise RuntimeError("original failure")
    trace = json.loads((tmp_path / "failure_trace.json").read_text())
    assert trace["assertions_passed"] is False
    provider = json.loads((tmp_path / live.SCENARIO / "provider_exact_llm_io.json").read_text())
    assert provider["records"] == records
    assert (tmp_path / live.MANIFEST_NAME).is_file()
    assert (tmp_path / live.REPORT_NAME).is_file()
    assert json.loads((tmp_path / "runner_failure.json").read_text())["error_type"] == "RuntimeError"


@pytest.fixture
def fake_replay(monkeypatch, tmp_path, recorded_runtime_inputs):
    """Exercise real runner control flow; replace only external execution edges."""
    # The CLI replay changes these settings directly; restore them at fixture teardown.
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_COPILOT_MODEL", "fake-no-network")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key-not-usable")
    monkeypatch.setenv("PRODUCT_APP_ORIGIN", "http://127.0.0.1:49152")
    historical = recorded_runtime_inputs / fixtures.HARDENED_REPLAY_EVIDENCE / live.SCENARIO / "provider_exact_llm_io.json"
    output = tmp_path / "replay"
    monkeypatch.setattr(live, "_port_available", lambda port: True)
    monkeypatch.setattr(live, "_source_rows", lambda: [])
    monkeypatch.setattr(
        live,
        "create_product_app",
        lambda **kw: SimpleNamespace(
            state=SimpleNamespace(
                product_runtime=SimpleNamespace(
                    service=SimpleNamespace(copilot=kw["copilot"]), config=SimpleNamespace(copilot_adapter="openai")
                )
            )
        ),
    )

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(live.threading, "Thread", Thread)
    monkeypatch.setattr(
        live.uvicorn, "Server", lambda config: SimpleNamespace(run=lambda: None, started=True, should_exit=False)
    )
    monkeypatch.setattr(live, "_wait_ready", lambda thread, url: {})

    def subprocess_run(*args, **kwargs):
        live._write_json(output / live.SCENARIO / "browser_assertions.json", {"session_id": fixtures.SESSION_ID})
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s\n", stderr="")

    monkeypatch.setattr(live.subprocess, "run", subprocess_run)
    monkeypatch.setattr(
        live,
        "_build_trace",
        lambda *a, **kw: {
            "provider_calls_current_run": 0,
            "historical_provider_evidence": kw["historical_provider_evidence"],
        },
    )

    def manifest(*args, **kwargs):
        value = {
            "authorization": {"execution_mode": "recorded_exact_output_replay"},
            "provider_transport_attempts": 0,
            "usage": live.ZERO_USAGE,
            "pricing": {"billed_cost_usd": "0"},
        }
        return {**value, "content_hash": live._canonical_sha256(value)}

    monkeypatch.setattr(live, "build_manifest", manifest)
    monkeypatch.setattr(live, "render_report", lambda *args: "test report")
    return historical, output


@pytest.mark.parametrize("reject", [False, True])
def test_replay_must_run_independent_verifier_before_success(fake_replay, monkeypatch, reject, capsys):
    import verify_e2e1_live_copilot_chat as independent

    calls = []

    def verify(root):
        calls.append(root)
        if reject:
            raise independent.LiveE2E1VerificationError("e2e1_live_source_binding_invalid")
        return {"verified": True}

    monkeypatch.setattr(independent, "verify", verify)
    historical, output = fake_replay
    if reject:
        with pytest.raises(independent.LiveE2E1VerificationError, match="source_binding"):
            live.replay_recorded_provider_output(provider_io_path=historical, output_dir=output)
        assert json.loads((output / "failure_trace.json").read_text())["assertions_passed"] is False
        assert '"assertions_passed": true' not in capsys.readouterr().out
    else:
        assert live.replay_recorded_provider_output(provider_io_path=historical, output_dir=output) == 0
    assert calls == [output]


def test_authorized_but_uncompleted_transport_has_unknown_usage(tmp_path):
    scenario = tmp_path / live.SCENARIO
    scenario.mkdir()
    request = json.dumps(fixtures._request()).encode()
    live._one_provider_attempt_validator(scenario / "provider_request_authorized.json")(request)
    live._preserve_failed_run(
        tmp_path, exit_code=1, command=[], health={}, source_inputs=[], execution_mode="live_openai", current_records=[]
    )
    trace = json.loads((tmp_path / "failure_trace.json").read_text())
    assert trace["provider_completion_unknown"] is True
    assert trace["current_usage"] is None and trace["current_billed_cost_usd"] is None
    assert (
        json.loads((scenario / "provider_request_authorized.json").read_text())["request_body_utf8"] == request.decode()
    )


@pytest.mark.parametrize("phase", ["startup", "subprocess", "shutdown_with_original_error"])
def test_real_replay_control_flow_preserves_execution_exceptions(fake_replay, monkeypatch, phase):
    historical, output = fake_replay

    def fail(*args, **kwargs):
        raise RuntimeError("original runner failure")

    if phase == "subprocess":
        monkeypatch.setattr(live.subprocess, "run", fail)
    else:
        monkeypatch.setattr(live, "_wait_ready", fail)
        if phase == "shutdown_with_original_error":
            monkeypatch.setattr(live.threading.Thread, "is_alive", lambda self: True)
    with pytest.raises(RuntimeError, match="original runner failure"):
        live.replay_recorded_provider_output(provider_io_path=historical, output_dir=output)
    assert json.loads((output / "runner_failure.json").read_text())["error_type"] == "RuntimeError"
    assert json.loads((output / "failure_trace.json").read_text())["assertions_passed"] is False
    isolation = json.loads((output / "provider_isolation.json").read_text())
    assert isolation["finalized"] is False and isolation["sent_provider_calls"] == 0


@pytest.mark.parametrize("tamper", [None, "accept", "parent", "missing_preview", "changed_body", "incomplete_capture"])
def test_acceptance_ledger_binds_exact_http_and_immutable_parent(tmp_path, tamper):
    base = tmp_path / live.SCENARIO
    base.mkdir()
    ledger = []
    for endpoint, name in (
        ("selection", "initial_selection"),
        ("copilot/messages", "copilot"),
        ("draft/operations", "draft"),
        ("preview", "preview"),
    ):
        http = actual_http_record(endpoint)
        live._write_json(base / f"product_{name}_http_io.json", http)
        ledger.append(
            {
                "sequence": len(ledger) + 1,
                "method": "POST",
                "url": http["url"],
                "observed_at": "2026-09-08T10:00:00+00:00",
                "response_observed_at": "2026-09-08T10:00:01+00:00",
                "request_body_utf8": http["request_body_utf8"],
                "request_body_sha256": http["request_body_sha256"],
                "request_original_body_sha256": http["request_body_sha256"],
                "credential_redactions": [],
                "response_status_code": 200,
            }
        )
    sessions = {
        name: {"accepted_plan_id": "plan_parent"}
        for name in ("after_chat", "after_stage", "after_preview", "after_reload")
    }
    observation = {
        "schema_version": "e2e1-live-acceptance-observation-v1",
        "session_id": fixtures.SESSION_ID,
        "initial_plan_id": "plan_parent",
        "initial_content_hash": "hash_parent",
        "preview_parent_id": "plan_parent",
        "preview_parent_content_hash": "hash_parent",
        "accepted_plan_checkpoints": {"before_chat": "plan_parent", **{name: "plan_parent" for name in sessions}},
        "acceptance_mutations": [],
        "accept_or_keep_invoked": False,
    }
    if tamper == "accept":
        ledger.append({**ledger[-1], "sequence": 5, "url": ledger[-1]["url"].removesuffix("preview") + "accept"})
    elif tamper == "parent":
        sessions["after_reload"]["accepted_plan_id"] = "plan_child"
    elif tamper == "missing_preview":
        ledger.pop()
    elif tamper == "changed_body":
        ledger[-1]["request_body_utf8"] = '{"unexpected":true}'
    elif tamper == "incomplete_capture":
        http = actual_http_record()
        http["body_capture"] = "incomplete_utf8"
        live._write_json(base / "product_initial_selection_http_io.json", http)
    live._write_json(base / "product_mutation_ledger.json", ledger)
    live._write_json(base / "acceptance_observation.json", observation)

    def invoke():
        fixtures.verifier._verify_acceptance_observation(
            tmp_path,
            {"base_url": "http://127.0.0.1:49152", "generated_at": "2026-09-08T10:00:02+00:00"},
            fixtures.SESSION_ID,
            {"parent_plan_id": "plan_parent", "parent_plan_content_hash": "hash_parent"},
            sessions,
        )

    if tamper:
        with pytest.raises(fixtures.verifier.LiveE2E1VerificationError):
            invoke()
    else:
        invoke()


@pytest.mark.parametrize(
    "tamper", [None, "adapter", "blocked", "unfinished", "different_run", "zero_connections", "late_install"]
)
def test_replay_isolation_requires_finalized_current_run_receipt(tmp_path, tamper):
    from e2e1_provider_isolation import BOUNDARIES

    record = {
        "schema_version": "e2e1-provider-isolation-v1",
        "run_id": tmp_path.name,
        "process_id": 123,
        "scope": "application_server",
        "guard_boundaries": BOUNDARIES,
        "installed_at": "2026-09-08T10:00:01+00:00",
        "finalized_at": "2026-09-08T10:00:05+00:00",
        "finalized": True,
        "effective_adapter": "recorded_exact_output_replay",
        "actual_adapter": "RecordedCopilotAdapter",
        "api_key_disabled": True,
        "app_initialized": True,
        "server_started": True,
        "server_stopped": True,
        "blocked_attempts": [],
        "sent_provider_calls": 0,
        "allowed_local_connections": 4,
        "failure_code": None,
    }
    if tamper == "adapter":
        record["actual_adapter"] = "OpenAICopilotAdapter"
    elif tamper == "blocked":
        record["blocked_attempts"] = [{"boundary": "httpx.AsyncClient.send"}]
    elif tamper == "unfinished":
        record["finalized"] = False
    elif tamper == "different_run":
        record["run_id"] = "another_run"
    elif tamper == "zero_connections":
        record["allowed_local_connections"] = 0
    elif tamper == "late_install":
        record["installed_at"] = "2026-09-08T10:00:03+00:00"
    live._write_json(tmp_path / "provider_isolation.json", record)
    manifest = {
        "authorization": {"execution_mode": "recorded_exact_output_replay"},
        "generated_at": "2026-09-08T10:00:06+00:00",
        "run_lifecycle": {
            "record": {
                "process_id": 123,
                "observed_at": "2026-09-08T10:00:00+00:00",
                "server_ready_at": "2026-09-08T10:00:02+00:00",
                "server_stopped_at": "2026-09-08T10:00:04+00:00",
            }
        },
    }
    if tamper:
        with pytest.raises(fixtures.verifier.LiveE2E1VerificationError):
            fixtures.verifier._verify_replay_isolation(tmp_path, manifest)
    else:
        fixtures.verifier._verify_replay_isolation(tmp_path, manifest)


def test_cost_report_labels_estimates_not_invoice_billing(recorded_runtime_inputs):
    root = recorded_runtime_inputs / fixtures.HARDENED_REPLAY_EVIDENCE
    manifest = json.loads((root / live.MANIFEST_NAME).read_text())
    report = live.render_report(root, manifest)
    assert "Current-run estimated API cost" in report
    assert "Historical estimated API cost" in report
    assert "not invoice-confirmed" in report
    assert "Current-run billed cost" not in report and "Historical billed cost" not in report


@pytest.mark.parametrize("outcome", ["rejected", "timeout", "cancelled", "complete", "complete_then_cap_rejected"])
def test_failure_completion_status_distinguishes_authorized_unknown_from_no_dispatch(tmp_path, outcome):
    """Real validation/recording/reporting, never-fallthrough external fake only."""
    scenario = tmp_path / live.SCENARIO
    scenario.mkdir()
    authorized_path = scenario / "provider_request_authorized.json"
    forwarded = []
    response_text = fixtures._record()["response_body_utf8"]

    def receive(request):
        forwarded.append({"method":request.method,"url":str(request.url),"request_body_utf8":request.content.decode()})
        if outcome == "timeout":
            raise httpx.ReadTimeout("Synthetic external transport timeout", request=request)
        if outcome == "cancelled":
            raise asyncio.CancelledError("Synthetic external transport cancellation")
        return httpx.Response(200,content=response_text.encode(),headers={"content-type":"application/json"})

    transport = live.ExactIOTransport(httpx.MockTransport(receive),
        request_validator=live._one_provider_attempt_validator(authorized_path))
    transport.begin_case(effort=live.EFFORT,case_id="e2e1_live_website_chat")
    request = fixtures._request()
    if outcome == "rejected":
        request["instructions"] += " NOT_AUTHORIZED"

    async def submit():
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(live.PROVIDER_ENDPOINT,json=request)
            if outcome == "complete_then_cap_rejected":
                await client.post(live.PROVIDER_ENDPOINT,json=request)
        raise RuntimeError("Synthetic later browser failure")

    error = {"rejected":live.LiveE2E1Error,"timeout":httpx.ReadTimeout,
        "cancelled":asyncio.CancelledError,"complete":RuntimeError,
        "complete_then_cap_rejected":live.LiveE2E1Error}[outcome]
    with pytest.raises(error):
        with live._failure_evidence(tmp_path,source_inputs=[],execution_mode="live_openai",
                current_records=transport.records,health={}):
            asyncio.run(submit())
    trace = json.loads((tmp_path / "failure_trace.json").read_text())
    provider = json.loads((scenario / "provider_exact_llm_io.json").read_text())
    # Persist provenance and all observed I/O before the red/green assertion.
    live._write_json(tmp_path / "fake_transport_provenance.json", {
        "scope":"external_httpx_MockTransport_only_no_fallthrough","actual_provider_calls":0,
        "outcome":outcome,"fake_transport_attempts":len(forwarded),"exact_forwarded_requests":forwarded,
        "exact_response_body_utf8":response_text if outcome in {"complete","complete_then_cap_rejected"} else None,
        "transport_records":transport.records,"failure_trace":trace,
        "expected_provider_completion_unknown":outcome in {"timeout","cancelled"},
    })
    assert provider["records"] == transport.records
    assert len(forwarded) == (0 if outcome == "rejected" else 1)
    assert authorized_path.is_file() is (outcome != "rejected")
    assert len(transport.records) == (2 if outcome == "complete_then_cap_rejected" else 1)
    if forwarded:
        assert forwarded[0]["request_body_utf8"] == transport.records[0]["request_body_utf8"]
    if outcome in {"complete","complete_then_cap_rejected"}:
        assert transport.records[0]["response_body_utf8"] == response_text
    if outcome == "complete":
        assert trace["current_usage"] == live._usage(transport.records)
        assert trace["current_billed_cost_usd"] == live._cost(trace["current_usage"])
    else:
        assert trace["current_usage"] is None and trace["current_billed_cost_usd"] is None
    assert trace["assertions_passed"] is False
    assert trace["provider_completion_unknown"] is (outcome in {"timeout","cancelled"})
