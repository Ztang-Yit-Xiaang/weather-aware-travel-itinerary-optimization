from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

pytest_plugins = ["recorded_runtime_fixture"]
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_e2e1_live_copilot_chat.py"
SPEC = importlib.util.spec_from_file_location("run_e2e1_live_copilot_chat", SCRIPT)
assert SPEC and SPEC.loader
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)
VERIFY_SCRIPT = ROOT / "scripts" / "verify_e2e1_live_copilot_chat.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_e2e1_live_copilot_chat", VERIFY_SCRIPT)
assert VERIFY_SPEC and VERIFY_SPEC.loader
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)
HARDENED_REPLAY_EVIDENCE = "results/e2e/runtime_optimizer_repair_replay_20260905_v5"

SESSION_ID = "session_" + "a" * 32
CLIENT_MESSAGE_ID = "client_message_" + "b" * 32
STREAM_REQUEST_ID = "copilot_request_" + "c" * 32


def test_historical_replay_is_not_relabelled_as_current_source_evidence(recorded_runtime_inputs) -> None:
    root = recorded_runtime_inputs / HARDENED_REPLAY_EVIDENCE
    with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_manifest_fields_invalid"):
        verifier.verify(root)
    provider = json.loads((root / "live_copilot_chat/provider_exact_llm_io.json").read_text(encoding="utf-8"))
    records, context, _ = verifier._verify_provider_records(provider, allow_historical_schema=True)
    assert len(records) == 1 and context["schema_version"] == "copilot-context-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@cache
def _current_service_context() -> dict[str, object]:
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    with TemporaryDirectory(prefix="copilot-context-unit-") as directory:
        service = ProductService(registry, Path(directory), copilot=DeterministicCopilotAdapter())
        session, _, _ = service.create_session(registry.default.run_id)
        service.workspace.select(
            session, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": verifier.TARGET}
        )
        context = json.loads(json.dumps(service.copilot_context(session).as_dict()))
    context["session_id"] = SESSION_ID
    return context


def _context() -> dict[str, object]:
    return copy.deepcopy(_current_service_context())


def _request(context: dict[str, object] | None = None) -> dict[str, object]:
    context = context or _context()
    prompt_input = "\n".join(
        (
            "<workspace_context>",
            live._canonical_prompt_json(context),
            "</workspace_context>",
            "<recent_conversation>",
            live._canonical_prompt_json([]),
            "</recent_conversation>",
            "<user_request>",
            live._canonical_prompt_json(live.APPROVED_PROMPT),
            "</user_request>",
        )
    )
    return {
        "input": prompt_input,
        "instructions": live.DEVELOPER_INSTRUCTIONS,
        "max_output_tokens": 1_200,
        "model": live.MODEL,
        "reasoning": {"effort": "low", "context": "current_turn"},
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "OpenAIInterpretationSchemaV2",
                "schema": live.to_strict_json_schema(live.OpenAIInterpretationSchemaV2),
                "strict": True,
            },
            "verbosity": "low",
        },
        "tools": [],
    }


def _record(request: dict[str, object] | None = None) -> dict[str, object]:
    request_text = json.dumps(request or _request(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    response_text = json.dumps(
        {
            "status": "completed",
            "usage": {
                "input_tokens": 4_238,
                "input_tokens_details": {"cached_tokens": 3_800},
                "output_tokens": 266,
                "total_tokens": 4_504,
            },
            "output": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": "copilot-live-exact-transport-attempt-v1",
        "sequence": 1,
        "case_id": "e2e1_live_website_chat",
        "effort": "low",
        "case_attempt": 1,
        "started_at": "2026-09-05T12:00:00+00:00",
        "completed_at": "2026-09-05T12:00:00.250000+00:00",
        "latency_ms": 250.0,
        "method": "POST",
        "url": "https://api.openai.com/v1/responses",
        "request_body_utf8": request_text,
        "request_body_sha256": _sha256_text(request_text),
        "response_status_code": 200,
        "response_body_utf8": response_text,
        "response_body_sha256": _sha256_text(response_text),
        "transport_error_code": None,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
    }


def _provider_document(records: list[dict[str, object]] | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "e2e1-live-provider-exact-io-v1",
        "generated_at": "2026-09-05T12:00:01+00:00",
        "scope": "synthetic_california_coast_single_chat",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "records": records or [_record()],
    }
    value["content_hash"] = verifier._canonical_sha256(value)
    return value


def _http_record(session_id: str = SESSION_ID) -> dict[str, object]:
    request_text = "{}"
    response_text = "{}"
    return {
        "schema_version": "e2e1-product-http-io-v1",
        "method": "POST",
        "url": f"http://127.0.0.1:49152/api/sessions/{session_id}/copilot/messages",
        "request_body_utf8": request_text,
        "request_body_sha256": _sha256_text(request_text),
        "response_status_code": 200,
        "response_body_utf8": response_text,
        "response_body_sha256": _sha256_text(response_text),
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
        "request_original_body_sha256": _sha256_text(request_text),
        "response_original_body_sha256": _sha256_text(response_text),
        "credential_redactions": {"request": [], "response": []},
        "body_capture": "exact_utf8",
    }


def _sse_raw() -> str:
    common = {
        "schema_version": "copilot-stream-v1",
        "request_id": STREAM_REQUEST_ID,
        "client_message_id": CLIENT_MESSAGE_ID,
    }
    events = [
        {**common, "type": "message.accepted", "sequence": 1},
        {
            **common,
            "type": "assistant.status",
            "sequence": 2,
            "state": "interpreting",
            "message": "Reading the current itinerary and validating a typed suggestion.",
        },
        {
            **common,
            "type": "turn.completed",
            "sequence": 3,
            "payload": {"turn": {"client_message_id": CLIENT_MESSAGE_ID}},
        },
    ]
    return "\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n"


def _set_verifier_fixture_context_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    normalized = _context()
    normalized["session_id"] = "<SESSION_ID>"
    monkeypatch.setattr(
        verifier,
        "EXPECTED_NORMALIZED_CONTEXT_SHA256",
        _sha256_text(verifier._canonical_prompt_json(normalized)),
    )


def test_live_runner_requires_explicit_billed_confirmation(tmp_path: Path) -> None:
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_billed_authorization_required"):
        live.run_live(output_dir=tmp_path / "evidence", confirmed=False)


def test_live_runner_refuses_an_occupied_explicit_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_port_available", lambda port: False)
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_port_8127_occupied"):
        live.run_live(output_dir=tmp_path / "evidence", confirmed=True, port=8127)


def test_live_transport_binds_exact_authorized_request_and_one_attempt() -> None:
    authorization = live._validate_transport([_record()], expected_session_id=SESSION_ID)
    assert authorization["workspace_context"] == _context()
    assert authorization["request_body_sha256"] == _record()["request_body_sha256"]
    assert authorization["output_schema_current"] is True
    assert authorization["text"]["format"]["schema"] == live.to_strict_json_schema(live.OpenAIInterpretationSchemaV2)
    assert authorization["text"]["format"]["schema"] != (live.OpenAIInterpretationSchemaV2.model_json_schema())

    forged_request = _request()
    forged_request["instructions"] = str(forged_request["instructions"]) + " extra"
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_provider_request_invalid"):
        live._validate_transport([_record(forged_request)])
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_attempt_cap_invalid"):
        live._validate_transport([_record(), _record()])


def test_live_transport_guard_authorizes_only_one_provider_attempt() -> None:
    request_body = json.dumps(_request(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    validate = live._one_provider_attempt_validator()
    validate(request_body)
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_attempt_cap_exceeded"):
        validate(request_body)


def test_combined_live_profile_is_exact_and_one_attempt_only() -> None:
    request = _request()
    request["input"] = str(request["input"]).replace(
        live._canonical_prompt_json(live.APPROVED_PROMPT),
        live._canonical_prompt_json(live.COMBINED_PROMPT),
    )
    body = json.dumps(request).encode()
    authorization = live._provider_request_authorization(body)
    assert authorization["user_prompt"] == live.COMBINED_PROMPT
    verifier._verify_provider_records(_provider_document([_record(request)]))
    wrong_profile = live._one_provider_attempt_validator(expected_prompt=live.APPROVED_PROMPT)
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_selected_prompt_mismatch"):
        wrong_profile(body)
    validate = live._one_provider_attempt_validator(expected_prompt=live.COMBINED_PROMPT)
    validate(body)
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_attempt_cap_exceeded"):
        validate(body)
    request["input"] = str(request["input"]).replace("90 minutes", "91 minutes")
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_provider_request_invalid"):
        live._provider_request_authorization(json.dumps(request).encode())


def test_combined_verifier_binds_every_clause_and_local_highlights(recorded_runtime_inputs) -> None:
    interpretation = json.loads(
        (
            recorded_runtime_inputs
            / "results/e2e/combined_chat_final_regression_20261007_v4/exact_io/combined_fake_complete.json"
        ).read_text()
    )["validated_local_output"]
    raw = copy.deepcopy(interpretation)
    raw["highlights"] = {"day_ids": [], "stop_ids": [], "segment_ids": [], "candidate_ids": []}
    verifier._verify_approved_interpretation(raw, interpretation, prompt=live.COMBINED_PROMPT)
    for mutation in ("drop", "retarget", "reorder", "duration", "highlights"):
        changed = copy.deepcopy(interpretation)
        if mutation == "drop":
            changed["intents"].pop()
        elif mutation == "retarget":
            changed["intents"][1]["target_stop_id"] = verifier.TARGET
        elif mutation == "reorder":
            changed["intents"].reverse()
        elif mutation == "duration":
            changed["intents"][1]["parameters"]["duration"]["preferred_minutes"] = 91
        else:
            changed["highlights"]["stop_ids"] = [verifier.TARGET]
        with pytest.raises(verifier.LiveE2E1VerificationError):
            verifier._verify_approved_interpretation(raw, changed, prompt=live.COMBINED_PROMPT)


def test_combined_runtime_verifier_checks_final_evaluation_and_optimizer(recorded_runtime_inputs) -> None:
    proposal = json.loads(
        (
            recorded_runtime_inputs
            / "results/e2e/combined_chat_browser_20261007_v3/conversation/combined_chat/preview_response.json"
        ).read_text()
    )["proposal"]
    receipt = verifier._verify_runtime_execution(proposal)
    assert receipt["optimizer_execution"]["child_plan_id"] != receipt["child_plan_id"]
    for mutation in ("optimizer", "duration", "operations", "evaluation"):
        changed = copy.deepcopy(proposal)
        runtime = changed["runtime_execution"]
        if mutation == "optimizer":
            runtime["optimizer_execution"]["evaluation_executed"] = False
        elif mutation == "duration":
            next(stop for stop in runtime["child_plan"]["selected_stops"] if stop["stop_id"] == "griffith_observatory")[
                "visit_duration_minutes"
            ] = 91
        elif mutation == "operations":
            runtime["operations"].pop()
        else:
            runtime["evaluation_certificate"]["plan_content_hash"] = "0" * 16
        with pytest.raises(verifier.LiveE2E1VerificationError):
            verifier._verify_runtime_execution(changed)


def test_combined_paid_bundle_has_no_leftover_single_operation_assumption(recorded_runtime_inputs, monkeypatch) -> None:
    # Checks recorded semantics only, not current-source or new paid-call proof.
    root = recorded_runtime_inputs / "results/e2e/combined_paid_website_20261007_v1"
    lifecycle = json.loads((root / "run_lifecycle.json").read_text())
    trace = json.loads((root / live.SCENARIO / "trace.json").read_text())
    manifest = live.build_manifest(
        root,
        command=lifecycle["command"],
        pytest_exit_code=0,
        health={},
        trace=trace,
        base_url=lifecycle["base_url"],
        source_inputs=[],
    )
    with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_provider_controls_invalid"):
        verifier._verify_bundle(root, manifest)
    # Retain the old bytes and check semantics under their pinned old contract.
    provider_path = root / live.SCENARIO / "provider_exact_llm_io.json"
    original = provider_path.read_bytes()
    recorded = json.loads(json.loads(original)["records"][0]["request_body_utf8"])
    assert _sha256_text(recorded["instructions"]) == "2b0d9433261534d0a21225f53110e77b1fac66f3af277cebaa2a20fcc1ba8041"
    monkeypatch.setattr(verifier, "DEVELOPER_INSTRUCTIONS", recorded["instructions"])
    monkeypatch.setattr(
        verifier,
        "EXPECTED_NORMALIZED_CONTEXT_SHA256",
        "f350c66602a66de56de9539191602d506433dea56ae6e43496323a7bdd300e2a",
    )
    verifier._verify_bundle(root, manifest)
    assert provider_path.read_bytes() == original


def test_live_cost_and_browser_failure_gate_are_exact() -> None:
    assert (
        live._cost(
            {
                "input_tokens": 4_238,
                "cached_input_tokens": 3_800,
                "uncached_input_tokens": 438,
                "output_tokens": 266,
                "total_tokens": 4_504,
            }
        )
        == "0.0018105"
    )
    assert live._browser_transport_gate(3, []) == 3
    with pytest.raises(live.LiveE2E1Error, match="e2e1_live_attempt_cap_invalid"):
        live._browser_transport_gate(0, [])


def test_live_source_universe_and_command_are_bound() -> None:
    assert "src/itinerary_system/product_app/draft_compiler.py" in live.SOURCE_INPUTS
    assert live.SOURCE_INPUTS == verifier.SOURCE_INPUTS
    verifier._verify_command(
        [
            sys.executable,
            "-m",
            "pytest",
            live.BROWSER_TEST,
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            "temporary-pytest-root",
            "--junitxml",
            "temporary-evidence/pytest_junit.xml",
        ]
    )
    with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_command_binding_invalid"):
        verifier._verify_command(["not-the-executed-command"])


def test_live_verifier_requires_canonical_session_origin_endpoint_and_query() -> None:
    verifier._verify_http_record(
        _http_record(),
        "test",
        base_url="http://127.0.0.1:49152",
        endpoint_suffix="/copilot/messages",
    )
    invalid_records = []
    invalid_records.append(_http_record("session_1"))
    for url in (
        f"http://127.0.0.1:8127/api/sessions/{SESSION_ID}/copilot/messages",
        f"http://127.0.0.1:49152/api/sessions/{SESSION_ID}/preview",
        f"http://127.0.0.1:49152/api/sessions/{SESSION_ID}/copilot/messages?x=1",
        f"http://127.0.0.1:49152/api/sessions/{SESSION_ID}/copilot/messages#x",
    ):
        invalid_records.append({**_http_record(), "url": url})
    invalid_records.append({**_http_record(), "method": "GET"})
    for forged in invalid_records:
        with pytest.raises(verifier.LiveE2E1VerificationError, match="test_request_target_invalid"):
            verifier._verify_http_record(
                forged,
                "test",
                base_url="http://127.0.0.1:49152",
                endpoint_suffix="/copilot/messages",
            )


def test_live_verifier_binds_exact_sse_schema_sequence_and_identity() -> None:
    events = verifier._sse_events(_sse_raw())
    assert [event["sequence"] for event in events] == [1, 2, 3]
    frames = _sse_raw().strip().split("\n\n")
    forged_events = json.loads(frames[1].removeprefix("data: "))
    for mutation in (
        {"extra": True},
        {"sequence": 3},
        {"request_id": "copilot_request_" + "d" * 32},
        {"client_message_id": "client_message_" + "e" * 32},
        {"schema_version": "copilot-stream-v2"},
    ):
        forged = {**forged_events, **mutation}
        candidate = [frames[0], "data: " + json.dumps(forged), frames[2]]
        with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_copilot_stream_invalid"):
            verifier._sse_events("\n\n".join(candidate) + "\n\n")


def test_live_verifier_binds_provider_authorization_and_attempt_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_verifier_fixture_context_hash(monkeypatch)
    records, context, authorization = verifier._verify_provider_records(_provider_document())
    assert context == _context()
    assert authorization == live._provider_request_authorization(
        records[0]["request_body_utf8"], expected_session_id=SESSION_ID
    )

    for mutation, code in (
        ({"started_at": "2026-09-05T12:00:00"}, "e2e1_live_attempt_timestamp_invalid"),
        ({"response_status_code": 500}, "e2e1_live_provider_response_hash_invalid"),
        ({"method": "GET"}, "e2e1_live_transport_identity_invalid"),
        ({"request_body_sha256": "0" * 64}, "e2e1_live_provider_request_hash_invalid"),
    ):
        forged = {**_record(), **mutation}
        with pytest.raises(verifier.LiveE2E1VerificationError, match=code):
            verifier._verify_provider_records(_provider_document([forged]))
    with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_attempts_invalid"):
        verifier._verify_provider_records(_provider_document([_record(), _record()]))


def test_live_verifier_rejects_provider_prompt_schema_or_context_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_verifier_fixture_context_hash(monkeypatch)
    mutations = []
    instructions = _request()
    instructions["instructions"] = str(instructions["instructions"]) + " changed"
    mutations.append(instructions)
    schema = _request()
    schema["text"] = {"verbosity": "low"}
    mutations.append(schema)
    context = _context()
    context["selected_day"] = 6
    mutations.append(_request(context))
    for request in mutations:
        with pytest.raises(verifier.LiveE2E1VerificationError):
            verifier._verify_provider_records(_provider_document([_record(request)]))


def _replay_accounting() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    historical = {"schema_version": "e2e1-live-historical-provider-evidence-v1", "usage": {"input_tokens": 10}}
    trace = {
        "execution_mode": "recorded_exact_output_replay",
        "provider_transport_attempts": 0,
        "provider_calls_current_run": 0,
        "provider_request_body_sha256": None,
        "provider_response_body_sha256": None,
        "provider_request_authorization": None,
        **verifier.ZERO_USAGE,
        "billed_cost_usd": "0",
        "historical_provider_evidence": historical,
    }
    manifest = {
        "provider_transport_attempts": 0,
        "usage": verifier.ZERO_USAGE,
        "pricing": {"billed_cost_usd": "0"},
        "authorization": {
            "confirmed": False,
            "execution_mode": "recorded_exact_output_replay",
            "current_run_provider_calls": 0,
            "transport_attempt_cap": 0,
            "exact_provider_request": None,
            "historical_provider_evidence": historical,
        },
    }
    usage_file = {
        "schema_version": "e2e1-live-replay-usage-and-cost-v1",
        "execution_mode": "recorded_exact_output_replay",
        "current_run_provider_calls": 0,
        "current_run_usage": verifier.ZERO_USAGE,
        "current_run_billed_cost_usd": "0",
        "historical_provider_evidence": historical,
    }
    return trace, manifest, usage_file, historical


def test_replay_requires_zero_current_calls_and_labeled_historical_usage() -> None:
    trace, manifest, usage_file, historical = _replay_accounting()
    verifier._verify_replay_current_run_accounting(
        trace=trace,
        manifest=manifest,
        usage_file=usage_file,
        historical_provider_evidence=historical,
    )
    forged_trace = {**trace, "provider_calls_current_run": 1}
    with pytest.raises(
        verifier.LiveE2E1VerificationError,
        match="e2e1_live_replay_current_run_accounting_invalid",
    ):
        verifier._verify_replay_current_run_accounting(
            trace=forged_trace,
            manifest=manifest,
            usage_file=usage_file,
            historical_provider_evidence=historical,
        )
    unlabeled = copy.deepcopy(usage_file)
    unlabeled.pop("historical_provider_evidence")
    with pytest.raises(
        verifier.LiveE2E1VerificationError,
        match="e2e1_live_replay_current_run_accounting_invalid",
    ):
        verifier._verify_replay_current_run_accounting(
            trace=trace,
            manifest=manifest,
            usage_file=unlabeled,
            historical_provider_evidence=historical,
        )


@pytest.mark.parametrize(
    "name,text,code",
    [
        ("api-key.txt", "synthetic leak SK-PROJ-abcdefghijklmnop", "e2e1_live_api_key_persisted"),
        ("bearer.txt", "header value BeArEr abcdefghijklmnop", "e2e1_live_authorization_persisted"),
        ("session.txt", "X-SESSION-TOKEN: real-secret", "e2e1_live_session_token_persisted"),
    ],
)
def test_live_verifier_detects_text_secrets(tmp_path: Path, name: str, text: str, code: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")
    with pytest.raises(verifier.LiveE2E1VerificationError, match=code):
        verifier._verify_text_security(tmp_path)


def test_live_verifier_detects_recursive_structured_secrets(tmp_path: Path) -> None:
    (tmp_path / "leak.json").write_text(
        json.dumps({"outer": [{"OpenAI-Api-Key": "not-redacted"}]}),
        encoding="utf-8",
    )
    with pytest.raises(
        verifier.LiveE2E1VerificationError,
        match="e2e1_live_structured_secret_persisted",
    ):
        verifier._verify_text_security(tmp_path)


def test_live_verifier_allows_only_explicit_secret_sentinel(tmp_path: Path) -> None:
    (tmp_path / "redacted.json").write_text(
        json.dumps({"nested": {"X-Session-Token": "OMITTED_SECRET"}}),
        encoding="utf-8",
    )
    verifier._verify_text_security(tmp_path)


@pytest.mark.parametrize("field", ["mutation_token", "session_token", "X-Session-Token"])
def test_live_verifier_checks_credentials_inside_exact_json_body(tmp_path: Path, field: str) -> None:
    record = {"response_body_utf8": json.dumps({field: "synthetic-private-token"})}
    (tmp_path / "http.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(verifier.LiveE2E1VerificationError, match="secret_persisted|token_persisted"):
        verifier._verify_text_security(tmp_path)
    record["response_body_utf8"] = json.dumps({field: "OMITTED_SECRET"})
    (tmp_path / "http.json").write_text(json.dumps(record), encoding="utf-8")
    verifier._verify_text_security(tmp_path)


def _runtime_proposal() -> dict[str, object]:
    def hashed(record):
        return {**record, "content_hash": verifier._artifact_hash(record)}

    request = {
        "request_id": "request_current",
        "confirmed_constraints": {"must_delete": [verifier.TARGET], "must_include": [verifier.CANDIDATE]},
    }
    child = hashed(
        {
            "plan_id": "plan_child",
            "parent_plan_id": "plan_parent",
            "source_run_id": "run_current",
            "planning_request_id": "request_current",
            "context_snapshot_id": "context_current",
            "sequence": [verifier.CANDIDATE],
            "day_assignments": {verifier.CANDIDATE: 5},
        }
    )
    planner = hashed(
        {
            "method_executed": "deterministic_context_aware_heuristic",
            "execution_status": "COMPLETED",
            "result_plan_id": child["plan_id"],
            "run_id": "run_current",
            "planning_request_id": "request_current",
            "context_snapshot_id": "context_current",
        }
    )
    certificate = hashed(
        {
            "certificate_id": "cert_current",
            "plan_id": child["plan_id"],
            "plan_content_hash": child["content_hash"],
            "comparison_eligibility": "eligible",
            "evaluation_status": "PASSED",
            "source_run_id": "run_current",
            "evaluated_at": "2026-09-05T12:00:01+00:00",
            "route_validation": {"context_snapshot_id": "context_current"},
        }
    )
    diff = {"diff_id": "diff_current", "parent_plan_id": "plan_parent", "child_plan_id": child["plan_id"]}
    proposal = {
        "execution_mode": "deterministic_repair_pipeline",
        "compiled_request": {"source_request_id": "request_current"},
        "parent_plan_id": "plan_parent",
        "child_plan_id": child["plan_id"],
        "child_plan_content_hash": child["content_hash"],
        "certificate_id": certificate["certificate_id"],
        "certificate_content_hash": certificate["content_hash"],
        "diff_id": diff["diff_id"],
        "diff_content_hash": verifier._artifact_hash(diff),
    }
    proposal["runtime_execution"] = {
        "schema_version": "runtime-repair-execution-v1",
        "execution_id": "execution_" + "a" * 32,
        "started_at": "2026-09-05T12:00:00+00:00",
        "completed_at": "2026-09-05T12:00:02+00:00",
        "request_id": request["request_id"],
        "request_content_hash": verifier._artifact_hash(request),
        "parent_plan_id": "plan_parent",
        "planner_run": planner,
        "repair_request": request,
        "child_plan": child,
        "plan_diff": diff,
        "evaluation_certificate": certificate,
        "child_plan_id": child["plan_id"],
        "child_plan_content_hash": child["content_hash"],
        "evaluation_id": certificate["certificate_id"],
        "evaluation_content_hash": certificate["content_hash"],
        "evaluation_executed": True,
        "evaluator": "PlanEvaluator.evaluate_final_plan",
        "planner_trace": {
            "request_id": request["request_id"],
            "planner_run": planner,
            "child_plan_id": child["plan_id"],
            "diff_id": diff["diff_id"],
            "failure_reasons": [],
            "decision_records": [
                {"day": 5, "selected": True, "feasible": True, "stop_sequence": [verifier.CANDIDATE], "violations": []}
            ],
        },
    }
    return proposal


def test_live_gate_rejects_registered_plan_and_binds_actual_runtime_artifacts() -> None:
    proposal = _runtime_proposal()
    verifier._verify_runtime_execution(proposal)
    forged = copy.deepcopy(proposal)
    forged["execution_mode"] = "registered_independent_evaluation"
    with pytest.raises(verifier.LiveE2E1VerificationError, match="runtime_execution_missing"):
        verifier._verify_runtime_execution(forged)
    for artifact, field, value in (
        ("repair_request", "request_id", "historical_request"),
        ("planner_run", "result_plan_id", "other_child"),
        ("child_plan", "content_hash", "0" * 16),
        ("evaluation_certificate", "plan_content_hash", "0" * 16),
        ("evaluation_certificate", "evaluated_at", "2026-09-04T12:00:01+00:00"),
        ("evaluation_certificate", "source_run_id", "historical_run"),
        ("plan_diff", "diff_id", "other_diff"),
    ):
        forged = copy.deepcopy(proposal)
        forged["runtime_execution"][artifact][field] = value
        with pytest.raises(verifier.LiveE2E1VerificationError):
            verifier._verify_runtime_execution(forged)


def test_live_metrics_use_evaluated_child_identity_and_exact_visible_values() -> None:
    proposal = {
        "child_plan_id": "child",
        "child_plan_content_hash": "child_hash",
        "certificate_id": "certificate",
        "eligibility": "eligible",
        "geography_plan": {
            "validated_legs": {"features": [{"properties": {"duration_s": 120}}, {"properties": {"duration_s": 30}}]}
        },
        "repair": {"tradeoffs": {"preservation_rate": 0.875, "weather_risk_delta": None, "weighted_edit_cost": 6}},
    }
    identity = {"planId": "child", "planContentHash": "child_hash"}
    snapshot = {
        "scope": {**identity, "evaluationId": "certificate"},
        "metrics": [
            {"scope": dict(identity), "label": label, "value": value}
            for label, value in (
                ("Route time", "2.5 min"),
                ("Contextual risk", "Unavailable"),
                ("Preservation", "88%"),
                ("Edit cost", "6"),
                ("Certificate", "eligible"),
            )
        ],
    }
    verifier._verify_visible_metrics(snapshot, proposal)
    for field, value in (
        ("planId", "parent"),
        ("planContentHash", "parent_hash"),
        ("evaluationId", "foreign_certificate"),
    ):
        forged = copy.deepcopy(snapshot)
        forged["scope"][field] = value
        with pytest.raises(verifier.LiveE2E1VerificationError, match="metrics_identity_mismatch"):
            verifier._verify_visible_metrics(forged, proposal)
    forged = copy.deepcopy(snapshot)
    forged["metrics"][0]["scope"]["planId"] = "parent"
    with pytest.raises(verifier.LiveE2E1VerificationError, match="metrics_identity_mismatch"):
        verifier._verify_visible_metrics(forged, proposal)
    for index in range(5):
        forged = copy.deepcopy(snapshot)
        forged["metrics"][index]["value"] = "Wrong value"
        with pytest.raises(verifier.LiveE2E1VerificationError, match="metrics_values_mismatch"):
            verifier._verify_visible_metrics(forged, proposal)


def test_independent_geometry_digest_ignores_key_order_but_preserves_coordinates() -> None:
    first = {
        "validated_legs": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[-122.0, 37.0], [-121.1, 36.1]]},
                    "properties": {"plan_id": "child", "role": "draft_preview"},
                }
            ],
        }
    }
    reordered = json.loads(json.dumps(first, sort_keys=True))
    reordered["validated_legs"]["features"][0]["properties"]["role"] = "active"
    digest = verifier._independent_geometry_hash(first)
    assert verifier._independent_geometry_hash(reordered) == digest
    reordered["validated_legs"]["features"][0]["geometry"]["coordinates"].reverse()
    assert verifier._independent_geometry_hash(reordered) != digest


def test_failed_browser_run_preserves_false_status_and_observed_exact_io(tmp_path: Path) -> None:
    scenario = tmp_path / live.SCENARIO
    scenario.mkdir()
    exact_io = {"request_body_utf8": "exact user input", "response_body_utf8": "observed response"}
    live._write_json(scenario / "product_copilot_http_io.json", exact_io)
    live._write_json(scenario / "browser_failure.json", {"assertions_passed": False, "message": "map mismatch"})
    live._preserve_failed_run(
        tmp_path,
        exit_code=1,
        command=["pytest"],
        health={},
        source_inputs=[],
        execution_mode="recorded_exact_output_replay",
        current_records=[],
    )
    trace = json.loads((tmp_path / "failure_trace.json").read_text(encoding="utf-8"))
    assert trace["assertions_passed"] is False
    assert trace["current_provider_transport_records"] == 0
    assert trace["current_billed_cost_usd"] == "0"
    assert "map_render_ack.json" in trace["missing_success_artifacts"]
    assert not (scenario / "map_render_ack.json").exists()
    assert "exact user input" in (tmp_path / live.REPORT_NAME).read_text(encoding="utf-8")
