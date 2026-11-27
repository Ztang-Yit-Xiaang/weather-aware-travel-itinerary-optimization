from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

pytest_plugins = ["recorded_runtime_fixture"]
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_e2e1_vertical_slice.py"
EVIDENCE = None
SPEC = importlib.util.spec_from_file_location("verify_e2e1_vertical_slice", SCRIPT)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture(autouse=True)
def historical_evidence_root(recorded_runtime_inputs, monkeypatch):
    monkeypatch.setattr(
        __import__(__name__), "EVIDENCE", recorded_runtime_inputs / "results/e2e/r01_provider_isolation_20260907_v2"
    )


def _manifest() -> dict:
    return json.loads((EVIDENCE / verifier.MANIFEST_NAME).read_text(encoding="utf-8"))


def _rehash(manifest: dict) -> None:
    seed = dict(manifest)
    seed.pop("content_hash")
    manifest["content_hash"] = verifier._canonical_sha256(seed)


def _isolation_manifest() -> dict:
    return {
        "provider_calls": 0,
        "provider_isolation": {
            "artifact": "provider_isolation.json",
            "run_id": "r01_synthetic_verifier_case",
            "server_pid": 123,
            "record": {
                "schema_version": "e2e1-provider-isolation-v1",
                "run_id": "r01_synthetic_verifier_case",
                "process_id": 123,
                "scope": "application_server",
                "guard_boundaries": [
                    "httpx.Client.send",
                    "httpx.AsyncClient.send",
                    "socket.connect",
                ],
                "installed_at": "2026-09-07T12:00:00+00:00",
                "finalized_at": "2026-09-07T12:01:00+00:00",
                "finalized": True,
                "effective_adapter": "deterministic",
                "actual_adapter": "DeterministicCopilotAdapter",
                "api_key_disabled": True,
                "app_initialized": True,
                "server_started": True,
                "server_stopped": True,
                "blocked_attempts": [],
                "sent_provider_calls": 0,
                "allowed_local_connections": 3,
                "failure_code": None,
            },
        },
    }


def _verify_isolation_record(manifest: dict) -> None:
    verifier._verify_provider_isolation_record(
        manifest,
        datetime.fromisoformat("2026-09-07T12:02:00+00:00"),
    )


def test_provider_isolation_verifier_accepts_complete_bound_witness(tmp_path: Path) -> None:
    manifest = _isolation_manifest()
    record = manifest["provider_isolation"]["record"]
    (tmp_path / "provider_isolation.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    _verify_isolation_record(manifest)
    verifier._verify_provider_isolation(tmp_path, manifest)


def _isolation_root_manifest() -> dict:
    manifest = _isolation_manifest()
    manifest.update(
        {
            "schema_version": "e2e1-file-io-manifest-v3",
            "generated_at": "2026-09-07T12:02:00+00:00",
            "base_url": "http://127.0.0.1:8127",
            "assertions_passed": True,
            "pytest_exit_code": 0,
            "scenarios": list(verifier.SCENARIOS),
            "health": {
                "build_id": "build_test",
                "map": "maplibre_ready",
                "routing": "runtime_osrm_ready",
                "copilot": "deterministic_adapter_selected",
            },
            "command": [
                "python",
                "-m",
                "pytest",
                "tests/product_app/browser/test_e2e1_vertical_slice_live.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            "files": [],
            "source_inputs": [],
            "llm_records": [],
            "llm_input_output_policy": "Synthetic root-validation fixture only.",
            "content_hash": "",
        }
    )
    _rehash(manifest)
    return manifest


def test_provider_isolation_manifest_root_accepts_new_schema() -> None:
    verifier._verify_root(_isolation_root_manifest())


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda m: m.update(schema_version="e2e1-file-io-manifest-v2"), "manifest_schema_invalid"),
        (lambda m: m.pop("provider_isolation"), "manifest_fields_invalid"),
        (lambda m: m.update(provider_calls=False), "provider_calls_nonzero"),
        (lambda m: m.update(assertions_passed=False), "assertions_not_passed"),
        (lambda m: m.update(pytest_exit_code=3), "pytest_exit_invalid"),
        (lambda m: m["health"].update(copilot="openai_configured"), "health_binding_invalid"),
        (lambda m: m["provider_isolation"]["record"].update(finalized=False), "isolation_incomplete"),
        (lambda m: m["provider_isolation"]["record"].update(sent_provider_calls=1), "isolation_provider_calls_invalid"),
    ],
)
def test_provider_isolation_manifest_rejects_coherently_rehashed_forgeries(
    mutation,
    error: str,
) -> None:
    manifest = _isolation_root_manifest()
    mutation(manifest)
    _rehash(manifest)
    with pytest.raises(verifier.E2E1VerificationError, match=error):
        verifier._verify_root(manifest)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", "old-schema", "scope_invalid"),
        ("scope", "browser", "scope_invalid"),
        ("guard_boundaries", ["socket.connect"], "scope_invalid"),
        ("run_id", "another_run", "identity_mismatch"),
        ("process_id", 321, "identity_mismatch"),
        ("process_id", True, "identity_mismatch"),
        ("finalized", False, "incomplete"),
        ("finalized", 1, "incomplete"),
        ("api_key_disabled", False, "incomplete"),
        ("app_initialized", False, "incomplete"),
        ("server_started", False, "incomplete"),
        ("server_stopped", False, "incomplete"),
        ("failure_code", "server_failed", "incomplete"),
        ("effective_adapter", "openai", "adapter_invalid"),
        ("actual_adapter", "OpenAICopilotAdapter", "adapter_invalid"),
        ("installed_at", "invalid", "timestamp_invalid"),
        ("installed_at", "2026-09-07T12:00:00", "timestamp_not_utc"),
        ("installed_at", "2026-09-07T12:00:00+01:00", "timestamp_not_utc"),
        ("installed_at", "2026-09-07T12:01:30+00:00", "timestamp_order_invalid"),
        ("finalized_at", None, "timestamp_invalid"),
        ("finalized_at", "2026-09-07T12:02:30+00:00", "timestamp_order_invalid"),
        ("sent_provider_calls", 1, "provider_calls_invalid"),
        ("sent_provider_calls", False, "provider_calls_invalid"),
        ("sent_provider_calls", "0", "provider_calls_invalid"),
        ("sent_provider_calls", None, "provider_calls_invalid"),
        ("sent_provider_calls", -1, "provider_calls_invalid"),
        ("allowed_local_connections", 0, "local_connections_invalid"),
        ("allowed_local_connections", True, "local_connections_invalid"),
        ("allowed_local_connections", "3", "local_connections_invalid"),
        ("blocked_attempts", None, "attempts_invalid"),
    ],
)
def test_provider_isolation_verifier_rejects_forged_witness_fields(
    field: str,
    value: object,
    error: str,
) -> None:
    manifest = _isolation_manifest()
    manifest["provider_isolation"]["record"][field] = value
    with pytest.raises(verifier.E2E1VerificationError, match=error):
        _verify_isolation_record(manifest)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("artifact", "other.json", "artifact_invalid"),
        ("run_id", "", "identity_invalid"),
        ("server_pid", True, "identity_invalid"),
        ("server_pid", 0, "identity_invalid"),
        ("record", None, "record_fields_invalid"),
    ],
)
def test_provider_isolation_verifier_rejects_invalid_projection_fields(
    field: str,
    value: object,
    error: str,
) -> None:
    manifest = _isolation_manifest()
    manifest["provider_isolation"][field] = value
    with pytest.raises(verifier.E2E1VerificationError, match=error):
        _verify_isolation_record(manifest)


@pytest.mark.parametrize("provider_calls", [None, False, True, "0", -1, 1])
def test_provider_isolation_verifier_rejects_unmeasured_or_mismatched_count(
    provider_calls: object,
) -> None:
    manifest = _isolation_manifest()
    manifest["provider_calls"] = provider_calls
    with pytest.raises(verifier.E2E1VerificationError, match="provider_calls_invalid"):
        _verify_isolation_record(manifest)


@pytest.mark.parametrize("target", ["projection", "record"])
def test_provider_isolation_verifier_rejects_missing_required_fields(target: str) -> None:
    manifest = _isolation_manifest()
    if target == "projection":
        manifest.pop("provider_isolation")
    else:
        manifest["provider_isolation"]["record"].pop("finalized")
    with pytest.raises(verifier.E2E1VerificationError, match="fields_invalid"):
        _verify_isolation_record(manifest)


@pytest.mark.parametrize(
    "change,error",
    [
        ({}, "attempts_blocked"),
        ({"sequence": 2}, "attempt_invalid"),
        ({"sequence": True}, "attempt_invalid"),
        ({"boundary": "unknown"}, "attempt_invalid"),
        ({"reason": "unknown"}, "attempt_invalid"),
        ({"at": "2026-09-07T12:01:30+00:00"}, "attempt_time_invalid"),
    ],
)
def test_provider_isolation_verifier_rejects_blocked_attempts(
    change: dict,
    error: str,
) -> None:
    manifest = _isolation_manifest()
    attempt = {
        "sequence": 1,
        "boundary": "httpx.AsyncClient.send",
        "reason": "non_loopback_target",
        "at": "2026-09-07T12:00:30+00:00",
    }
    attempt.update(change)
    manifest["provider_isolation"]["record"]["blocked_attempts"] = [attempt]
    with pytest.raises(verifier.E2E1VerificationError, match=error):
        _verify_isolation_record(manifest)


def test_provider_isolation_verifier_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(verifier.E2E1VerificationError, match="artifact_missing"):
        verifier._verify_provider_isolation(tmp_path, _isolation_manifest())


@pytest.mark.parametrize(
    "mutation",
    [
        {"run_id": "another_run"},
        {"sent_provider_calls": False},
    ],
)
def test_provider_isolation_verifier_rejects_artifact_projection_mismatch(
    tmp_path: Path,
    mutation: dict,
) -> None:
    manifest = _isolation_manifest()
    record = copy.deepcopy(manifest["provider_isolation"]["record"])
    record.update(mutation)
    (tmp_path / "provider_isolation.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    with pytest.raises(verifier.E2E1VerificationError, match="file_projection_mismatch"):
        verifier._verify_provider_isolation(tmp_path, manifest)


def test_historical_bundle_cannot_claim_current_source_verification() -> None:
    # Keep the immutable earlier run for artifact-level regression fixtures; its
    # former pass does not establish that today's expanded source universe ran.
    recorded_sources = [row["path"] for row in _manifest()["source_inputs"]]
    assert recorded_sources != list(verifier.SOURCE_INPUTS)
    with pytest.raises(verifier.E2E1VerificationError, match="^e2e1_source_universe_invalid$"):
        verifier.verify(EVIDENCE)


def test_verifier_accepts_ephemeral_loopback_base_url() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["base_url"] = "http://127.0.0.1:49152"
    _rehash(manifest)

    verifier._verify_root(manifest)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:49152",
        "http://0.0.0.0:49152",
        "https://127.0.0.1:49152",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:49152/app",
    ],
)
def test_verifier_rejects_noncanonical_runtime_url(base_url: str) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["base_url"] = base_url
    _rehash(manifest)

    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_base_url_invalid"):
        verifier._verify_root(manifest)


def test_verifier_rejects_coherently_rehashed_nonnull_llm_input() -> None:
    forged = copy.deepcopy(_manifest())
    forged["llm_records"][0]["record"]["llm_input"] = {"forged": True}
    forged["content_hash"] = verifier._canonical_sha256(forged)
    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_llm_input_persisted"):
        verifier._verify_llm_records(EVIDENCE, forged)


def test_verifier_rejects_coherently_rehashed_file_binding() -> None:
    forged = copy.deepcopy(_manifest())
    forged["files"][0]["sha256"] = "f" * 64
    forged["content_hash"] = verifier._canonical_sha256(forged)
    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_artifact_hash_mismatch"):
        verifier._verify_files(EVIDENCE, forged)


def test_verifier_rejects_forged_cross_layer_child_identity() -> None:
    def forged_read(path: Path) -> dict:
        value = verifier._load(path)
        if path == EVIDENCE / "registered_replacement" / "trace.json":
            value = copy.deepcopy(value)
            value["child_plan_id"] = "plan_forged00000000"
        return value

    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_registered_plan_identity_mismatch"):
        verifier._verify_scenarios(EVIDENCE, forged_read)


def test_verifier_rejects_report_that_is_not_exactly_reproducible(tmp_path: Path) -> None:
    shutil.copy2(EVIDENCE / verifier.REPORT_NAME, tmp_path / verifier.REPORT_NAME)
    (tmp_path / verifier.REPORT_NAME).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_report_bytes_mismatch"):
        verifier._verify_report(tmp_path, _manifest())


def test_verifier_rejects_manifest_url_disconnected_from_recorded_requests() -> None:
    with pytest.raises(
        verifier.E2E1VerificationError,
        match="e2e1_recorded_request_origin_mismatch",
    ):
        verifier._verify_recorded_request_origins(
            EVIDENCE,
            "http://127.0.0.1:49152",
        )


def test_verifier_rejects_hostile_one_click_request_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = verifier._load
    target = (EVIDENCE / "registered_replacement" / "one_click_preview_request.json").resolve()

    def forged_load(path: Path) -> dict:
        value = original_load(path)
        if path.resolve() == target:
            value = copy.deepcopy(value)
            value["url"] = "https://attacker.example/api/sessions/session_" + "d" * 32 + "/preview"
        return value

    monkeypatch.setattr(verifier, "_load", forged_load)
    with pytest.raises(
        verifier.E2E1VerificationError,
        match="e2e1_recorded_request_origin_mismatch",
    ):
        verifier._verify_recorded_request_origins(
            EVIDENCE,
            _manifest()["base_url"],
        )


def test_verifier_rejects_one_click_preview_from_another_session() -> None:
    target = (EVIDENCE / "registered_replacement" / "one_click_preview_response.json").resolve()

    def forged_read(path: Path) -> dict:
        value = verifier._load(path)
        if path.resolve() == target:
            value = copy.deepcopy(value)
            value["session"]["session_id"] = "session_" + "e" * 32
        return value

    with pytest.raises(
        verifier.E2E1VerificationError,
        match="e2e1_one_click_session_identity_mismatch",
    ):
        verifier._verify_one_click_registered(
            EVIDENCE / "registered_replacement",
            forged_read,
        )


@pytest.mark.parametrize(
    ("url", "query_keys"),
    [
        (
            "http://127.0.0.1:8127/api/sessions/session_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/not-preview?expected=/preview",
            None,
        ),
        (
            "http://127.0.0.1:8127/api/sessions/session_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/preview?unexpected=1",
            None,
        ),
    ],
)
def test_request_target_rejects_endpoint_or_query_substitution(
    url: str,
    query_keys: set[str] | None,
) -> None:
    assert (
        verifier._valid_request_target(
            url,
            base_url="http://127.0.0.1:8127",
            method="POST",
            expected_method="POST",
            endpoint="/preview",
            query_keys=query_keys,
        )
        is False
    )


def _valid_stream() -> list[dict]:
    client_id = "client_message_" + "a" * 32
    request_id = "copilot_request_" + "b" * 32
    return [
        {
            "schema_version": "copilot-stream-v1",
            "type": "message.accepted",
            "sequence": 1,
            "request_id": request_id,
            "client_message_id": client_id,
        },
        {
            "schema_version": "copilot-stream-v1",
            "type": "assistant.status",
            "sequence": 2,
            "request_id": request_id,
            "client_message_id": client_id,
            "state": "interpreting",
            "message": "Reading the current itinerary and validating a typed suggestion.",
        },
        {
            "schema_version": "copilot-stream-v1",
            "type": "turn.completed",
            "sequence": 3,
            "request_id": request_id,
            "client_message_id": client_id,
            "payload": {"turn": {"client_message_id": client_id}},
        },
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda events: events[0].update(schema_version="copilot-stream-v0"), "schema_invalid"),
        (lambda events: events[1].update(sequence=3), "sequence_invalid"),
        (lambda events: events[1].update(client_message_id="client_message_" + "c" * 32), "client_message_id_mismatch"),
        (lambda events: events[2].update(request_id="copilot_request_" + "d" * 32), "request_id_invalid"),
        (lambda events: events[1].update(state="working"), "status_invalid"),
        (lambda events: events[0].update(unexpected=True), "event_fields_invalid"),
    ],
)
def test_copilot_stream_verifier_rejects_identity_and_schema_mutations(
    mutation,
    error: str,
) -> None:
    events = _valid_stream()
    mutation(events)
    with pytest.raises(verifier.E2E1VerificationError, match=error):
        verifier._verify_copilot_stream(
            events,
            expected_client_message_id="client_message_" + "a" * 32,
        )


@pytest.mark.parametrize(
    "leak",
    [
        "Authorization: Bearer TOPSECRET",
        "aUtHoRiZaTiOn=BeArEr TOPSECRET",
        "sk-proj-ABCdef0123456789",
        "OMITTED_SECRET\nX-Session-Token: RealSessionToken123",
    ],
)
def test_security_verifier_rejects_case_insensitive_secret_leaks(
    tmp_path: Path,
    leak: str,
) -> None:
    (tmp_path / "leak.txt").write_text(leak, encoding="utf-8")
    with pytest.raises(verifier.E2E1VerificationError):
        verifier._verify_text_security(tmp_path)


def test_security_verifier_accepts_only_the_explicit_session_token_sentinel(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe.json").write_text(
        json.dumps({"headers": {"X-Session-Token": "OMITTED_SECRET"}}),
        encoding="utf-8",
    )
    verifier._verify_text_security(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "authorization",
        "api_key",
        "OPENAI_API_KEY",
        "cookie",
        "set-cookie",
        "x-session-token",
        "mutation_token",
        "session_token",
    ],
)
def test_security_verifier_accepts_explicit_redaction_not_secret_material(tmp_path: Path, field: str) -> None:
    payload = {"nested": {field: "OMITTED_SECRET"}}
    (tmp_path / "redacted.json").write_text(json.dumps(payload), encoding="utf-8")
    verifier._verify_text_security(tmp_path)
    (tmp_path / "security_exact_io.json").write_text(
        json.dumps(
            {
                "scope": "synthetic_security_fixture_no_credentials",
                "input": payload,
                "output": {"accepted_explicit_redaction": True},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "field",
    [
        "authorization",
        "api_key",
        "OPENAI_API_KEY",
        "cookie",
        "set-cookie",
        "x-session-token",
        "mutation_token",
        "session_token",
    ],
)
@pytest.mark.parametrize("location", ["nested", "list", "encoded_json"])
def test_security_verifier_rejects_unredacted_structured_sensitive_values(
    tmp_path: Path,
    field: str,
    location: str,
) -> None:
    sensitive = {field: "SYNTHETIC_UNREDACTED_TEST_VALUE"}
    payload = {
        "nested": sensitive if location == "nested" else [sensitive] if location == "list" else json.dumps(sensitive)
    }
    (tmp_path / "synthetic_leak.json").write_text(json.dumps(payload), encoding="utf-8")
    code = (
        "e2e1_session_token_persisted"
        if field == "x-session-token" and location != "encoded_json"
        else "e2e1_structured_secret_persisted"
    )
    with pytest.raises(verifier.E2E1VerificationError, match="^" + code + "$") as caught:
        verifier._verify_text_security(tmp_path)
    (tmp_path / "security_exact_io.json").write_text(
        json.dumps(
            {
                "scope": "synthetic_security_fixture_no_credentials",
                "input": payload,
                "output": {"rejected": str(caught.value)},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("value", ["omitted_secret", "OMITTED_SECRET ", "prefix_OMITTED_SECRET", None, False, {}, []])
def test_security_verifier_does_not_accept_near_sentinels_or_other_value_types(tmp_path: Path, value: object) -> None:
    payload = {"nested": {"OPENAI_API_KEY": value}}
    (tmp_path / "synthetic_leak.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.E2E1VerificationError, match="^e2e1_structured_secret_persisted$") as caught:
        verifier._verify_text_security(tmp_path)
    (tmp_path / "security_exact_io.json").write_text(
        json.dumps(
            {
                "scope": "synthetic_security_fixture_no_credentials",
                "input": payload,
                "output": {"rejected": str(caught.value)},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "artifact,field,value,error",
    [
        ("reload_http_io.json", "response_status_code", 409, "accepted_reload_http_invalid"),
        (
            "session_reload_assertions.json",
            "session_creation_count_after_reload",
            2,
            "accepted_session_continuity_invalid",
        ),
        (
            "session_reload_assertions.json",
            "selected_stop_id",
            "bixby_creek_bridge_viewpoint",
            "accepted_session_continuity_invalid",
        ),
        ("session_reload_assertions.json", "conversation_id", None, "accepted_session_continuity_invalid"),
        ("reload_rendered_diagnostics.json", "console_errors", ["409 Conflict"], "accepted_browser_errors"),
    ],
)
def test_accepted_reload_verifier_rejects_session_fallback_and_lost_state(
    artifact: str,
    field: str,
    value: object,
    error: str,
) -> None:
    def forged_read(path: Path) -> dict:
        payload = verifier._load(path)
        if path == EVIDENCE / "accepted_reload" / artifact:
            payload = copy.deepcopy(payload)
            payload[field] = value
        return payload

    with pytest.raises(verifier.E2E1VerificationError, match=error):
        verifier._verify_accepted(EVIDENCE, forged_read)


@pytest.fixture
def mobile_keyboard_observation():
    prompts = ["Keep Stearns Wharf.", "Unsupported mixed request.", "Replace Bixby."]
    typing = []
    for number, prompt in enumerate(prompts):
        events = [
            {"key": key, "shift": False, "trusted": True, "target": "copilot-input", "value": prompt[:index]}
            for index, key in enumerate(prompt)
        ]
        submitted = copy.deepcopy(events)
        if number == 2:
            submitted.append(
                {"key": "Enter", "shift": False, "trusted": True, "target": "copilot-input", "value": prompt}
            )
        typing.append(
            {
                "input": prompt,
                "events": events,
                "value_before_submit": prompt,
                "focused_before_submit": "copilot-input",
                "submit_method": "enter" if number == 2 else "touch",
                "events_through_submit": submitted,
                "requests_before": number,
                "requests_after": number + 1,
            }
        )
    return prompts, {
        "schema_version": "mobile-keyboard-browser-v1",
        "os_virtual_keyboard_tested": False,
        "message_request_count": 3,
        "focus": {
            "opened": "copilot-input",
            "tab": "copilot-submit",
            "shift_tab": "copilot-input",
            "escape": "open-copilot",
        },
        "newline": {
            "value": "\n",
            "request_count": 0,
            "events": [{"key": "Enter", "shift": True, "trusted": True, "target": "copilot-input", "value": ""}],
        },
        "typing": typing,
        "reduced_viewport": {
            "width": 390,
            "height": 480,
            "controls": [
                {"selector": selector, "x": 10, "y": 350, "width": width, "height": 44, "center_unoccluded": True}
                for selector, width in (("#copilot-input", 300), ("#copilot-submit", 44), ("#copilot-form", 370))
            ],
        },
        "conversation_scroll": {
            "input_method": "browser_wheel",
            "before": {"top": 400, "height": 100, "content_height": 500, "page_x": 0, "page_y": 120},
            "after": {"top": 200, "height": 100, "content_height": 500, "page_x": 0, "page_y": 120},
        },
        "boundary_scroll": {
            "before": {"top": 400, "height": 100, "content_height": 500, "page_x": 0, "page_y": 120},
            "after": {"top": 400, "height": 100, "content_height": 500, "page_x": 0, "page_y": 120},
        },
    }


def test_mobile_keyboard_verifier_accepts_observed_browser_contract(mobile_keyboard_observation, tmp_path):
    prompts, observed = mobile_keyboard_observation
    verifier._verify_mobile_keyboard(observed, prompts)
    (tmp_path / "keyboard_verifier_io.json").write_text(
        json.dumps(
            {
                "input": observed,
                "prompts": prompts,
                "output": {"accepted": True},
                "scope": "synthetic verifier fixture, not browser evidence",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda row: row.update(os_virtual_keyboard_tested=True), "scope_invalid"),
        (lambda row: row.update(message_request_count=4), "scope_invalid"),
        (lambda row: row["focus"].update(escape="copilot-input"), "focus_invalid"),
        (lambda row: row["newline"].update(request_count=1), "newline_invalid"),
        (lambda row: row["newline"]["events"][0].update(trusted=False), "newline_invalid"),
        (lambda row: row["typing"][0]["events"][0].update(trusted=False), "typing_invalid"),
        (lambda row: row["typing"][0]["events"][0].update(key="Wrong"), "typing_invalid"),
        (lambda row: row["typing"][0]["events"][1].update(value=""), "typing_invalid"),
        (lambda row: row["typing"][0].update(focused_before_submit="copilot-submit"), "typing_invalid"),
        (lambda row: row["typing"][2].update(submit_method="touch"), "submission_invalid"),
        (lambda row: row["typing"][2]["events_through_submit"][-1].update(shift=True), "submission_invalid"),
        (lambda row: row["typing"][1].update(requests_after=3), "submission_invalid"),
        (lambda row: row["reduced_viewport"].update(height=844), "viewport_invalid"),
        (lambda row: row["reduced_viewport"]["controls"][0].update(y=479), "controls_obscured"),
        (lambda row: row["reduced_viewport"]["controls"][1].update(center_unoccluded=False), "controls_obscured"),
        (lambda row: row["conversation_scroll"]["after"].update(top=400), "scroll_invalid"),
        (lambda row: row["conversation_scroll"]["after"].update(page_y=150), "scroll_invalid"),
        (lambda row: row["boundary_scroll"]["before"].update(top=200), "scroll_chaining"),
        (lambda row: row["boundary_scroll"]["after"].update(page_y=150), "scroll_chaining"),
    ],
)
def test_mobile_keyboard_verifier_rejects_false_claims(mobile_keyboard_observation, mutation, error, tmp_path):
    prompts, observed = mobile_keyboard_observation
    mutation(observed)
    with pytest.raises(verifier.E2E1VerificationError, match=error) as caught:
        verifier._verify_mobile_keyboard(observed, prompts)
    (tmp_path / "keyboard_verifier_io.json").write_text(
        json.dumps(
            {
                "input": observed,
                "prompts": prompts,
                "output": {"rejected": str(caught.value)},
                "scope": "synthetic verifier fixture, not browser evidence",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def desktop_visibility_observation():
    viewport = {"width": 1024, "height": 768}
    parent = {
        "tag": "HTML",
        "class_name": "",
        "overflow_x": "visible",
        "overflow_y": "visible",
        "scroll_top": 0,
        "scroll_left": 0,
        "client_left": 0,
        "client_top": 0,
        "client_width": 1024,
        "client_height": 768,
        "display": "block",
        "visibility": "visible",
        "opacity": 1,
        "rect": {"left": 0, "top": 0, "right": 1024, "bottom": 768, "width": 1024, "height": 768},
    }
    expected = {"#map-context-title": "Bixby Creek Bridge Viewpoint", "#copilot-submit": "→"}
    elements = []
    for selector, text in expected.items():
        is_title = selector == "#map-context-title"
        left, top, width, height = (100, 200, 280, 30) if is_title else (900, 650, 44, 44)
        rect = {
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "width": width,
            "height": height,
        }
        elements.append(
            {
                "selector": selector,
                "present": True,
                "text": text,
                "disabled": None if is_title else False,
                "rect": rect,
                "clip": {key: rect[key] for key in ("left", "top", "right", "bottom")},
                "display": "block",
                "visibility": "visible",
                "opacity": 1,
                "visible_ratio": 1,
                "ancestors": [copy.deepcopy(parent)],
                "points": [
                    {
                        "x": left + width * dx,
                        "y": top + height * dy,
                        "unoccluded": True,
                        "hit": {"id": selector[1:], "tag": "H3" if is_title else "BUTTON", "class_name": ""},
                    }
                    for dx, dy in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))
                ],
            }
        )
    observed = {
        "at": "2026-10-07T07:00:01+00:00",
        "viewport": viewport,
        "elements": elements,
        "scroll_owners": [
            {
                "tag": "HTML",
                "class_name": "",
                "scroll_top": 0,
                "scroll_left": 0,
                "client_height": 768,
                "scroll_height": 768,
            }
        ],
    }
    checkpoint = {"captured_at": "2026-10-07T07:00:00+00:00", "readiness": {"viewport": viewport}}
    return observed, checkpoint, expected


def test_desktop_visibility_verifier_accepts_exact_unobscured_controls(desktop_visibility_observation, tmp_path):
    observed, checkpoint, expected = desktop_visibility_observation
    verifier._verify_workspace_visibility(observed, checkpoint, "2026-10-07T07:00:02+00:00", expected)
    (tmp_path / "visibility_verifier_io.json").write_text(
        json.dumps(
            {
                "input": observed,
                "checkpoint": checkpoint,
                "expected_text": expected,
                "output": {"accepted": True},
                "scope": "synthetic verifier fixture, not browser evidence",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["elements"][0]["points"][0]["hit"].update(id="copilot-disclosure"),
        lambda row: row["elements"][1]["points"][0]["hit"].update(id="conversation"),
        lambda row: row["elements"][0]["points"][0].update(unoccluded=False),
        lambda row: row["elements"][1].update(disabled=True),
        lambda row: row["elements"][0]["ancestors"][0].update(display="none"),
        lambda row: row["elements"][0].update(visible_ratio=0.9),
        lambda row: row["elements"][0].update(text="Different stop"),
        lambda row: row.update(at="2026-10-07T06:59:59+00:00"),
    ],
)
def test_desktop_visibility_verifier_rejects_false_claims(desktop_visibility_observation, mutation, tmp_path):
    observed, checkpoint, expected = desktop_visibility_observation
    mutation(observed)
    with pytest.raises(verifier.E2E1VerificationError, match="e2e1_workspace_visible_controls_invalid") as caught:
        verifier._verify_workspace_visibility(observed, checkpoint, "2026-10-07T07:00:02+00:00", expected)
    (tmp_path / "visibility_verifier_io.json").write_text(
        json.dumps(
            {
                "input": observed,
                "checkpoint": checkpoint,
                "expected_text": expected,
                "output": {"rejected": str(caught.value)},
                "scope": "synthetic verifier fixture, not browser evidence",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        lambda row: row.update(x=-39.49),
        lambda row: row.update(y=574),
        lambda row: row.update(x=True),
        lambda row: row.update(width=500),
        lambda row: row.update(plan_id="foreign_plan"),
        lambda row: row.update(coordinates=[-121.9, 36.37]),
        lambda row: row.update(rendered_stop_ids=[]),
        lambda row: row.update(observed_at="2026-10-07T06:59:59Z"),
        lambda row: row["visibility_probes"][0].update(unoccluded=False),
        lambda row: row["visibility_probes"][0]["hit"].update(tag="SECTION"),
        lambda row: row["visibility_probes"][0]["hit"].update(class_name="map-context-card"),
        lambda row: row["visibility_probes"][0].update(x=90),
    ],
)
def test_selected_marker_verifier_checks_exact_canonical_stop_and_canvas(mutation, tmp_path):
    observed = {
        "feature_found": True,
        "stop_id": "bixby",
        "plan_id": "parent",
        "coordinates": [-121.901894, 36.371479],
        "x": 54,
        "y": 355,
        "width": 451,
        "height": 573,
        "rendered_stop_ids": ["bixby"],
        "observed_at": "2026-10-07T07:00:01Z",
    }
    observed["visibility_probes"] = [
        {
            "x": 54 + dx,
            "y": 355 + dy,
            "unoccluded": True,
            "hit": {"tag": "CANVAS", "id": "", "class_name": "maplibregl-canvas"},
        }
        for dx, dy in ((0, 0), (-10, 0), (10, 0), (0, -10), (0, 10))
    ]
    checkpoint = {
        "selected_stop": "bixby",
        "captured_at": "2026-10-07T07:00:00Z",
        "readiness": {"container": {"width": 451, "height": 573}},
    }
    plan = {
        "plan_id": "parent",
        "stops": {
            "features": [{"properties": {"stop_id": "bixby"}, "geometry": {"coordinates": [-121.901894, 36.371479]}}]
        },
    }
    if mutation:
        mutation(observed)
        with pytest.raises(
            verifier.E2E1VerificationError, match="e2e1_workspace_selected_marker_not_visible"
        ) as caught:
            verifier._verify_workspace_selected_marker(observed, checkpoint, plan, "2026-10-07T07:00:02Z")
        result = {"rejected": str(caught.value)}
    else:
        verifier._verify_workspace_selected_marker(observed, checkpoint, plan, "2026-10-07T07:00:02Z")
        result = {"accepted": True}
    (tmp_path / "selected_marker_verifier_io.json").write_text(
        json.dumps(
            {
                "input": observed,
                "checkpoint": checkpoint,
                "canonical_plan": plan,
                "output": result,
                "scope": "synthetic verifier fixture, not browser evidence",
            }
        ),
        encoding="utf-8",
    )
