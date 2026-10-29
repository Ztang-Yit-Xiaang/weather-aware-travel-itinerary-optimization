"""Bootstrap registration and adversarial private-fault evidence, without a server."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/product_app/browser"))
import bootstrap_recovery_fixture as fixture  # noqa: E402
import run_e2e1_aggregate as aggregate  # noqa: E402
import run_e2e1_self_managed as runner  # noqa: E402
import run_e2e1_vertical_slice as dispatch  # noqa: E402
import verify_e2e1_bootstrap_recovery as verifier  # noqa: E402
import verify_e2e1_self_managed as self_managed  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402


def test_bootstrap_mode_has_two_real_cases_and_exact_shared_source_registration(tmp_path, monkeypatch):
    assert aggregate.EXPECTED_TEST_COUNT == 72 and len(aggregate.SUITES) == 23
    suite = next(row for row in aggregate.SUITES if row[0] == "bootstrap-recovery")
    assert suite[2] == tuple(verifier.TEST + "[" + case + "]" for case in verifier.CASES)
    source = set(deterministic_source_inputs("tests/product_app/browser/test_bootstrap_recovery_live.py"))
    assert {
        "tests/product_app/browser/bootstrap_recovery_fixture.py",
        "tests/product_app/browser/build_coherence_fixture.py",
        "scripts/run_e2e1_self_managed.py",
        "scripts/verify_e2e1_self_managed.py",
        "scripts/verify_e2e1_bootstrap_recovery.py",
    } <= source
    assert runner.MODES["bootstrap-recovery"]["scenario"] == "bootstrap_recovery"
    xml = (
        "<testsuite>"
        + "".join(
            '<testcase classname="tests.product_app.browser.test_bootstrap_recovery_live" name="' + name + '"/>'
            for name in suite[2]
        )
        + "</testsuite>"
    )
    (tmp_path / "pytest_junit.xml").write_text(xml, encoding="utf-8")
    assert len(self_managed._junit(tmp_path, "bootstrap-recovery")) == 2
    monkeypatch.setattr(sys, "argv", ["runner", "--browser-test", "bootstrap-recovery"])
    assert dispatch.parse_args().browser_test == "bootstrap-recovery"


@pytest.mark.parametrize("case", verifier.CASES)
@pytest.mark.parametrize(
    "fault", [None, "unrelated_change", "wrong_hash", "not_restored", "changed_production", "frontend_mock"]
)
def test_only_exact_private_bootstrap_source_faults_are_accepted(tmp_path, case, fault):
    repository = tmp_path / "repo"
    relative = (
        "configs/product_app_registry.json" if case == "core_not_ready" else "src/itinerary_system/product_app/api.py"
    )
    path = repository / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / relative).read_bytes())
    target, before = fixture.install_fault(repository, tmp_path, case)
    fixture.restore_fault(target, before, tmp_path)
    change = json.loads((tmp_path / "fault_source_change.json").read_text(encoding="utf-8"))
    restored = json.loads((tmp_path / "fault_source_restored.json").read_text(encoding="utf-8"))
    if fault == "unrelated_change":
        change["fault_utf8"] += "\n# unrelated fixture change"
        change["fault_sha256"] = restored["fault_sha256"] = hashlib.sha256(change["fault_utf8"].encode()).hexdigest()
    elif fault == "wrong_hash":
        change["before_sha256"] = "0" * 64
    elif fault == "not_restored":
        restored["restored_sha256"] = restored["fault_sha256"]
    elif fault == "changed_production":
        change["private_copy_only"] = False
    elif fault == "frontend_mock":
        change["frontend_fetch_mocked"] = True
    if fault:
        with pytest.raises((core.E2E1VerificationError, ValueError)):
            verifier.source_change(change, restored, case)
    else:
        verifier.source_change(change, restored, case)
    assert (ROOT / relative).read_bytes() == before


def fault_receipts(case):
    receipt = {
        "schema_version": verifier.SCHEMA,
        "case": case,
        "run_id": "case_fault",
        "process_id": 77,
        **{
            key: True
            for key in ("finalized", "api_key_disabled", "app_initialized", "server_started", "server_stopped")
        },
        "failure_code": None,
        "guard_boundaries": core.ISOLATION_BOUNDARIES,
        "scope": "application_server",
        "sent_provider_calls": 0,
        "allowed_local_connections": 0,
        "blocked_attempts": [],
        "normal_guard_relaxed": False,
        "fixture_scope": "private_copy_bootstrap_fault",
        "actual_service_available": case == "invalid_health",
        "core_ready": case == "invalid_health",
        "actual_adapter": "DeterministicCopilotAdapter" if case == "invalid_health" else None,
        "effective_adapter": "deterministic" if case == "invalid_health" else "unavailable_core_not_ready",
        "installed_at": "2026-09-08T10:00:01+00:00",
        "finalized_at": "2026-09-08T10:00:04+00:00",
    }
    lifecycle = {
        "run_id": "case_fault",
        "pid": 77,
        "exit_code": 0,
        "process_started_at": "2026-09-08T10:00:00+00:00",
        "fault_observed_at": "2026-09-08T10:00:02+00:00",
        "process_stopped_at": "2026-09-08T10:00:05+00:00",
    }
    process = {"started_at": "2026-09-08T09:59:59+00:00", "completed_at": "2026-09-08T10:00:06+00:00"}
    return receipt, lifecycle, process


@pytest.mark.parametrize("case", verifier.CASES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "paid",
        "unfinalized",
        "not_stopped",
        "normal_guard_weakened",
        "pretend_service",
        "wrong_pid",
        "old_run",
        "outside_time",
        "old_schema",
    ],
)
def test_fault_generation_never_impersonates_a_normal_ready_service(tmp_path, case, fault):
    receipt, lifecycle, process = fault_receipts(case)
    if fault == "paid":
        receipt["sent_provider_calls"] = 1
    elif fault == "unfinalized":
        receipt["finalized"] = False
    elif fault == "not_stopped":
        receipt["server_stopped"] = False
    elif fault == "normal_guard_weakened":
        receipt["normal_guard_relaxed"] = True
    elif fault == "pretend_service":
        receipt["actual_service_available"] = not receipt["actual_service_available"]
    elif fault == "wrong_pid":
        lifecycle["pid"] = 78
    elif fault == "old_run":
        receipt["run_id"] = "old"
    elif fault == "outside_time":
        receipt["finalized_at"] = "2026-09-08T10:30:00+00:00"
    elif fault == "old_schema":
        receipt["schema_version"] = "e2e1-provider-isolation-v1"
    path = tmp_path / "provider_isolation.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    lifecycle["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "server_lifecycle.json").write_text(json.dumps(lifecycle), encoding="utf-8")
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.fault_lifecycle(tmp_path, "case_fault", process, case)
    else:
        assert verifier.fault_lifecycle(tmp_path, "case_fault", process, case) == receipt


@pytest.mark.parametrize("case", verifier.CASES)
@pytest.mark.parametrize(
    "fault", [None, "map_ready", "rendered_ack", "session_created", "visible_app", "screenshot", "wrong_health"]
)
def test_blocked_bootstrap_does_not_claim_map_or_session_success(case, fault):
    blocked = {
        "case": case,
        "title": "The workspace is not ready" if case == "core_not_ready" else "Copilot could not start",
        "html": '<button id="retry-runtime">Retry</button>invalid_product_health',
        "app_hidden": True,
        "ready_maps": 0,
        "rendered_ack_count": 0,
        "session_present": False,
        "screenshot_taken": False,
    }
    health = {
        "schema_version": "product-health-v2"
        if case == "core_not_ready"
        else "bootstrap-intentionally-invalid-health-v1",
        "core_ready": case == "invalid_health",
        "components": {"registry": {"status": "failed"}},
    }
    if fault == "map_ready":
        blocked["ready_maps"] = 1
    elif fault == "rendered_ack":
        blocked["rendered_ack_count"] = 1
    elif fault == "session_created":
        blocked["session_present"] = True
    elif fault == "visible_app":
        blocked["app_hidden"] = False
    elif fault == "screenshot":
        blocked["screenshot_taken"] = True
    elif fault == "wrong_health":
        health["schema_version"] = "unknown"
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.blocked_result(blocked, health, case)
    else:
        verifier.blocked_result(blocked, health, case)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unanswered",
        "extra_http",
        "wrong_body",
        "wrong_method",
        "wrong_phase",
        "wrong_sequence",
        "no_finished",
        "failed",
        "missing_response",
        "late_response",
        "wrong_status",
        "non_api_post",
    ],
)
def test_complete_raw_requests_cannot_hide_attempts_without_responses(fault):
    lifecycle = {"installed_at": "2026-09-08T10:00:00Z", "finalized_at": "2026-09-08T10:00:10Z"}
    body = hashlib.sha256(b"").hexdigest()
    row = {
        "sequence": 1,
        "phase": "fault",
        "method": "GET",
        "url": "http://127.0.0.1:8127/api/health",
        "started_at": "2026-09-08T10:00:01Z",
        "response_at": "2026-09-08T10:00:02Z",
        "finished_at": "2026-09-08T10:00:03Z",
        "failed_at": None,
        "failure": None,
        "status": 200,
        "http_file": "http_001.json",
        "request_body_sha256": body,
    }
    record = {
        "request_sequence": 1,
        "phase": "fault",
        "method": "GET",
        "url": row["url"],
        "response_status_code": 200,
        "request_original_body_sha256": body,
    }
    rows, records = [row], [record]
    if fault == "unanswered":
        records = []
    elif fault == "extra_http":
        records.append({**record, "request_sequence": 2})
    elif fault == "wrong_body":
        row["request_body_sha256"] = "0" * 64
    elif fault == "wrong_method":
        row["method"] = "POST"
    elif fault == "wrong_phase":
        row["phase"] = "unrecorded"
    elif fault == "wrong_sequence":
        row["sequence"] = 2
    elif fault == "no_finished":
        row["finished_at"] = None
    elif fault == "failed":
        row["failure"] = "net::ERR_FAILED"
    elif fault == "missing_response":
        row["response_at"] = None
    elif fault == "late_response":
        row["response_at"] = "2026-09-08T11:00:02Z"
    elif fault == "wrong_status":
        row["status"] = 503
    elif fault == "non_api_post":
        row.update(url="http://127.0.0.1:8127/static/app.js", method="POST", http_file=None)
        records = []
    if fault:
        with pytest.raises((core.E2E1VerificationError, TypeError, ValueError)):
            verifier.raw_requests(rows, records, lifecycle, lifecycle)
    else:
        verifier.raw_requests(rows, records, lifecycle, lifecycle)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_raw",
        "different_raw",
        "old_prefix",
        "untrusted",
        "wrong_title",
        "before_start",
        "after_capture",
        "wrong_path",
    ],
)
def test_retry_click_must_be_a_new_trusted_raw_observation_after_repaired_server_start(fault):
    click = {
        "sequence": 1,
        "trusted": True,
        "id": "retry-runtime",
        "title": "The workspace is not ready",
        "path": "/app/itinerary",
        "clicked_at": "2026-09-08T10:00:03Z",
    }
    observer = {"before": [], "after": [click.copy()]}
    blocked = {"observer_before": [], "title": click["title"], "observed_at": "2026-09-08T10:00:01Z"}
    result = {"retry": click}
    recovered = {"installed_at": "2026-09-08T10:00:02Z"}
    ready = {"screenshot_started_at": "2026-09-08T10:00:04Z"}
    if fault == "missing_raw":
        observer["after"] = []
    elif fault == "different_raw":
        observer["after"][0]["sequence"] = 4
    elif fault == "old_prefix":
        blocked["observer_before"] = [click.copy()]
    elif fault == "untrusted":
        click["trusted"] = observer["after"][0]["trusted"] = False
    elif fault == "wrong_title":
        click["title"] = observer["after"][0]["title"] = "Other"
    elif fault == "before_start":
        click["clicked_at"] = observer["after"][0]["clicked_at"] = "2026-09-08T10:00:00Z"
    elif fault == "after_capture":
        click["clicked_at"] = observer["after"][0]["clicked_at"] = "2026-09-08T10:00:05Z"
    elif fault == "wrong_path":
        click["path"] = observer["after"][0]["path"] = "/unrelated"
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.retry_observer(observer, blocked, result, recovered, ready)
    else:
        verifier.retry_observer(observer, blocked, result, recovered, ready)


@pytest.mark.parametrize("context", ["original", "accepted_child"])
def test_both_bootstrap_contexts_require_their_actual_session_presence(context):
    blocked = {
        "case": "core_not_ready",
        "title": "The workspace is not ready",
        "html": '<button id="retry-runtime"></button>',
        "app_hidden": True,
        "ready_maps": 0,
        "rendered_ack_count": 0,
        "session_present": context == "accepted_child",
        "screenshot_taken": False,
    }
    health = {
        "schema_version": "product-health-v2",
        "core_ready": False,
        "components": {"registry": {"status": "failed"}},
    }
    verifier.blocked_result(blocked, health, "core_not_ready", context)
    blocked["session_present"] = not blocked["session_present"]
    with pytest.raises(core.E2E1VerificationError):
        verifier.blocked_result(blocked, health, "core_not_ready", context)


def test_acceptance_hash_uses_full_utf8_canonical_json_not_short_plan_hash():
    value = {"name": "海岸", "value": 2, "content_hash": "ignored"}
    expected = hashlib.sha256('{"name":"海岸","value":2}'.encode()).hexdigest()
    assert verifier.acceptance_hash(value) == expected
    assert len(expected) == 64 and expected != core._artifact_content_hash(value)
    assert (
        expected
        != hashlib.sha256(
            json.dumps({"name": "海岸", "value": 2}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), object()])
def test_acceptance_hash_rejects_noncanonical_numbers_and_objects(value):
    with pytest.raises(core.E2E1VerificationError, match="acceptance_hash_input_invalid"):
        verifier.acceptance_hash({"value": value})


def accepted_restore_fixture():
    pointer = {
        "schema_version": "workspace-pointer-v1",
        "accepted_plan_id": "child",
        "accepted_plan_content_hash": "child_hash",
        "original_plan_id": "parent",
        "original_plan_content_hash": "parent_hash",
        "decision_id": "decision",
        "revision": 1,
        "trip_id": "trip",
        "run_id": "run",
    }
    pointer["content_hash"] = verifier.acceptance_hash(pointer)
    public = {
        key: pointer[key]
        for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
    }
    plan = {
        "plan_id": "child",
        "content_hash": "child_hash",
        "stops": {"features": [{"properties": {"stop_id": "stop"}}]},
        "route_path": {"coordinates": [[1, 2], [3, 4]]},
    }
    accepted = {
        "result": {"pointer": pointer},
        "session": {
            "session_id": "session",
            "trip_id": "trip",
            "run_id": "run",
            "accepted_plan_id": "child",
            "draft": [],
            "proposal": None,
        },
        "workspace": {"accepted_plan_pointer": public, "geography": {"plans": [plan]}},
    }
    transition = {
        "accept_response": accepted,
        "session": {"session_id": "session"},
        "proposal": {"child_plan_id": "child", "child_plan_content_hash": "child_hash", "parent_plan_id": "parent"},
        "parent": {"active_plan_content_hash": "parent_hash"},
    }
    return deepcopy(accepted), transition


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "public_hash",
        "public_id",
        "public_revision",
        "public_extra",
        "full_hash",
        "full_original_id",
        "full_original_hash",
        "full_accepted_hash",
        "session_owner",
        "session_trip",
        "session_run",
        "coordinated_owner",
        "missing_child",
        "duplicate_child",
        "duplicate_both_child",
        "marker_duplicate_both",
        "wrong_geometry",
        "wrong_plan_hash",
    ],
)
def test_accepted_retry_binds_exact_pointer_owner_and_canonical_geography(fault):
    payload, transition = accepted_restore_fixture()
    pointer = transition["accept_response"]["result"]["pointer"]
    if fault in {"public_hash", "public_id", "public_revision", "public_extra"}:
        key = {
            "public_hash": "accepted_plan_content_hash",
            "public_id": "accepted_plan_id",
            "public_revision": "revision",
            "public_extra": "original_plan_id",
        }[fault]
        payload["workspace"]["accepted_plan_pointer"][key] = "foreign"
    elif fault == "full_hash":
        pointer["content_hash"] = "0" * 64
    elif fault in {"full_original_id", "full_original_hash", "full_accepted_hash"}:
        key = {
            "full_original_id": "original_plan_id",
            "full_original_hash": "original_plan_content_hash",
            "full_accepted_hash": "accepted_plan_content_hash",
        }[fault]
        pointer[key] = "foreign"
        pointer["content_hash"] = verifier.acceptance_hash(pointer)
    elif fault in {"session_owner", "session_trip", "session_run"}:
        key = {"session_owner": "session_id", "session_trip": "trip_id", "session_run": "run_id"}[fault]
        payload["session"][key] = "foreign"
    elif fault == "coordinated_owner":
        payload["session"]["trip_id"] = transition["accept_response"]["session"]["trip_id"] = "foreign"
    elif fault == "missing_child":
        payload["workspace"]["geography"]["plans"] = []
    elif fault in {"duplicate_child", "duplicate_both_child"}:
        payload["workspace"]["geography"]["plans"] *= 2
        if fault == "duplicate_both_child":
            transition["accept_response"]["workspace"]["geography"] = deepcopy(payload["workspace"]["geography"])
    elif fault == "marker_duplicate_both":
        payload["workspace"]["geography"]["plans"][0]["stops"]["features"] *= 2
        transition["accept_response"]["workspace"]["geography"] = deepcopy(payload["workspace"]["geography"])
    elif fault == "wrong_geometry":
        payload["workspace"]["geography"]["plans"][0]["route_path"]["coordinates"][0] = [9, 9]
    elif fault == "wrong_plan_hash":
        payload["workspace"]["geography"]["plans"][0]["content_hash"] = "foreign"
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.restored_acceptance(payload, transition)
    else:
        verifier.restored_acceptance(payload, transition)
