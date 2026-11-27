"""Synthetic adversarial verifier tests; no provider, server, or browser launched."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests/product_app/browser")]
import e2e1_runtime_gate_coverage as coverage  # noqa: E402
import run_e2e1_aggregate as aggregate  # noqa: E402
import run_e2e1_self_managed as runner  # noqa: E402
import test_preview_geography_fault_live as browser  # noqa: E402
import verify_e2e1_preview_geography_fault as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402
from test_e2e1_optional_evidence_fault_verifier import ack, fixture  # noqa: E402


def test_backend_evidence_retention_accepts_only_the_declared_verifier_redaction(tmp_path):
    repository = tmp_path / "repo"
    state = tmp_path / "state"
    evidence = tmp_path / "evidence"
    repository.mkdir()
    (state / "sessions").mkdir(parents=True)
    evidence.mkdir()
    (repository / "geography_fault_boundary.json").write_text(
        json.dumps({"canonical": {"proposal": "eligible"}, "wire": {"proposal": "faulted"}}),
        encoding="utf-8",
    )
    session_path = state / "sessions" / "session.json"
    session_path.write_text(
        json.dumps(
            {
                "session": {"session_id": "session"},
                "token_verifier": {
                    "algorithm": "sha256-salted-v1",
                    "salt": "a" * 32,
                    "digest": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    browser.retain_backend_evidence(repository, state, "session", evidence)

    receipt = json.loads((evidence / "persisted_session.json").read_text(encoding="utf-8"))
    assert json.loads(receipt["utf8"])["token_verifier"] == "OMITTED_SECRET"
    assert receipt["credential_redactions"] == ["/token_verifier"]
    assert receipt["original_sha256"] == hashlib.sha256(session_path.read_bytes()).hexdigest()
    assert receipt["plaintext_credentials_persisted"] is False


def case(kind):
    reference = fixture()[0]
    proposal = reference["proposal"]
    proposal["expected_revision"] = 3
    proposal["runtime_execution"]["repair_request"].update(
        {
            "parent_plan_id": "plan_parent",
            "confirmed_constraints": {
                "must_delete": ["bixby_creek_bridge_viewpoint"],
                "must_include": ["pfeiffer_big_sur_state_park"],
                "affected_days": [5],
                "require_road_validated_routes": True,
            },
            "tolerance_profile": {"route_fallback_allowed": False},
            "candidate_pois": [{"stop_id": "pfeiffer_big_sur_state_park", "day": 5}],
            "baseline_route": [{"day": 5, "stop_ids": ["bixby_creek_bridge_viewpoint"]}],
        }
    )
    parent = reference["initial"]["workspace"]["geography"]["plans"][0]
    initial = {
        "session": {
            "schema_version": "workspace-context-v1",
            "session_id": "session",
            "trip_id": "trip",
            "run_id": "run",
            "accepted_plan_id": parent["plan_id"],
            "revision": 0,
            "proposal": None,
            "draft": [],
        },
        "workspace": {
            "geography": {"plans": [parent]},
            "accepted_plan_pointer": {
                "accepted_plan_id": parent["plan_id"],
                "accepted_plan_content_hash": parent["content_hash"],
            },
        },
    }
    operation = {
        "operation_id": "operation",
        "type": "replace_nearby",
        "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        "source": "map",
        "created_at": "2026-09-08T10:00:00+00:00",
        "evidence_refs": [],
    }
    staged = {**initial["session"], "revision": 3, "draft": [operation]}
    canonical = {"proposal": proposal, "session": {**staged, "revision": 4, "proposal": deepcopy(proposal)}}
    wire = deepcopy(canonical)
    for value in (wire["proposal"], wire["session"]["proposal"]):
        value["geography_plan"] = None if kind == "missing" else deepcopy(parent)
    restoration = {"session": deepcopy(staged), "workspace": deepcopy(initial["workspace"])}
    value = {
        "schema_version": "preview-geography-fault-browser-v1",
        "kind": kind,
        "coverage_category": "declared_backend_response_failure_path",
        "assertions_passed": True,
        "initial": initial,
        "staged": staged,
        "restoration": restoration,
        "wire": deepcopy(wire),
        "checkpoints": {
            label: ack(parent, "PREVIEW_FAILED" if label == "blocked_parent" else "PARENT_RENDERED")
            for label in ("initial_parent", "staged_parent", "restored_staged_parent", "blocked_parent")
        },
        "ui": {
            "map": {
                "previewState": "PREVIEW_FAILED",
                "activePlanId": parent["plan_id"],
                "renderedMapPlanId": parent["plan_id"],
                "activePlanContentHash": parent["content_hash"],
            },
            "metric": {"planId": parent["plan_id"], "planContentHash": parent["content_hash"]},
            "accept_disabled": True,
            "undo_disabled": False,
            "draft_text": "1 draft change",
            "toast": "The evaluated repair response did not match",
            "timeline": "Bixby Creek Bridge",
        },
    }
    boundary = {"canonical": canonical, "wire": wire}
    records = [
        {
            "url": verifier.BASE_URL + "/api/sessions/session" + suffix,
            "method": method,
            "response_status_code": 200,
            "response_body_utf8": json.dumps(payload),
            "request_body_utf8": json.dumps(request),
        }
        for suffix, method, payload, request in (
            ("", "GET", initial, {}),
            (
                "/draft/operations",
                "POST",
                {"session": staged, "operation": operation},
                {
                    "type": "replace_nearby",
                    "target": "bixby_creek_bridge_viewpoint",
                    "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
                    "source": "map",
                    "expected_revision": 2,
                },
            ),
            ("", "GET", restoration, {}),
            ("/preview", "POST", wire, {"expected_revision": 3}),
        )
    ]
    stored = {key: item for key, item in canonical["session"].items() if key != "schema_version"}
    stored.update(created_at="2026-09-08T10:00:00+00:00", last_accessed_at="2026-09-08T10:00:01+00:00")
    original = json.dumps(
        {
            "schema_version": "product-session-snapshot-v1",
            "session": stored,
            "token_verifier": {"algorithm": "sha256-salted-v1", "salt": "a" * 32, "digest": "b" * 64},
        }
    )
    body = json.dumps(
        {"schema_version": "product-session-snapshot-v1", "session": stored, "token_verifier": "OMITTED_SECRET"}
    )
    return (
        value,
        boundary,
        records,
        {
            "utf8": body,
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "body_capture": "credential_redacted",
            "credential_redactions": ["/token_verifier"],
            "plaintext_credentials_persisted": False,
        },
    )


@pytest.mark.parametrize("kind", verifier.CASES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "not_evaluated",
        "ineligible",
        "child_hash",
        "certificate_hash",
        "diff_child",
        "no_real_geometry",
        "wrong_replacement",
        "wire_extra_change",
        "wire_not_faulted",
        "mutated_persistence",
        "public_shape_as_persistence",
        "draft_lost",
        "parent_changed",
        "owner_changed",
        "missing_http",
        "duplicate_preview",
        "foreign_initial",
        "accepted",
        "child_rendered",
        "wrong_markers",
        "ready_claim",
        "accept_enabled",
        "undo_disabled",
        "wrong_metrics",
        "wrong_timeline",
        "hidden_failure",
        "raw_token",
        "missing_verifier",
        "wrong_token_algorithm",
        "redacted_claim",
    ],
)
def test_real_eligible_child_fault_is_distinct_from_ineligibility_or_false_success(kind, fault):
    value, boundary, records, persisted = case(kind)
    proposal = boundary["canonical"]["proposal"]
    if fault == "not_evaluated":
        proposal["runtime_execution"]["evaluation_executed"] = False
    elif fault == "ineligible":
        proposal["runtime_execution"]["evaluation_certificate"]["comparison_eligibility"] = "ineligible"
    elif fault == "child_hash":
        proposal["runtime_execution"]["child_plan"]["content_hash"] = "foreign"
    elif fault == "certificate_hash":
        proposal["certificate_content_hash"] = "foreign"
    elif fault == "diff_child":
        proposal["runtime_execution"]["plan_diff"]["child_plan_id"] = "foreign"
    elif fault == "no_real_geometry":
        proposal["geography_plan"]["validated_legs"]["features"] = []
    elif fault == "wrong_replacement":
        proposal["geography_plan"]["stops"]["features"][0]["properties"]["stop_id"] = "bixby_creek_bridge_viewpoint"
    elif fault == "wire_extra_change":
        boundary["wire"]["session"]["draft"] = []
    elif fault == "wire_not_faulted":
        boundary["wire"] = deepcopy(boundary["canonical"])
    elif fault in {"mutated_persistence", "public_shape_as_persistence"}:
        stored = json.loads(persisted["utf8"])
        if fault == "mutated_persistence":
            stored["session"]["proposal"] = deepcopy(boundary["wire"]["proposal"])
        else:
            stored["session"] = deepcopy(boundary["canonical"]["session"])
        persisted["utf8"] = json.dumps(stored)
        persisted["sha256"] = hashlib.sha256(persisted["utf8"].encode()).hexdigest()
    elif fault == "draft_lost":
        boundary["canonical"]["session"]["draft"] = []
    elif fault == "parent_changed":
        boundary["canonical"]["session"]["accepted_plan_id"] = proposal["child_plan_id"]
    elif fault == "owner_changed":
        boundary["canonical"]["session"]["trip_id"] = "foreign"
    elif fault == "missing_http":
        records.pop()
    elif fault == "duplicate_preview":
        records.append(deepcopy(records[-1]))
    elif fault == "foreign_initial":
        value["initial"]["workspace"]["accepted_plan_pointer"]["accepted_plan_id"] = "foreign"
    elif fault == "accepted":
        records.append({"url": verifier.BASE_URL + "/api/sessions/session/accept", "response_body_utf8": "{}"})
    elif fault == "child_rendered":
        value["checkpoints"]["blocked_parent"]["rendered_map_plan_id"] = proposal["child_plan_id"]
    elif fault == "wrong_markers":
        value["checkpoints"]["blocked_parent"]["rendered_marker_ids"] = ["pfeiffer_big_sur_state_park"]
    elif fault == "ready_claim":
        value["checkpoints"]["blocked_parent"]["preview_state"] = "PREVIEW_RENDERED"
    elif fault == "accept_enabled":
        value["ui"]["accept_disabled"] = False
    elif fault == "undo_disabled":
        value["ui"]["undo_disabled"] = True
    elif fault == "wrong_metrics":
        value["ui"]["metric"]["planId"] = proposal["child_plan_id"]
    elif fault == "wrong_timeline":
        value["ui"]["timeline"] = "Pfeiffer Big Sur State Park"
    elif fault == "hidden_failure":
        value["ui"]["toast"] = "Preview ready"
    elif fault in {"raw_token", "missing_verifier", "wrong_token_algorithm"}:
        stored = json.loads(persisted["utf8"])
        if fault == "raw_token":
            stored["mutation_token"] = "unit-only-plaintext"
        elif fault == "missing_verifier":
            stored.pop("token_verifier")
        else:
            stored["token_verifier"] = {"algorithm": "plaintext"}
        persisted["utf8"] = json.dumps(stored)
        persisted["sha256"] = hashlib.sha256(persisted["utf8"].encode()).hexdigest()
    elif fault == "redacted_claim":
        persisted["credential_redactions"] = ["/mutation_token"]
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.semantic_contract(value, boundary, records, persisted)
    else:
        result = verifier.semantic_contract(value, boundary, records, persisted)
        assert result["kind"] == kind and result["state"] == "PREVIEW_FAILED" and result["staged_draft_retained"]


@pytest.mark.parametrize("kind", verifier.CASES)
@pytest.mark.parametrize("fault", [None, "other_edit", "evaluation_mock", "wrong_origin", "wrong_hash"])
def test_fault_patch_is_only_the_declared_post_publication_geometry_change(kind, fault):
    original = (ROOT / verifier.RELATIVE).read_bytes()
    modified = browser.fault_source(original.decode(), kind)
    value = {
        "schema_version": "preview-geography-response-fault-v1",
        "kind": kind,
        "path": verifier.RELATIVE,
        "before_utf8": original.decode(),
        "after_utf8": modified,
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "production_source_modified": False,
        "frontend_fetch_mocking": False,
        "optimizer_mocking": False,
        "evaluator_mocking": False,
        "canonical_artifacts_mutated": False,
    }
    if fault == "other_edit":
        value["after_utf8"] += "\n# unrelated edit\n"
        value["after_sha256"] = hashlib.sha256(value["after_utf8"].encode()).hexdigest()
    elif fault == "evaluation_mock":
        value["evaluator_mocking"] = True
    elif fault == "wrong_origin":
        value["before_utf8"] += "\n"
    elif fault == "wrong_hash":
        value["after_sha256"] = "f" * 64
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.patch_contract(value, original)
    else:
        verifier.patch_contract(value, original)


@pytest.mark.parametrize(
    "fault", [None, "missing_response", "failed", "unfinished", "wrong_phase", "outside_lifecycle", "wrong_hash"]
)
def test_raw_request_universe_rejects_orphans_failures_and_unfinished_attempts(fault):
    url = verifier.BASE_URL + "/api/sessions/session/preview"
    requests = [
        {
            "sequence": 1,
            "phase": "generation_b",
            "method": "POST",
            "url": url,
            "started_at": "2026-09-08T10:00:01Z",
            "response_at": "2026-09-08T10:00:02Z",
            "finished_at": "2026-09-08T10:00:03Z",
            "failed_at": None,
            "failure": None,
            "status": 200,
            "http_file": "http_001.json",
            "request_body_sha256": "a" * 64,
        }
    ]
    records = [
        {
            "request_sequence": 1,
            "phase": "generation_b",
            "method": "POST",
            "url": url,
            "request_original_body_sha256": "a" * 64,
            "response_status_code": 200,
        }
    ]
    witnesses = [{"installed_at": "2026-09-08T10:00:00Z", "finalized_at": "2026-09-08T10:00:04Z"}] * 2
    if fault == "missing_response":
        records = []
    elif fault == "failed":
        requests[0]["failure"] = "net::ERR_FAILED"
    elif fault == "unfinished":
        requests[0]["finished_at"] = "2026-09-08T10:00:01Z"
    elif fault == "wrong_phase":
        requests[0]["phase"] = "invented"
    elif fault == "outside_lifecycle":
        requests[0]["started_at"] = "2026-09-08T09:00:00Z"
    elif fault == "wrong_hash":
        requests[0]["request_body_sha256"] = "b" * 64
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.raw_http_contract(requests, records, witnesses)
    else:
        verifier.raw_http_contract(requests, records, witnesses)


def test_geometry_gate_registration_and_missing_receipts_remain_fail_closed():
    spec = runner.MODES["preview-geography-fault"]
    assert spec["scenario"] == "preview_geography_fault"
    suite = next(row for row in aggregate.SUITES if row[0] == "preview-geography-fault")
    assert suite[2] == tuple(verifier.TEST + "[" + kind + "]" for kind in verifier.CASES)
    assert aggregate.EXPECTED_TEST_COUNT == 72 and len(aggregate.SUITES) == 23
    assert {
        "scripts/verify_e2e1_preview_geography_fault.py",
        "tests/product_app/browser/build_coherence_fixture.py",
        "tests/product_app/browser/test_optional_evidence_fault_live.py",
    } <= set(deterministic_source_inputs(spec["browser_test"]))
    result = coverage.gate_report(None, [], {"unverified_requirements": [coverage.GEOGRAPHY_GATE]}, [])
    assert result["requirements"][0]["passed"] is False
    assert result["requirements"][0]["reason"] == "independently_passed_suite_missing"
