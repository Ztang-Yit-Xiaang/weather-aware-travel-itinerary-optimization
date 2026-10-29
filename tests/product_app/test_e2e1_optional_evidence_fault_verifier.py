"""Portable adversarial unit contracts; no real-browser or provider-pass claims."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_e2e1_aggregate as aggregate  # noqa: E402
import run_e2e1_self_managed as runner  # noqa: E402
import verify_e2e1_optional_evidence_fault as verifier  # noqa: E402
import verify_e2e1_self_managed as managed  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402


def hashed(value):
    return {**value, "content_hash": core._artifact_content_hash(value)}


def geo(plan, stop, role="draft_preview"):
    feature = {
        "type": "Feature",
        "properties": {
            "role": role,
            "day": 5,
            "origin_id": "start",
            "destination_id": stop,
            "evidence_scope": "runtime_optimizer_route_leg",
        },
        "geometry": {"type": "LineString", "coordinates": [[-121, 36], [-120, 35]]},
    }
    return {
        "plan_id": plan["plan_id"],
        "content_hash": plan["content_hash"],
        "role": role,
        "label": "Unit preview",
        "validated_legs": {"type": "FeatureCollection", "features": [feature]},
        "stops": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"stop_id": stop, "role": role},
                    "geometry": {"type": "Point", "coordinates": [-120, 35]},
                }
            ],
        },
        "route_path": {"type": "FeatureCollection", "features": []},
        "gaps": {"type": "FeatureCollection", "features": []},
    }


def ack(plan, state="PARENT_RENDERED"):
    return {
        "active_plan_id": plan["plan_id"],
        "rendered_map_plan_id": plan["plan_id"],
        "active_plan_content_hash": plan["content_hash"],
        "preview_state": state,
        "rendered_geometry_hash": core._completed_race_geometry_hash(plan),
        "rendered_marker_ids": [row["properties"]["stop_id"] for row in plan["stops"]["features"]],
    }


def fixture():
    parent = hashed({"plan_id": "plan_parent"})
    child = hashed({"plan_id": "plan_child", "parent_plan_id": "plan_parent", "planning_request_id": "request"})
    certificate = hashed(
        {
            "plan_id": child["plan_id"],
            "plan_content_hash": child["content_hash"],
            "certificate_id": "certificate",
            "comparison_eligibility": "eligible",
        }
    )
    diff = hashed({"parent_plan_id": "plan_parent", "child_plan_id": "plan_child", "diff_id": "diff"})
    proposal = {
        "state": "eligible",
        "execution_mode": "deterministic_repair_pipeline",
        "child_plan_id": "plan_child",
        "child_plan_content_hash": child["content_hash"],
        "parent_plan_id": "plan_parent",
        "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"],
        "diff_id": "diff",
        "diff_content_hash": diff["content_hash"],
        "geography_plan": geo(child, "pfeiffer_big_sur_state_park"),
        "runtime_execution": {
            "evaluation_executed": True,
            "child_plan": child,
            "evaluation_certificate": certificate,
            "plan_diff": diff,
            "request_id": "request",
            "repair_request": {"request_id": "request"},
        },
    }
    accepted_geo = verifier.accepted_geography(proposal)
    original_geo = geo(parent, "bixby_creek_bridge_viewpoint", "original")
    optional = geo(hashed({"plan_id": "plan_optional"}), "optional_stop", "alternative")
    snapshot = {
        "schema_version": "acceptance-snapshot-v1",
        "decision_id": "decision",
        "trip_id": "trip",
        "run_id": "run",
        "plan": child,
        "diff": diff,
        "certificate": certificate,
        "geography_plan": accepted_geo,
        "route_legs": [
            {
                key: accepted_geo["validated_legs"]["features"][0]["properties"][key]
                for key in ("day", "origin_id", "destination_id", "evidence_scope")
            }
        ],
        "proposal_identity": {
            key: proposal.get(key)
            for key in (
                "proposal_id",
                "draft_content_hash",
                "diff_id",
                "diff_content_hash",
                "certificate_id",
                "certificate_content_hash",
            )
        },
    }
    snapshot["content_hash"] = verifier.acceptance_hash(snapshot)
    pointer = {
        "schema_version": "workspace-pointer-v1",
        "accepted_plan_id": "plan_child",
        "accepted_plan_content_hash": child["content_hash"],
        "original_plan_id": "plan_parent",
        "original_plan_content_hash": parent["content_hash"],
        "decision_id": "decision",
        "trip_id": "trip",
        "run_id": "run",
        "revision": 1,
    }
    pointer["content_hash"] = verifier.acceptance_hash(pointer)
    decision = {
        "decision_id": "decision",
        "selected_plan_id": "plan_child",
        "selected_plan_content_hash": child["content_hash"],
        "previous_plan_id": "plan_parent",
        "certificate_id": "certificate",
        "idempotency_key": "accepted_once",
        "snapshot_content_hash": snapshot["content_hash"],
    }
    decision["content_hash"] = verifier.acceptance_hash(decision)
    session = {
        "session_id": "session",
        "trip_id": "trip",
        "run_id": "run",
        "accepted_plan_id": "plan_child",
        "draft": [],
        "revision": 5,
        "selected_alternative_id": None,
    }
    workspace = {
        "accepted_plan_pointer": {
            key: pointer[key]
            for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
        },
        "geography": {"plans": [original_geo, accepted_geo, optional]},
        "alternatives": [{"plan_id": "plan_optional", "role": "low_driving"}],
        "alternative_evidence": {
            "plan_optional": {"evidence": "unit optional"},
            "plan_child": {"certificate": certificate},
        },
    }
    accepted = {
        "session": deepcopy(session),
        "workspace": deepcopy(workspace),
        "result": {"pointer": pointer, "decision": decision},
    }
    before = {"session": deepcopy(session), "workspace": deepcopy(workspace)}
    after = deepcopy(before)
    del after["workspace"]["alternative_evidence"]["plan_optional"]
    selected = {"session": {**session, "selected_alternative_id": "plan_optional", "revision": 6}}
    maps = [
        {
            "renderedMapPlanId": plan["plan_id"],
            "mapReady": "true",
            "mapTileCoverage": "1",
            "renderedPlanContentHash": plan["content_hash"],
            "renderedGeometryHash": core._completed_race_geometry_hash(plan),
            "renderedMarkerIds": json.dumps(ack(plan)["rendered_marker_ids"]),
        }
        for plan in (accepted_geo, optional)
    ]
    value = {
        "schema_version": "optional-evidence-fault-browser-v1",
        "context": "accepted_child",
        "coverage_category": verifier.CATEGORY,
        "natural_interaction_claimed": False,
        "assertions_passed": True,
        "initial": before,
        "faulted": after,
        "acceptance": accepted,
        "proposal": proposal,
        "selected": selected,
        "optional_plan_id": "plan_optional",
        "compare_after_back": maps,
        "compare_labels": {
            "baseline_heading": "Current trip route",
            "baseline_card": "Current accepted trip",
            "baseline_plan_id": "plan_child",
            "original_plan_id": "plan_parent",
        },
        "unavailable": {
            "path": verifier.BASE_URL + "/app/evidence",
            "heading": "Evidence unavailable",
            "code": "compare_option_evidence_unavailable",
            "back_visible": True,
            "back_disabled": False,
            "evidence_panel_count": 0,
            "html": '<button id="evidence-back">Back to Compare</button>',
        },
        "checkpoints": {
            "initial_parent": ack(original_geo),
            "evaluated_child": ack(proposal["geography_plan"], "PREVIEW_RENDERED"),
            **{
                label: ack(accepted_geo)
                for label in ("accepted_child", "accepted_before_fault", "accepted_during_fault", "accepted_after_back")
            },
        },
    }
    events = [
        {
            "trusted": True,
            "type": "click",
            "visible": True,
            "disabled": False,
            "at": index + 1,
            "path": path,
            "route": route,
            "id": "evidence-back" if index == 3 else "",
            "option": "plan_optional" if index == 1 else None,
            "label": "Back to Compare" if index == 3 else "Unit",
        }
        for index, (path, route) in enumerate(
            [
                ("/app/itinerary", "/app/compare"),
                ("/app/compare", None),
                ("/app/compare", "/app/evidence"),
                ("/app/evidence", None),
                ("/app/compare", "/app/itinerary"),
            ]
        )
    ]

    def wire(suffix, payload, phase="generation_a", method="POST", request=None):
        return {
            "url": verifier.BASE_URL + "/api/sessions/session" + suffix,
            "method": method,
            "phase": phase,
            "response_body_utf8": json.dumps(payload),
            "request_body_utf8": json.dumps(request or {}),
        }

    records = [
        wire("/preview", {"proposal": proposal}),
        wire(
            "/accept", accepted, request={"confirmation": "accept_evaluated_repair", "idempotency_key": "accepted_once"}
        ),
        wire("", before, method="GET"),
        wire("", after, "generation_b_reload", "GET"),
        wire(
            "/selection",
            selected,
            "select_optional",
            request={"selected_alternative_id": "plan_optional", "expected_revision": 5},
        ),
    ]
    trip = hashlib.sha256(b"trip").hexdigest()[:32]
    documents = {
        "workspaces/" + trip + ".json": pointer,
        "decisions/" + trip + "/ledger.jsonl": decision,
        "plan_repository/plan_child.json": child,
        "plan_repository/plan_parent.json": parent,
        "acceptance_snapshots/decision.json": snapshot,
    }
    states = {}
    for name, item in documents.items():
        body = json.dumps(item).encode()
        states[name] = {"utf8": body.decode(), "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}
    return (
        value,
        events,
        records,
        {"target_plan_id": "plan_optional"},
        {"before": deepcopy(states), "after": deepcopy(states)},
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "natural_claim",
        "wrong_context",
        "missing_evidence_both",
        "accepted_evidence_removed",
        "pointer_hash",
        "foreign_accepted_hash",
        "missing_http",
        "fake_click",
        "duplicate_click",
        "wrong_back_path",
        "wrong_option",
        "map_not_ready",
        "map_wrong_geometry",
        "map_duplicate_marker",
        "different_accepted_map",
        "second_accept",
        "chat_call",
    ],
)
def test_exact_fault_receipt_rejects_false_positive_branches(fault):
    value, events, records, patch, _ = fixture()
    if fault == "natural_claim":
        value["natural_interaction_claimed"] = True
    elif fault == "wrong_context":
        value["context"] = "original"
    elif fault == "missing_evidence_both":
        value["initial"]["workspace"]["alternative_evidence"].pop("plan_optional")
    elif fault == "accepted_evidence_removed":
        value["faulted"]["workspace"]["alternative_evidence"].pop("plan_child")
    elif fault == "pointer_hash":
        value["acceptance"]["result"]["pointer"]["content_hash"] = "0" * 64
    elif fault == "foreign_accepted_hash":
        value["faulted"]["workspace"]["accepted_plan_pointer"]["accepted_plan_content_hash"] = "foreign"
    elif fault == "missing_http":
        records.pop()
    elif fault == "fake_click":
        events[3]["trusted"] = False
    elif fault == "duplicate_click":
        events.append(deepcopy(events[-1]))
    elif fault == "wrong_back_path":
        events[3]["path"] = "/app/itinerary"
    elif fault == "wrong_option":
        value["optional_plan_id"] = "plan_child"
    elif fault == "map_not_ready":
        value["compare_after_back"][0]["mapReady"] = "false"
    elif fault == "map_wrong_geometry":
        value["compare_after_back"][1]["renderedGeometryHash"] = "0" * 64
    elif fault == "map_duplicate_marker":
        value["compare_after_back"][0]["renderedMarkerIds"] = (
            '["pfeiffer_big_sur_state_park","pfeiffer_big_sur_state_park"]'
        )
    elif fault == "different_accepted_map":
        value["checkpoints"]["accepted_during_fault"]["active_plan_id"] = "other"
    elif fault == "second_accept":
        records.append(deepcopy(records[1]))
    elif fault == "chat_call":
        records.append(
            {
                "url": verifier.BASE_URL + "/api/sessions/session/copilot/messages",
                "method": "POST",
                "phase": "generation_b_reload",
            }
        )
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.semantic_contract(value, events, records, patch)
    else:
        assert verifier.semantic_contract(value, events, records, patch)["coverage_category"] == verifier.CATEGORY


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "changed_file",
        "missing_pointer",
        "missing_snapshot",
        "wrong_snapshot_shape",
        "forged_bytes",
        "wrong_route",
        "wrong_identity",
    ],
)
def test_actual_snapshot_schema_and_exact_protected_files(fault):
    value, _, _, _, state = fixture()
    if fault == "changed_file":
        state["after"].pop(next(iter(state["after"])))
    elif fault in {"missing_pointer", "missing_snapshot"}:
        prefix = "workspaces/" if fault == "missing_pointer" else "acceptance_snapshots/"
        key = next(name for name in state["before"] if name.startswith(prefix))
        del state["before"][key]
        del state["after"][key]
    elif fault in {"wrong_snapshot_shape", "wrong_route", "wrong_identity"}:
        row = state["before"]["acceptance_snapshots/decision.json"]
        payload = json.loads(row["utf8"])
        if fault == "wrong_snapshot_shape":
            payload["runtime_execution"] = payload.pop("plan")
        elif fault == "wrong_route":
            payload["route_legs"][0]["destination_id"] = "foreign"
        else:
            payload["proposal_identity"]["certificate_id"] = "foreign"
        payload["content_hash"] = verifier.acceptance_hash(payload)
        body = json.dumps(payload).encode()
        state["before"]["acceptance_snapshots/decision.json"] = {
            "utf8": body.decode(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
        }
        state["after"] = deepcopy(state["before"])
    elif fault == "forged_bytes":
        state["before"][next(iter(state["before"]))]["sha256"] = "0" * 64
        state["after"] = deepcopy(state["before"])
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.protected_contract(state, value)
    else:
        verifier.protected_contract(state, value)


@pytest.mark.parametrize(
    "fault", [None, "different_source", "unrelated_edit", "accepted_target", "frontend_mock", "wrong_hash"]
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_fault_patch_changes_only_optional_backend_projection(fault, newline):
    original = (core.ROOT / verifier.RELATIVE).read_text(encoding="utf-8").replace("\n", newline).encode()
    after = original.decode().replace(verifier.ANCHOR.replace("\n", newline), verifier.FAULT.replace("\n", newline))
    row = {
        "schema_version": "optional-evidence-projection-fault-v1",
        "path": verifier.RELATIVE,
        "before_utf8": original.decode(),
        "after_utf8": after,
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "target_role": "low_driving",
        "production_source_modified": False,
        "client_state_injection": False,
        "fetch_mocking": False,
    }
    if fault == "different_source":
        row["before_utf8"] += "\n"
    elif fault == "unrelated_edit":
        row["after_utf8"] += "\n"
    elif fault == "accepted_target":
        row["target_role"] = "accepted"
    elif fault == "frontend_mock":
        row["fetch_mocking"] = True
    elif fault == "wrong_hash":
        row["after_sha256"] = "0" * 64
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.patch_contract(row, original)
    else:
        verifier.patch_contract(row, original)


def test_fault_registration_has_single_case_and_both_guarded_generations(tmp_path, monkeypatch):
    suite = next(row for row in aggregate.SUITES if row[0] == "optional-evidence-fault")
    assert suite[2] == (verifier.TEST,) and aggregate.EXPECTED_TEST_COUNT == 72 and len(aggregate.SUITES) == 23
    assert runner.MODES[suite[0]]["scenario"] == "optional_evidence_fault"
    (tmp_path / "pytest_junit.xml").write_text(
        '<testsuite><testcase classname="tests.product_app.browser.test_optional_evidence_fault_live" name="'
        + verifier.TEST
        + '"/></testsuite>'
    )
    assert managed._junit(tmp_path, suite[0]) == {"tests/product_app/browser/" + suite[1] + "::" + verifier.TEST}
    sources = set(deterministic_source_inputs("tests/product_app/browser/" + suite[1]))
    assert {
        "scripts/verify_e2e1_optional_evidence_fault.py",
        "scripts/verify_e2e1_bootstrap_recovery.py",
        "tests/product_app/browser/build_coherence_fixture.py",
        "tests/product_app/browser/test_e2e1_live_copilot_chat.py",
    } <= sources
    (tmp_path / "test_process.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        managed, "_lifecycle", lambda base, identity, process, **kw: (calls.append((base, identity, kw)), {})[1]
    )
    assert (
        len(
            aggregate.child_lifecycles(
                tmp_path,
                {"schema_version": "e2e1-self-managed-browser-manifest-v1", "mode": suite[0], "run_id": "unit"},
            )
        )
        == 2
    )
    assert [call[1] for call in calls] == ["unit_generation_a", "unit_generation_b"]
    assert all(call[2] == {"child_process": True} for call in calls)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_hash",
        "wrong_owner",
        "certified",
        "nonempty_draft",
        "wrong_revision",
        "repair_endpoint",
        "wrong_method",
    ],
)
def test_read_only_impact_post_cannot_hide_draft_or_plan_mutation(fault):
    value, _, _, _, _ = fixture()
    session = value["initial"]["session"]
    pointer = value["acceptance"]["result"]["pointer"]
    response = {
        "schema_version": "product-draft-impact-preview-v1",
        "session_id": session["session_id"],
        "session_revision": session["revision"],
        "parent_plan_id": pointer["accepted_plan_id"],
        "parent_plan_content_hash": pointer["accepted_plan_content_hash"],
        "certification_status": "not_certified",
        "is_certified": False,
        "operations": [],
        "summary": {
            "operation_count": 0,
            "evaluated_executable_count": 0,
            "draft_only_count": 0,
            "can_run_evaluated_preview": False,
            "blocking_codes": [],
        },
    }
    row = {
        "method": "POST",
        "phase": "generation_b_reload",
        "url": verifier.BASE_URL + "/api/sessions/session/draft/impact-preview",
        "request_body_utf8": json.dumps({"expected_revision": session["revision"]}),
    }
    if fault == "wrong_hash":
        response["parent_plan_content_hash"] = "foreign"
    elif fault == "wrong_owner":
        response["session_id"] = "foreign"
    elif fault == "certified":
        response["is_certified"] = True
    elif fault == "nonempty_draft":
        response["operations"] = [{"type": "replace_stop"}]
    elif fault == "wrong_revision":
        response["session_revision"] += 1
    elif fault == "repair_endpoint":
        row["url"] = row["url"].replace("/draft/impact-preview", "/preview")
    elif fault == "wrong_method":
        row["method"] = "DELETE"
    row["response_body_utf8"] = json.dumps(response)
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.post_restart_contract([row], session, pointer)
    else:
        verifier.post_restart_contract([row], session, pointer)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "orphan",
        "extra_http",
        "body",
        "phase",
        "sequence",
        "unfinished",
        "failed",
        "late",
        "status",
        "non_api_write",
    ],
)
def test_raw_request_universe_rejects_missing_or_disconnected_responses(fault):
    lifecycle = {"installed_at": "2026-09-08T10:00:00Z", "finalized_at": "2026-09-08T10:00:10Z"}
    row = {
        "sequence": 1,
        "phase": "generation_b_reload",
        "method": "GET",
        "url": verifier.BASE_URL + "/api/health",
        "started_at": "2026-09-08T10:00:01Z",
        "response_at": "2026-09-08T10:00:02Z",
        "finished_at": "2026-09-08T10:00:03Z",
        "failed_at": None,
        "failure": None,
        "status": 200,
        "http_file": "http_001.json",
        "request_body_sha256": hashlib.sha256(b"").hexdigest(),
    }
    record = {
        "request_sequence": 1,
        "phase": row["phase"],
        "method": "GET",
        "url": row["url"],
        "response_status_code": 200,
        "request_original_body_sha256": row["request_body_sha256"],
    }
    records = [record]
    if fault == "orphan":
        records = []
    elif fault == "extra_http":
        records.append({**record, "request_sequence": 2})
    elif fault == "body":
        row["request_body_sha256"] = "0" * 64
    elif fault == "phase":
        row["phase"] = "unobserved"
    elif fault == "sequence":
        row["sequence"] = 2
    elif fault == "unfinished":
        row["finished_at"] = None
    elif fault == "failed":
        row["failure"] = "net::ERR_FAILED"
    elif fault == "late":
        row["response_at"] = "2026-09-08T11:00:02Z"
    elif fault == "status":
        row["status"] = 500
    elif fault == "non_api_write":
        row.update(url=verifier.BASE_URL + "/static/app.js", method="POST", http_file=None)
        records = []
    if fault:
        with pytest.raises((core.E2E1VerificationError, TypeError, ValueError)):
            verifier.raw_http_contract([row], records, [lifecycle, lifecycle])
    else:
        verifier.raw_http_contract([row], records, [lifecycle, lifecycle])


def test_fault_projection_is_explicit_and_never_credits_natural_or_original_branch():
    import e2e1_runtime_gate_coverage as coverage

    value, events, records, patch, _ = fixture()
    base = "optional_evidence_fault/"
    data = {
        base + "browser_assertions.json": value,
        base + "control_inputs.json": events,
        base + "fault_patch.json": patch,
        base + "http_index.json": [{"file": f"http_{index}.json"} for index in range(len(records))],
        **{base + f"http_{index}.json": row for index, row in enumerate(records)},
    }

    class UnitReceipts:
        def __init__(self):
            self.used = []

        def read(self, mode, name):
            if mode != "optional-evidence-fault" or name not in data:
                raise coverage.ReceiptError("unit_scope_missing")
            self.used.append({"mode": mode, "artifact": name})
            return data[name]

    proofs, _ = coverage.existing_control_proofs(UnitReceipts())
    detail = proofs[("evidence_back_unavailable", "accepted_child")]
    assert detail["coverage_category"] == verifier.CATEGORY and detail["natural_interaction_claimed"] is False
    assert ("evidence_back_unavailable", "original") not in proofs
    value["natural_interaction_claimed"] = True
    proofs, _ = coverage.existing_control_proofs(UnitReceipts())
    assert ("evidence_back_unavailable", "accepted_child") not in proofs


def test_old_original_wording_cannot_describe_accepted_baseline():
    value, events, records, patch, _ = fixture()
    value["compare_labels"]["baseline_heading"] = "Original route"
    with pytest.raises(core.E2E1VerificationError, match="compare_baseline_label_misleading"):
        verifier.semantic_contract(value, events, records, patch)
