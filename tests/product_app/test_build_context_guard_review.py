"""Portable unit-scope adversarial review of actual build identity/reload guards.

Only the identity/reload guard section is executed on small synthetic contracts.
These tests never call full verification, launch processes, claim screenshots,
or portray the synthetic contracts as optimizer or browser execution evidence.
The earlier real-capture red/green audit is retained separately in results.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_e2e1_self_managed as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def _guard_code():
    function = ast.parse(inspect.getsource(verifier._build_case)).body[0]
    assignments = {
        node.targets[0].id: index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["previews"] < assignments["browser_errors"]
    statements = function.body[assignments["previews"] : assignments["browser_errors"]]
    return compile(ast.Module(body=statements, type_ignores=[]), str(Path(verifier.__file__)), "exec")


def _hashed(value, *, short=True):
    result = deepcopy(value)
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    result["content_hash"] = hashlib.sha256(body).hexdigest()[: 16 if short else 64]
    return result


def _wire(payload, method, path, phase):
    request, response = ("{}" if method == "POST" else ""), json.dumps(payload, separators=(",", ":"))
    result = {
        "schema_version": "e2e1-product-http-io-v1",
        "phase": phase,
        "method": method,
        "url": "http://127.0.0.1:8127" + path,
        "response_status_code": 200,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
        "credential_redactions": {"request": [], "response": []},
        "body_capture": "exact_utf8",
    }
    for name, body in (("request", request), ("response", response)):
        result[name + "_body_utf8"] = body
        result[name + "_body_sha256"] = result[name + "_original_body_sha256"] = hashlib.sha256(
            body.encode()
        ).hexdigest()
    return result


def _geography(plan, stop_id, role):
    return {
        "plan_id": plan["plan_id"],
        "content_hash": plan["content_hash"],
        "role": role,
        "label": "Synthetic unit plan",
        "stops": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"stop_id": stop_id},
                    "geometry": {"type": "Point", "coordinates": [-121.9, 36.2]},
                }
            ],
        },
        "validated_legs": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"leg_id": "unit-leg", "role": role},
                    "geometry": {"type": "LineString", "coordinates": [[-122.0, 36.3], [-121.9, 36.2]]},
                }
            ],
        },
    }


def _ack(geography, state="PARENT_RENDERED"):
    legs = deepcopy(geography["validated_legs"])
    for feature in legs["features"]:
        feature["properties"].pop("role")
        feature["geometry"]["coordinates"] = [
            [int(value) if value.is_integer() else value for value in point]
            for point in feature["geometry"]["coordinates"]
        ]
    return {
        "active_plan_id": geography["plan_id"],
        "rendered_map_plan_id": geography["plan_id"],
        "active_plan_content_hash": geography["content_hash"],
        "preview_state": state,
        "rendered_marker_ids": [feature["properties"]["stop_id"] for feature in geography["stops"]["features"]],
        "rendered_geometry_hash": hashlib.sha256(
            json.dumps(legs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@pytest.fixture(params=["original", "accepted_child"])
def capture(request, tmp_path):
    context = request.param
    destination = tmp_path / context
    destination.mkdir()
    parent = _hashed({"plan_id": "unit-parent", "selected_stops": [{"stop_id": "unit-original"}]})
    parent_geography = _geography(parent, "unit-original", "original")
    session = {
        "session_id": "unit-session",
        "trip_id": "unit-trip",
        "run_id": "unit-run",
        "draft": [],
        "accepted_plan_id": parent["plan_id"],
    }
    created = {"session": deepcopy(session), "workspace": {"geography": {"plans": [parent_geography]}}}
    records = [_wire(created, "POST", "/api/sessions", "generation_a")]
    baseline = deepcopy(created)
    plan, geography, decision, certificate = parent, parent_geography, None, None
    checkpoints = {"initial_parent": _ack(parent_geography)}
    if context == "accepted_child":
        plan = _hashed(
            {
                "plan_id": "unit-child",
                "parent_plan_id": parent["plan_id"],
                "planning_request_id": "unit-repair-request",
                "selected_stops": [{"stop_id": "unit-replacement"}],
            }
        )
        certificate = _hashed(
            {
                "certificate_id": "unit-certificate",
                "plan_id": plan["plan_id"],
                "plan_content_hash": plan["content_hash"],
                "comparison_eligibility": "eligible",
            }
        )
        diff = _hashed({"diff_id": "unit-diff", "parent_plan_id": parent["plan_id"], "child_plan_id": plan["plan_id"]})
        geography = _geography(plan, "unit-replacement", "accepted")
        proposal = {
            "state": "eligible",
            "execution_mode": "deterministic_repair_pipeline",
            "parent_plan_id": parent["plan_id"],
            "child_plan_id": plan["plan_id"],
            "child_plan_content_hash": plan["content_hash"],
            "certificate_id": certificate["certificate_id"],
            "certificate_content_hash": certificate["content_hash"],
            "diff_content_hash": diff["content_hash"],
            "geography_plan": geography,
            "runtime_execution": {
                "evaluation_executed": True,
                "child_plan": plan,
                "evaluation_certificate": certificate,
                "plan_diff": diff,
                "request_id": "unit-repair-request",
                "repair_request": {"request_id": "unit-repair-request"},
            },
        }
        decision = _hashed(
            {
                "decision_id": "unit-decision",
                "selected_plan_id": plan["plan_id"],
                "selected_plan_content_hash": plan["content_hash"],
                "certificate_id": certificate["certificate_id"],
            },
            short=False,
        )
        session["accepted_plan_id"] = plan["plan_id"]
        baseline = {
            "session": deepcopy(session),
            "workspace": {"geography": {"plans": [parent_geography, geography]}},
            "result": {"decision": decision},
        }
        records.extend(
            [
                _wire({"proposal": proposal}, "POST", "/api/sessions/unit-session/preview", "generation_a"),
                _wire(baseline, "POST", "/api/sessions/unit-session/accept", "generation_a"),
            ]
        )
        checkpoints.update(evaluated_child=_ack(geography, "PREVIEW_RENDERED"), accepted_child=_ack(geography))
    restored = deepcopy(baseline)
    restored["workspace"]["accepted_plan_pointer"] = {
        "accepted_plan_id": plan["plan_id"],
        "accepted_plan_content_hash": plan["content_hash"],
        "decision_id": decision["decision_id"] if decision else None,
    }
    records.append(_wire(restored, "GET", "/api/sessions/unit-session", "generation_b_reloaded"))
    checkpoints["active_plan_after_update"] = _ack(geography)
    assertions = {
        "accepted_plan_id": plan["plan_id"],
        "accepted_plan_content_hash": plan["content_hash"],
        "session": {"session_id": session["session_id"]},
        "first_build_id": "unit-build-a",
        "certificate_id": certificate["certificate_id"] if certificate else None,
        "decision_id": decision["decision_id"] if decision else None,
    }
    sidecars = {
        "map_checkpoints.json": checkpoints,
        "reload_control_input.json": {
            "trusted": True,
            "selector": "#reload-current-build",
            "client_build": "unit-build-a",
            "path": "/app/itinerary",
            "title": "Unit recovery",
            "clicked_at": "2026-09-08T12:00:02Z",
        },
        "product_client_build_mismatch_visible_recovery.json": {
            "text": "Unit recovery",
            "captured_at": "2026-09-08T12:00:01Z",
        },
        "generation_b/active_plan_after_update.readiness.json": {"screenshot_started_at": "2026-09-08T12:00:03Z"},
    }
    for relative, value in sidecars.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
    return {"base": destination, "context": context, "assertions": assertions, "records": records}


def _check(capture):
    for row in capture["records"]:
        core._verify_product_http_capture(row)
    exec(_guard_code(), {**vars(verifier), **capture, "identity": capture["assertions"]["session"]})


def _sidecar(capture, filename, edit):
    path = capture["base"] / filename
    value = core._load(path)
    edit(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _restore_mutation(capture, edit):
    row = next(
        row
        for row in capture["records"]
        if row["phase"] == "generation_b_reloaded"
        and row["method"] == "GET"
        and row["url"].endswith("/api/sessions/" + capture["assertions"]["session"]["session_id"])
    )
    before_hash = row["response_body_sha256"]
    value = json.loads(row["response_body_utf8"])
    edit(value)
    row["response_body_utf8"] = json.dumps(value, separators=(",", ":"))
    row["response_original_body_sha256"] = row["response_body_sha256"] = hashlib.sha256(
        row["response_body_utf8"].encode()
    ).hexdigest()
    core._verify_product_http_capture(row)
    (capture["base"] / "adversarial_wire.json").write_text(
        json.dumps(
            {
                "scope": "synthetic unit mutation; not a browser response",
                "source_response_sha256": before_hash,
                "mutated_record": row,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_synthetic_original_and_accepted_identity_reload_sections_pass(capture):
    _check(capture)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_plan",
        "wrong_hash",
        "wrong_decision",
        "untrusted_reload",
        "reload_before_recovery",
        "restored_session",
        "restored_trip",
        "restored_run",
        "restored_pointer_plan",
        "restored_pointer_hash",
        "restored_plan_hash",
        "restored_plan_geometry",
        "restored_plan_metadata",
        "duplicate_restored_plan",
        "restored_decision",
    ],
)
def test_build_identity_reload_counterexamples_reject(capture, fault):
    if fault in {"wrong_plan", "wrong_hash", "wrong_decision"}:
        capture["assertions"][
            {
                "wrong_plan": "accepted_plan_id",
                "wrong_hash": "accepted_plan_content_hash",
                "wrong_decision": "decision_id",
            }[fault]
        ] = "foreign"
    elif fault == "untrusted_reload":
        _sidecar(capture, "reload_control_input.json", lambda row: row.update(trusted=False))
    elif fault == "reload_before_recovery":
        _sidecar(capture, "reload_control_input.json", lambda row: row.update(clicked_at="2000-01-01T00:00:00Z"))
    else:

        def change(payload):
            pointer = payload["workspace"]["accepted_plan_pointer"]
            plan = next(
                plan
                for plan in payload["workspace"]["geography"]["plans"]
                if plan["plan_id"] == capture["assertions"]["accepted_plan_id"]
            )
            if fault == "restored_session":
                payload["session"]["session_id"] = "foreign-session"
            elif fault == "restored_trip":
                payload["session"]["trip_id"] = "foreign-trip"
            elif fault == "restored_run":
                payload["session"]["run_id"] = "foreign-run"
            elif fault == "restored_pointer_plan":
                pointer["accepted_plan_id"] = "foreign-plan"
            elif fault == "restored_pointer_hash":
                pointer["accepted_plan_content_hash"] = "foreign-hash"
            elif fault == "restored_plan_hash":
                plan["content_hash"] = "foreign-hash"
            elif fault == "restored_plan_geometry":
                plan["validated_legs"]["features"][0]["geometry"]["coordinates"][0][0] += 0.01
            elif fault == "restored_plan_metadata":
                plan["label"] = "Different canonical plan metadata"
            elif fault == "duplicate_restored_plan":
                payload["workspace"]["geography"]["plans"].append(deepcopy(plan))
            elif fault == "restored_decision":
                pointer["decision_id"] = "foreign-decision"

        _restore_mutation(capture, change)
    with pytest.raises(core.E2E1VerificationError):
        _check(capture)


@pytest.mark.parametrize("fault", ["overlapping_generations", "identical_builds", "missing_node"])
def test_actual_restart_order_guard_rejects_lifecycle_counterexamples(fault):
    bindings = [
        {"build_id": "unit-build-a", "installed_at": "2026-09-08T12:00:00Z", "finalized_at": "2026-09-08T12:00:01Z"},
        {"build_id": "unit-build-b", "installed_at": "2026-09-08T12:00:02Z", "finalized_at": "2026-09-08T12:00:03Z"},
    ]
    witnesses = deepcopy(bindings)
    node = "tests/product_app/browser/test_build_coherence_live.py::" + verifier.BUILD_TEST + "[original]"
    observed, nodes = {node}, {node}
    if fault == "overlapping_generations":
        witnesses[1]["installed_at"] = witnesses[0]["installed_at"]
    elif fault == "identical_builds":
        witnesses[1]["build_id"] = witnesses[0]["build_id"]
    else:
        observed = set()
    function = ast.parse(inspect.getsource(verifier._build_case)).body[0]
    matches = [
        statement
        for statement in function.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and any(
            isinstance(argument, ast.Constant) and argument.value == "build_restart_order_invalid"
            for argument in statement.value.args
        )
    ]
    assert len(matches) == 1
    code = compile(ast.Module(body=matches, type_ignores=[]), str(Path(verifier.__file__)), "exec")
    with pytest.raises(core.E2E1VerificationError, match="build_restart_order_invalid"):
        exec(code, {**vars(verifier), "witnesses": witnesses, "observed": observed, "nodes": nodes})
