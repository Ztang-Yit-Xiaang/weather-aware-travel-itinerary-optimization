"""Independently bind declared geometry faults to real evaluated children and UI rejection."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import verify_e2e1_vertical_slice as core
from verify_e2e1_bootstrap_recovery import fingerprint
from verify_e2e1_self_managed import _http, _lifecycle, _private_input_universe, _runtime, load, stamp

CASES = ("missing", "foreign")
TEST = "test_real_eligible_child_with_bad_geography_is_blocked_without_parent_or_draft_loss"
BASE_URL = "http://127.0.0.1:8127"
RELATIVE = "src/itinerary_system/product_app/api.py"
ANCHOR = "            return await asyncio.to_thread(publish_result)"


def require(condition, reason):
    core._require(condition, "e2e1_preview_geography_fault_" + reason)


def patch_contract(value, original):
    kind = value["kind"]
    require(
        kind in CASES
        and value["schema_version"] == "preview-geography-response-fault-v1"
        and value["path"] == RELATIVE
        and value["before_utf8"].encode() == original
        and all(
            value[key] is False
            for key in (
                "production_source_modified",
                "frontend_fetch_mocking",
                "optimizer_mocking",
                "evaluator_mocking",
                "canonical_artifacts_mutated",
            )
        ),
        "patch_scope_invalid",
    )
    replacement = """            canonical = await asyncio.to_thread(publish_result)
            # Declared disposable response-only fault AFTER real publication.
            from pathlib import Path as FaultPath
            wire = json.loads(json.dumps(canonical, allow_nan=False))
            if canonical["proposal"].get("state") != "eligible":
                raise RuntimeError("geometry_fault_requires_real_eligible_child")
            geometry = FAULT_GEOMETRY
            wire["proposal"]["geography_plan"] = geometry
            wire["session"]["proposal"]["geography_plan"] = geometry
            FaultPath(__file__).resolve().parents[3].joinpath("geography_fault_boundary.json").write_text(
                json.dumps({"canonical": canonical, "wire": wire}, ensure_ascii=False, allow_nan=False, sort_keys=True),
                encoding="utf-8")
            return wire"""
    expression = (
        "None"
        if kind == "missing"
        else (
            'next(plan for plan in service.workspace_view_for_session(session)["geography"]["plans"] '
            'if plan["plan_id"] == canonical["proposal"]["parent_plan_id"])'
        )
    )
    require(
        original.decode().count(ANCHOR) == 1
        and value["after_utf8"] == original.decode().replace(ANCHOR, replacement.replace("FAULT_GEOMETRY", expression))
        and value["before_sha256"] == hashlib.sha256(original).hexdigest()
        and value["after_sha256"] == hashlib.sha256(value["after_utf8"].encode()).hexdigest(),
        "patch_not_exact",
    )


def canonical_child(proposal):
    execution = proposal["runtime_execution"]
    child, diff, certificate = execution["child_plan"], execution["plan_diff"], execution["evaluation_certificate"]
    geometry = proposal["geography_plan"]
    require(
        proposal["state"] == "eligible"
        and proposal["execution_mode"] == "deterministic_repair_pipeline"
        and execution["evaluation_executed"] is True
        and certificate["comparison_eligibility"] == "eligible"
        and execution["request_id"] == child["planning_request_id"] == execution["repair_request"]["request_id"]
        and child["plan_id"] == diff["child_plan_id"] == certificate["plan_id"] == proposal["child_plan_id"]
        and child["parent_plan_id"] == diff["parent_plan_id"] == proposal["parent_plan_id"]
        and child["content_hash"]
        == core._artifact_content_hash(child)
        == certificate["plan_content_hash"]
        == proposal["child_plan_content_hash"]
        and certificate["content_hash"]
        == core._artifact_content_hash(certificate)
        == proposal["certificate_content_hash"]
        and certificate["certificate_id"] == proposal["certificate_id"]
        and core._artifact_content_hash(diff) == proposal["diff_content_hash"]
        and geometry["plan_id"] == child["plan_id"]
        and geometry["content_hash"] == child["content_hash"]
        and geometry["validated_legs"]["features"]
        and geometry["stops"]["features"],
        "underlying_child_not_canonical",
    )
    markers = [row["properties"]["stop_id"] for row in geometry["stops"]["features"]]
    require(
        "pfeiffer_big_sur_state_park" in markers and "bixby_creek_bridge_viewpoint" not in markers,
        "underlying_replacement_not_executed",
    )
    return child, certificate, geometry


def semantic_contract(value, boundary, records, persisted):
    kind = value["kind"]
    require(
        kind in CASES
        and value["schema_version"] == "preview-geography-fault-browser-v1"
        and value["coverage_category"] == "declared_backend_response_failure_path"
        and value["assertions_passed"] is True,
        "browser_scope_invalid",
    )
    initial, staged = value["initial"], value["staged"]
    payloads = [json.loads(row["response_body_utf8"]) for row in records]
    require(
        all(
            any(
                isinstance(payload, dict) and all(payload.get(key) == item for key, item in expected.items())
                for payload in payloads
            )
            for expected in (initial, value["restoration"])
        ),
        "workspace_http_unbound",
    )
    parent_id = initial["session"]["accepted_plan_id"]
    parent = next(plan for plan in initial["workspace"]["geography"]["plans"] if plan["plan_id"] == parent_id)
    canonical, wire = boundary["canonical"], boundary["wire"]
    child, certificate, geometry = canonical_child(canonical["proposal"])
    require(
        canonical["proposal"] == canonical["session"]["proposal"]
        and canonical["session"]["draft"] == staged["draft"]
        and len(staged["draft"]) == 1
        and staged["proposal"] is None
        and canonical["session"]["revision"] == staged["revision"] + 1
        and all(
            canonical["session"][key] == staged[key] == initial["session"][key]
            for key in ("session_id", "trip_id", "run_id", "accepted_plan_id")
        )
        and value["restoration"]["session"] == staged
        and value["restoration"]["workspace"]["accepted_plan_pointer"] == initial["workspace"]["accepted_plan_pointer"],
        "parent_draft_or_owner_changed",
    )
    expected = deepcopy(canonical)
    for proposal in (expected["proposal"], expected["session"]["proposal"]):
        proposal["geography_plan"] = None if kind == "missing" else parent
    require(wire == expected == value["wire"], "wire_fault_changed_more_than_geography")
    stored = json.loads(persisted["utf8"])
    session = stored["session"]
    public = {key: item for key, item in canonical["session"].items() if key != "schema_version"}
    require(
        hashlib.sha256(persisted["utf8"].encode()).hexdigest() == persisted["sha256"]
        and persisted["body_capture"] == "credential_redacted"
        and persisted["credential_redactions"] == ["/token_verifier"]
        and len(persisted["original_sha256"]) == 64
        and set(persisted["original_sha256"]) <= set("0123456789abcdef")
        and persisted["original_sha256"] != persisted["sha256"]
        and persisted["plaintext_credentials_persisted"] is False
        and set(stored) == {"schema_version", "session", "token_verifier"}
        and stored["token_verifier"] == "OMITTED_SECRET"
        and stored["schema_version"] == "product-session-snapshot-v1"
        and set(session) == set(public) | {"created_at", "last_accessed_at"}
        and all(session[key] == item for key, item in public.items())
        and stamp(session["created_at"]) <= stamp(session["last_accessed_at"]),
        "stored_canonical_result_mutated",
    )
    preview = [row for row in records if row["url"].endswith("/preview")]
    require(
        len(preview) == 1
        and preview[0]["method"] == "POST"
        and preview[0]["response_status_code"] == 200
        and json.loads(preview[0]["request_body_utf8"]) == {"expected_revision": staged["revision"]}
        and json.loads(preview[0]["response_body_utf8"]) == wire
        and not any(row["url"].endswith(("/accept", "/copilot/messages", "/draft/undo")) for row in records),
        "real_preview_http_unbound",
    )
    stages = [row for row in records if row["url"].endswith("/draft/operations")]
    require(
        len(stages) == 1
        and stages[0]["method"] == "POST"
        and json.loads(stages[0]["response_body_utf8"])["session"] == staged,
        "real_staging_unbound",
    )
    operation = staged["draft"][0]
    stage_request = json.loads(stages[0]["request_body_utf8"])
    request = canonical["proposal"]["runtime_execution"]["repair_request"]
    expected_operation = {
        "type": "replace_nearby",
        "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        "source": "map",
    }
    require(
        stage_request == {**expected_operation, "expected_revision": staged["revision"] - 1}
        and all(operation.get(key) == item for key, item in expected_operation.items())
        and set(operation) == set(expected_operation) | {"operation_id", "created_at", "evidence_refs"}
        and isinstance(operation["operation_id"], str)
        and operation["operation_id"]
        and operation["evidence_refs"] == []
        and json.loads(stages[0]["response_body_utf8"])["operation"] == operation
        and canonical["proposal"]["expected_revision"] == staged["revision"]
        and request["parent_plan_id"] == parent_id
        and request["confirmed_constraints"]["must_delete"] == [expected_operation["target"]]
        and request["confirmed_constraints"]["must_include"] == [expected_operation["parameters"]["candidate_id"]]
        and request["confirmed_constraints"]["affected_days"] == [5]
        and request["confirmed_constraints"]["require_road_validated_routes"] is True
        and request["tolerance_profile"]["route_fallback_allowed"] is False
        and [(row["stop_id"], row["day"]) for row in request["candidate_pois"]] == [("pfeiffer_big_sur_state_park", 5)]
        and any(
            row["day"] == 5 and row["stop_ids"] == ["bixby_creek_bridge_viewpoint"] for row in request["baseline_route"]
        ),
        "map_operation_optimizer_mapping_invalid",
    )
    parent_markers = [row["properties"]["stop_id"] for row in parent["stops"]["features"]]
    points = value["checkpoints"]
    require(
        set(points) == {"initial_parent", "staged_parent", "restored_staged_parent", "blocked_parent"},
        "capture_universe_invalid",
    )
    for label, ack in points.items():
        require(
            ack["preview_state"] == ("PREVIEW_FAILED" if label == "blocked_parent" else "PARENT_RENDERED")
            and ack["active_plan_id"] == ack["rendered_map_plan_id"] == parent_id
            and ack["active_plan_content_hash"] == parent["content_hash"]
            and ack["rendered_geometry_hash"] == core._completed_race_geometry_hash(parent)
            and len(ack["rendered_marker_ids"]) == len(set(ack["rendered_marker_ids"]))
            and set(ack["rendered_marker_ids"]) == set(parent_markers),
            "parent_map_not_preserved",
        )
    ui = value["ui"]
    require(
        ui["map"]["previewState"] == "PREVIEW_FAILED"
        and ui["map"]["activePlanId"] == parent_id
        and ui["map"]["renderedMapPlanId"] == parent_id
        and ui["map"]["activePlanContentHash"] == parent["content_hash"]
        and ui["metric"]["planId"] == parent_id
        and ui["metric"]["planContentHash"] == parent["content_hash"]
        and ui["accept_disabled"] is True
        and ui["undo_disabled"] is False
        and "1 draft" in ui["draft_text"]
        and "The evaluated repair response did not match" in ui["toast"]
        and "Bixby Creek Bridge" in ui["timeline"]
        and "Pfeiffer Big Sur State Park" not in ui["timeline"],
        "visible_failure_not_truthful",
    )
    return {
        "kind": kind,
        "parent_plan_id": parent_id,
        "child_plan_id": child["plan_id"],
        "child_plan_content_hash": child["content_hash"],
        "evaluation_id": certificate["certificate_id"],
        "evaluated_geometry_hash": core._completed_race_geometry_hash(geometry),
        "rendered_map_plan_id": parent_id,
        "state": "PREVIEW_FAILED",
        "staged_draft_retained": True,
        "accept_disabled": True,
        "coverage_category": "declared_backend_response_failure_path",
    }


def raw_http_contract(requests, records, witnesses):
    indexed = {row["request_sequence"]: row for row in records}
    require(
        len(indexed) == len(records) and [row["sequence"] for row in requests] == list(range(1, len(requests) + 1)),
        "request_universe_invalid",
    )
    seen = set()
    for request in requests:
        require(request["phase"] in {"generation_a", "generation_b"}, "request_phase_invalid")
        witness = witnesses[0 if request["phase"] == "generation_a" else 1]
        require(
            request["failed_at"] is request["failure"] is None
            and 200 <= request["status"] < 400
            and stamp(witness["installed_at"])
            <= stamp(request["started_at"])
            <= stamp(request["response_at"])
            <= stamp(request["finished_at"])
            <= stamp(witness["finalized_at"]),
            "request_incomplete_or_failed",
        )
        if request["url"].startswith(BASE_URL + "/api/"):
            require(request["sequence"] in indexed, "orphan_request")
            row = indexed[request["sequence"]]
            require(
                all(row[key] == request[key] for key in ("method", "url", "phase"))
                and row["request_original_body_sha256"] == request["request_body_sha256"]
                and row["response_status_code"] == request["status"]
                and request["http_file"] == f"http_{records.index(row) + 1:03d}.json",
                "request_response_unbound",
            )
            seen.add(request["sequence"])
        else:
            require(request["method"] == "GET" and request["http_file"] is None, "non_api_mutation")
    require(seen == set(indexed), "response_without_request")


def verify_scenarios(root, run_id, process, nodes):
    witnesses = []
    scenario = root / "preview_geography_fault"
    require({path.name for path in scenario.iterdir() if path.is_dir()} == set(CASES), "case_universe_invalid")
    for kind in CASES:
        base = scenario / kind
        identity = run_id + "_" + kind
        value, fault = core._load(base / "browser_assertions.json"), core._load(base / "fault_patch.json")
        require(value["kind"] == fault["kind"] == kind, "case_identity_invalid")
        patch_contract(fault, (core.ROOT / RELATIVE).read_bytes())
        records = _http(base, BASE_URL)
        boundary, raw = core._load(base / "backend_boundary.json"), core._load(base / "backend_boundary_bytes.json")
        require(
            json.loads(raw["utf8"]) == boundary and hashlib.sha256(raw["utf8"].encode()).hexdigest() == raw["sha256"],
            "backend_boundary_bytes_unbound",
        )
        require(
            raw["body_capture"] == "exact_utf8"
            and raw["credential_redactions"] == []
            and raw["original_sha256"] == raw["sha256"]
            and raw["plaintext_credentials_persisted"] is False,
            "backend_boundary_privacy_invalid",
        )
        proof = semantic_contract(value, boundary, records, core._load(base / "persisted_session.json"))
        protected = core._load(base / "protected_state.json")
        require(protected["before"] and protected["before"] == protected["after"], "accepted_artifacts_changed")
        for row in protected["before"].values():
            require(
                hashlib.sha256(row["utf8"].encode()).hexdigest() == row["sha256"]
                and len(row["utf8"].encode()) == row["size_bytes"],
                "protected_bytes_unbound",
            )
        copied = core._load(base / "private_fixture_inputs.json")
        rows = copied["files"]
        require(
            copied["schema_version"] == "build-coherence-private-inputs-v1"
            and copied["production_source_modified"] is copied["api_response_mocking"] is False
            and len(rows) == len({row["path"] for row in rows})
            and {row["path"] for row in rows} == _private_input_universe(core.ROOT),
            "source_universe_invalid",
        )
        for row in rows:
            path = core._safe_path(core.ROOT, row["path"])
            require(
                row["sha256"] == row["copied_sha256"] == core._sha256(path)
                and row["size_bytes"] == path.stat().st_size,
                "source_hash_invalid",
            )
        require(
            load(base / "private_restored_inputs.json")
            == [{key: row[key] for key in ("path", "sha256", "size_bytes")} for row in rows]
            and value["source_restored_sha256"] == fault["before_sha256"],
            "source_not_restored",
        )
        private = Path(copied["private_repository"])
        require(
            private.is_absolute()
            and private.resolve().is_relative_to((core.ROOT / "tmp").resolve())
            and private.name == "repo",
            "private_path_invalid",
        )
        bindings = load(base / "generation_bindings.json")
        require(
            [row["generation"] for row in bindings] == ["generation_a", "generation_b"], "generation_universe_invalid"
        )
        builds = [fingerprint(rows, {}), fingerprint(rows, {RELATIVE: fault["after_utf8"].encode()})]
        case_nodes = {node for node in nodes if node.endswith("[" + kind + "]")}
        require(len(case_nodes) == 1, "case_node_unbound")
        case_witnesses = []
        for index, binding in enumerate(bindings):
            generation = binding["generation"]
            directory = base / generation
            generation_id = identity + "_" + generation
            receipt = _lifecycle(directory, generation_id, process, child_process=True)
            require(_runtime(directory, receipt, case_nodes, BASE_URL) == case_nodes, "runtime_node_unbound")
            health = core._load(directory / "health.json")
            command = core._load(directory / "server_command.json")
            expected = [
                process["command"][0],
                "scripts/e2e1_provider_isolation.py",
                "--port",
                "8127",
                "--state-root",
                str(private.parent / "state"),
                "--output",
                str(directory / "provider_isolation.json"),
                "--run-id",
                generation_id,
            ]
            require(
                command["command"] == expected
                and command["cwd_scope"] == "private_product_copy"
                and command["run_id"] == generation_id
                and command["generation"] == generation
                and binding["run_id"] == generation_id
                and binding["pid"] == receipt["process_id"]
                and binding["build_id"] == health["build_id"] == builds[index]
                and binding["provider_call_count"] == 0
                and binding["provider_receipt"] == generation + "/provider_isolation.json"
                and binding["provider_receipt_sha256"] == core._sha256(directory / "provider_isolation.json")
                and all(binding[key] == receipt[key] for key in ("installed_at", "finalized_at")),
                "generation_binding_invalid",
            )
            labels = ("initial_parent", "staged_parent") if index == 0 else ("restored_staged_parent", "blocked_parent")
            require(
                binding["readiness_artifacts"]
                == sorted(generation + "/" + label + ".readiness.json" for label in labels),
                "capture_binding_invalid",
            )
            core._verify_journey_captures(directory, labels, {proof["parent_plan_id"]})
            case_witnesses.append({**binding, "provider_calls": 0})
        require(
            builds[0] != builds[1]
            and stamp(process["started_at"]) <= stamp(copied["copied_at"]) <= stamp(case_witnesses[0]["installed_at"])
            and stamp(case_witnesses[0]["finalized_at"]) <= stamp(case_witnesses[1]["installed_at"]),
            "generation_order_invalid",
        )
        raw_http_contract(load(base / "browser_requests.json"), records, case_witnesses)
        events, staging = load(base / "control_inputs.json"), load(base / "staging_inputs.json")
        require(
            len(events) == 1
            and events[0]["id"] == "inline-preview"
            and events[0]["trusted"] is True
            and events[0]["visible"] is True
            and events[0]["disabled"] is False
            and stamp(case_witnesses[1]["installed_at"]).timestamp() * 1000
            <= events[0]["at"]
            <= stamp(
                core._load(base / "generation_b/blocked_parent.readiness.json")["screenshot_started_at"]
            ).timestamp()
            * 1000
            and any(
                row["id"] == "draft-replace" and row["trusted"] is True and row["disabled"] is False for row in staging
            ),
            "native_stage_or_preview_missing",
        )
        llm = core._load(base / "llm_io.json")
        require(
            type(llm["provider_call_count"]) is int
            and llm["provider_call_count"] == 0
            and llm["llm_input"] is llm["llm_output"] is None
            and load(base / "console_errors.json") == load(base / "network_failures.json") == [],
            "provider_or_browser_errors",
        )
        witnesses.extend(case_witnesses)
    return witnesses
