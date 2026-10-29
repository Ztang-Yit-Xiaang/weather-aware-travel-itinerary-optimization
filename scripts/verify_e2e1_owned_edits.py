"""Independent canonical readback for owned-rule chat and real editor journeys."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import (
    _conversation_accepted_baseline,
    _http,
    _lifecycle,
    _proposal,
    _runtime,
    _stream,
    combined_chat_control_receipt,
    load,
    stamp,
)

TEST = "test_owned_mixed_chat_renders_one_child_or_preserves_parent"
CASES = {"combined_chat": "test_owned_replacement_and_duration_uses_existing_complete_browser_journey"}
CASES.update({
    "combined_chat_owned_" + family + ("_protected" if protected else "_eligible")
    + ("_editor" if entry == "editor" else "") + ("_accepted_child" if family != "mixed" else ""):
        f"{TEST}[{family}-{entry}-{protected}]"
    for family, entry in (("mixed", "chat"), ("duration", "chat"), ("time_window", "chat"),
                          ("duration", "editor"), ("time_window", "editor"))
    for protected in (False, True)
})
BASE_URL = "http://127.0.0.1:8127"
IDENTITIES = ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id",
              "rendered_geometry_hash", "rendered_marker_ids")


def require(condition, reason):
    core._require(condition, "e2e1_owned_" + reason)


def canonical_readback(root, case, payload):
    """Rerun the real compiler/evaluator against the exact persisted parent."""
    from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
    from itinerary_system.product_app.product_demo import load_product_demo_package
    from itinerary_system.product_app.route_delta import configured_route_matrix

    fixture = load(case / "owned_fixture_input.json")
    registry_path = Path(fixture["registry_path"]).resolve()
    require(registry_path.is_relative_to(root / "fixture_inputs") and registry_path.is_file()
        and registry_path.read_text(encoding="utf-8") == fixture["registry_body_utf8"], "fixture_registry_unbound")
    registry = json.loads(fixture["registry_body_utf8"])
    require(len(registry["runs"]) == 1, "fixture_registry_universe_invalid")
    registration = registry["runs"][0]
    package_path = (core.ROOT / registration["relative_path"]).resolve()
    require(package_path == registry_path.parent / "owned_product_package"
        and core._sha256(package_path / "manifest.json") == registration["manifest_sha256"],
        "fixture_manifest_unbound")
    package = load_product_demo_package(core.ROOT, package_path)
    proposal = payload["proposal"]
    execution = proposal["runtime_execution"]
    state = root / "_state" / hashlib.sha256(case.name.encode()).hexdigest()[:12]
    require(Path(load(case / "server_lifecycle.json")["state_directory"]).resolve() == state,
        "state_directory_unbound")
    parent = load(core._safe_path(state / "plan_repository", proposal["parent_plan_id"] + ".json"))
    fixture_parent = load(package_path / load(package_path / "manifest.json")["parent"]["plan_relative_path"])
    child, diff, certificate = (execution[key] for key in ("child_plan", "plan_diff", "evaluation_certificate"))
    require(parent["owned_constraints"] == fixture_parent["owned_constraints"] == child["owned_constraints"]
        and len(parent["owned_constraints"]) == 3
        and parent["plan_id"] == proposal["parent_plan_id"] == child["parent_plan_id"] == diff["parent_plan_id"]
        and parent["content_hash"] == proposal["parent_plan_content_hash"] == core._artifact_content_hash(parent)
        and child["plan_id"] == certificate["plan_id"] == diff["child_plan_id"] == proposal["child_plan_id"]
        and child["content_hash"] == certificate["plan_content_hash"] == proposal["child_plan_content_hash"] == core._artifact_content_hash(child)
        and certificate["content_hash"] == proposal["certificate_content_hash"] == core._artifact_content_hash(certificate)
        and certificate["certificate_id"] == proposal["certificate_id"]
        and core._artifact_content_hash(diff) == proposal["diff_content_hash"]
        and execution["evaluation_executed"] is True
        and execution["request_id"] == child["planning_request_id"] == execution["repair_request"]["request_id"],
        "canonical_identity_or_rule_universe_invalid")
    compiler = FrozenDraftCompiler(parent, package.evidence_bundles, repository_root=core.ROOT,
        route_matrix=configured_route_matrix(core.ROOT, registration["run_id"], package.primary_bundle.route_matrix))
    fresh = compiler.compile(payload["session"]["draft"], accepted_plan_id=parent["plan_id"])
    def normalize(value):
        return json.loads(json.dumps(value))
    require(normalize(fresh.child_plan) == child and normalize(fresh.diff) == diff,
        "fresh_child_or_diff_mismatch")
    def semantic(value):
        return {key: item for key, item in value.items() if key not in {"evaluated_at", "content_hash"}}
    require(normalize(semantic(fresh.certificate)) == semantic(certificate), "fresh_evaluation_mismatch")
    require(certificate["evaluator_version"] == "verify-001-plan-evaluator-v2"
        and diff["schema_version"] == ("plan-diff-v7" if fixture["family"] == "mixed" else "plan-diff-v5"),
        "owned_contract_version_invalid")
    require((proposal["state"] == "ineligible") is fixture["protected"]
        and certificate["comparison_eligibility"] == proposal["state"], "eligibility_invalid")
    if fixture["protected"]:
        require("owned_attribute_constraint_unsatisfied" in {row["code"] for row in certificate["failures"]},
            "protected_rule_not_enforced")
    else:
        require(certificate["failures"] == [], "eligible_child_has_failures")
        require(load(core._safe_path(state / "plan_repository", child["plan_id"] + ".json")) == child,
            "persisted_child_mismatch")
    return fixture, parent, child, certificate


def editor_contract(value, events, stage, family):
    """Bind trusted browser form input to the exact local API operation."""
    control = value["control"]
    operation = "set_stop_" + family
    target = "griffith_observatory" if family == "duration" else "golden_gate_bridge"
    parameters = ({"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}
        if family == "duration" else {"earliest_arrival": "10:00", "latest_departure": None})
    require(value["context"] == "accepted_child" and value["phase"] == "stage"
        and value["request"] == stage["input"] and value["response"] == stage["output"]
        and control["visible"] is True and control["disabled"] is False and control["operation"] == operation
        and value["raw_events"] == events[:value["event_end"]]
        and stage["input"]["type"] == operation and stage["input"]["target"] == target
        and stage["input"]["parameters"] == parameters and stage["input"]["source"] == "map", "editor_input_unbound")
    form_events = [row for row in events[value["event_start"]:value["event_end"]]
                   if row["form_id"] == control["form_id"] and row["operation"] == operation]
    require(any(row["type"] == "submit" and row["trusted"] is True and row["fields"] == control["fields"] for row in form_events)
        and any(row["type"] == "change" and row["trusted"] is True and row["operation"] == operation
            and row["name"] == ("duration-preferred" if family == "duration" else "earliest_arrival")
            and row["value"] == ("90" if family == "duration" else "10:00") for row in events)
        and any(row["type"] == "click" and row["trusted"] is True and row["id"] == "inline-preview" for row in events),
        "editor_trusted_input_missing")


def verify_scenarios(root, run_id, process, nodes):
    base = root / "conversation"
    require({path.name for path in base.iterdir() if path.is_dir()} == set(CASES), "case_universe_invalid")
    redactions = load(root / "state_credential_redaction.json")
    state_files = sorted((root / "_state").rglob("session_*.json"))
    require(redactions["assertions_passed"] is True and redactions["redacted_file_count"] == len(state_files)
        and [row["path"] for row in redactions["files"]] == [path.relative_to(root).as_posix() for path in state_files]
        and all(load(path)["token_verifier"] == "OMITTED_SECRET" and core._sha256(path) == row["post_redaction_sha256"]
            for path, row in zip(state_files, redactions["files"], strict=True)), "credential_redaction_invalid")
    observed, witnesses = set(), []
    previous = stamp(process["started_at"])
    for name, test in CASES.items():
        case = base / name
        receipt = _lifecycle(case, run_id + "_" + name, process)
        require(previous <= stamp(receipt["installed_at"]), "lifecycle_overlap")
        previous = stamp(receipt["finalized_at"])
        expected_node = "tests/product_app/browser/test_owned_combined_journey_live.py::" + test
        observed.update(_runtime(case, receipt, {expected_node}, BASE_URL))
        records = _http(case, BASE_URL)
        trace, payload, stage = (load(case / filename) for filename in ("trace.json", "preview_response.json", "staging_io.json"))
        fixture, parent, child, certificate = canonical_readback(root, case, payload)
        capture_plan_ids = {trace["initial"]["active_plan_id"], trace["rendered"]["active_plan_id"]}
        require(trace["assertions_passed"] is True and trace["accepted"] is not fixture["protected"], "trace_scope_invalid")
        if name.endswith("_accepted_child"):
            baseline = load(case / "accepted_baseline_artifacts.json")
            boundary = baseline["http_boundary_files"]
            _conversation_accepted_baseline(case, trace, records[:len(boundary)], continuation=True)
            capture_plan_ids.add(baseline["original"]["active_plan_id"])
            records = records[len(boundary):]
        if name == "combined_chat":
            combined_chat_control_receipt(lambda filename, directory=case: load(directory / filename), name=name)
        else:
            require(all(load(case / "staged_map_ack.json")[key] == trace["initial"][key] for key in IDENTITIES),
                "staging_changed_map")
        mutations = [row for row in records if row["method"] == "POST" and row["url"].endswith(
            ("/draft/operations", "/preview", "/accept", "/keep-original", "/rollback", "/draft/undo"))]
        require([row["url"].rsplit("/", 1)[-1] for row in mutations] ==
            (["operations", "preview"] if fixture["protected"] else ["operations", "preview", "accept"])
            and json.loads(mutations[0]["request_body_utf8"]) == stage["input"]
            and json.loads(mutations[0]["response_body_utf8"]) == stage["output"]
            and json.loads(mutations[1]["response_body_utf8"]) == payload == {"session": payload["session"], "proposal": payload["proposal"]}
            and json.loads(mutations[1]["request_body_utf8"])["expected_revision"] == stage["output"]["session"]["revision"]
            and payload["session"]["draft"] == stage["output"]["session"]["draft"]
            and payload["session"]["accepted_plan_id"] == parent["plan_id"], "real_mutation_io_unbound")
        chats = [row for row in records if row["url"].endswith("/copilot/messages")]
        require(len(chats) == (0 if fixture["entry"] == "editor" else 1), "chat_dispatch_inventory_invalid")
        if chats:
            turns = load(case / "exact_chat_io.json")
            request, events = _stream(chats[0])
            require(len(turns) == 1 and turns[0]["input"] == request and turns[0]["events"] == events
                and turns[0]["output"] == events[-1], "exact_chat_unbound")
        if fixture["entry"] == "editor":
            editor_contract(load(case / "editor_control_io.json"), load(case / "editor_final_events.json"), stage, fixture["family"])
            require("Mixing those families is not supported" not in load(case / "editor_policy_before.json")["text"],
                "obsolete_editor_guidance")
        rendered = trace["rendered"]
        restored = trace.get("reloaded", trace.get("accepted_map"))
        require(all(restored[key] == rendered[key] for key in IDENTITIES), "reload_map_changed")
        proposal = payload["proposal"]
        if fixture["protected"]:
            require(rendered["preview_state"] == "PREVIEW_INELIGIBLE"
                and all(rendered[key] == trace["initial"][key] for key in IDENTITIES)
                and load(case / "presentation_state.json")["accept_disabled"] is True, "ineligible_child_rendered")
        else:
            _proposal(proposal, rendered, combined=True)
            evidence = load(case / "evidence_panel.json")["scope"]
            require(evidence["evidencePlanId"] == child["plan_id"] and evidence["planContentHash"] == child["content_hash"],
                "evidence_identity_mismatch")
            accepted = load(case / "acceptance_response.json")
            for artifact in (accepted["result"]["pointer"], accepted["result"]["decision"]):
                require(artifact["content_hash"] == core._canonical_sha256({key: item for key, item in artifact.items()
                    if key != "content_hash"}), "acceptance_artifact_hash_invalid")
            require(accepted == json.loads(mutations[2]["response_body_utf8"])
                and accepted["result"]["pointer"]["accepted_plan_id"] == child["plan_id"]
                and accepted["result"]["pointer"]["accepted_plan_content_hash"] == child["content_hash"]
                and accepted["result"]["decision"]["previous_plan_id"] == parent["plan_id"]
                and accepted["result"]["decision"]["certificate_id"] == certificate["certificate_id"]
                and accepted["result"]["decision"]["certificate_content_hash"] == certificate["content_hash"]
                and accepted["result"]["decision"]["diff_content_hash"] == proposal["diff_content_hash"], "acceptance_unbound")
            if name != "combined_chat":
                metrics = load(case / "presentation_state.json")["metrics"]
                require(metrics["planId"] == child["plan_id"] and metrics["planContentHash"] == child["content_hash"]
                    and metrics["evaluationId"] == certificate["certificate_id"],
                    "metric_identity_mismatch")
        if name != "combined_chat":
            restoration = load(case / "restoration_response.json")
            require(any(row["method"] == "GET" and json.loads(row["response_body_utf8"]) == restoration for row in records)
                and restoration["session"]["accepted_plan_id"] == rendered["active_plan_id"]
                and restoration["session"]["draft"] == (payload["session"]["draft"] if fixture["protected"] else []),
                "restoration_http_unbound")
        labels = (["initial", "combined_preview", "combined_compare", "combined_accepted_reload"] if name == "combined_chat"
            else ["initial", "owned_preview", "owned_reload"])
        if name.endswith("_accepted_child"):
            labels += ["accepted_setup_original", "accepted_setup_preview"]
        core._verify_journey_captures(case, labels, capture_plan_ids)
        require(load(case / "console_errors.json") == load(case / "network_failures.json") == [], "browser_errors")
        witnesses.append({"scenario": name, "run_id": receipt["run_id"], "pid": receipt["process_id"],
            "provider_calls": 0, "request_id": proposal["runtime_execution"]["request_id"],
            "parent_plan_id": parent["plan_id"], "child_plan_id": child["plan_id"],
            "child_plan_content_hash": child["content_hash"], "evaluation_id": certificate["certificate_id"],
            "geometry_hash": rendered["rendered_geometry_hash"], "rendered_map_plan_id": rendered["rendered_map_plan_id"],
            "fresh_child_diff_evaluation_match": True})
    require(observed == nodes, "runtime_node_universe_invalid")
    return witnesses
