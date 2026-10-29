"""Fail-closed product interaction inventory, separate from browser pass counts."""
from __future__ import annotations

import json
from urllib.parse import urlsplit

MAPPING_FIELDS = {
    "action_id", "capability_group", "operation", "contexts", "selector", "handler",
    "state_action", "endpoint", "service", "artifacts", "ui_selector", "map_effect",
    "supported_semantics", "variant_source",
}


def required_variants(operation, capability, group):
    """Derive conditional modes from the captured server contract, never a happy path."""
    if capability.get("enabled") is not True:
        return [{"id": "disabled", "feedback_tier": "unsupported",
                 "blocking_code": capability.get("blocking_code") or capability.get("status")}]
    if operation in {"set_stop_duration", "set_stop_role"}:
        axis = "modes" if operation == "set_stop_duration" else "roles"
        evaluated, draft = capability.get("supported_evaluated_" + axis), capability.get("draft_only_" + axis)
        if (not isinstance(evaluated, list) or not isinstance(draft, list)
                or not evaluated or not draft or not all(isinstance(value, str) for value in evaluated + draft)
                or len(set(evaluated + draft)) != len(evaluated + draft)):
            return [{"id": "invalid_conditional_contract", "feedback_tier": "invalid"}]
        return ([{"id": value, "feedback_tier": "evaluated"} for value in evaluated]
                + [{"id": value, "feedback_tier": "draft_only"} for value in draft])
    if operation == "add_candidate" and isinstance(capability.get("candidate_day_readiness"), list):
        rows = capability["candidate_day_readiness"]
        return [{"id": str(row.get("candidate_id")) + "@" + str(row.get("day")),
                 "feedback_tier": "evaluated" if row.get("preview_executable") is True else "unsupported",
                 "blocking_code": row.get("blocking_code") or row.get("status")}
                for row in rows if isinstance(row, dict)]
    if operation == "set_stop_time_window":
        return [{"id": value, "feedback_tier": "evaluated"} for value in ("earliest_only", "latest_only", "both")]
    if operation == "set_stop_commitment":
        return [{"id": value, "feedback_tier": "unsupported" if value in capability.get("protected_strengths", [])
                 else "draft_only"} for value in group.get("vocabularies", {}).get("commitment_strengths", [])]
    if operation == "set_attribute_constraint":
        return ([{"id": "attribute:" + value, "feedback_tier": "draft_only"}
                 for value in group.get("vocabularies", {}).get("attributes", [])]
                + [{"id": "strength:" + value, "feedback_tier": "unsupported"}
                   for value in capability.get("protected_strengths", [])])
    if operation == "report_route_issue":
        return [{"id": value, "feedback_tier": "draft_only"}
                for value in group.get("vocabularies", {}).get("route_issue_types", [])]
    return [{"id": "default", "feedback_tier": capability.get("feedback_tier")
             or ("evaluated" if capability.get("preview_executable") else "draft_only")}]


def operation_variant(operation):
    parameters = operation.get("parameters") or {}
    kind = operation.get("type")
    if kind == "set_stop_duration":
        return parameters.get("duration", {}).get("mode")
    if kind == "set_stop_role":
        return parameters.get("role")
    if kind == "add_candidate":
        return str(operation.get("target")) + "@" + str(parameters.get("day"))
    if kind == "set_stop_time_window":
        return "both" if parameters.get("earliest_arrival") and parameters.get("latest_departure") else (
            "earliest_only" if parameters.get("earliest_arrival") else "latest_only")
    return "default"


def same_authoritative_parent(left, right):
    return all(left.get(key) == right.get(key) for key in ("context", "plan_id", "plan_content_hash"))


def collect_verified_operations(directory, snapshots, suites, registry):
    """Project exact UI-generated POSTs only from independently passed child suites.

    These are not new browser assertions. The registered case must already have
    passed its independent verifier, including render, artifact and I/O bindings.
    Draft-only, disabled and route-checked cases need dedicated proofs and cannot
    be inferred from an evaluated-case pass.
    """
    if directory is None:
        return []
    passed = {suite["mode"]: set(suite.get("test_nodeids", []))
              for suite in suites if suite.get("status") == "passed"}
    observations = []
    for mapping in registry.get("rows", []):
        for journey in mapping.get("journeys", []):
            mode = journey.get("browser_mode")
            if not any(node.endswith("::" + journey.get("test_name", ""))
                       for node in passed.get(mode, set())):
                continue
            base_root = (directory / "suites" / mode).resolve()
            base = (base_root / journey["case_directory"]).resolve()
            if not base.is_relative_to(base_root):
                continue
            required = [base / name for name in journey.get("evidence", [])]
            if not required or not all(path.is_file() and path.resolve().is_relative_to(base) for path in required):
                continue
            assertion = json.loads((base / journey["assertions_file"]).read_text(encoding="utf-8"))
            if not isinstance(assertion, dict) or assertion.get("assertions_passed") is not True:
                continue
            records = []
            for path in sorted(base.glob("http_*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(record, dict) or record.get("schema_version") != "e2e1-product-http-io-v1"
                        or record.get("method") != "POST" or record.get("response_status_code") != 200
                        or not urlsplit(record.get("url", "")).path.endswith(("/draft/operations", "/preview"))):
                    continue
                body = json.loads(record["request_body_utf8"])
                response = json.loads(record["response_body_utf8"])
                records.append((path, record, body, response))
            for path, record, operation, response in records:
                if (not urlsplit(record["url"]).path.endswith("/draft/operations")
                        or operation.get("type") != mapping["operation"]
                        or operation.get("source") != journey.get("source")
                        or operation_variant(operation) not in journey.get("variants", [])):
                    continue
                actual = response.get("operation") or {}
                if any(actual.get(key) != operation.get(key) for key in ("type", "target", "parameters")):
                    continue
                session = response.get("session") or {}
                matching_contexts = {(item["context"], item["plan_content_hash"])
                    for item in snapshots if item.get("session_id") == session.get("session_id")
                    and item.get("plan_id") == session.get("accepted_plan_id")
                    and item.get("context") in journey.get("contexts", [])}
                if len(matching_contexts) != 1:
                    continue
                context, parent_hash = next(iter(matching_contexts))
                previews = [(preview_path, result.get("proposal")) for preview_path, http, _, result in records
                    if urlsplit(http["url"]).path.endswith("/preview")
                    and any(draft.get("operation_id") == actual.get("operation_id") and actual.get("operation_id")
                            for draft in result.get("session", {}).get("draft", []))]
                for preview_path, proposal in previews:
                    if (not isinstance(proposal, dict) or proposal.get("state") not in {"eligible", "ineligible"}
                            or proposal.get("parent_plan_id") != session.get("accepted_plan_id")
                            or proposal.get("parent_plan_content_hash") != parent_hash):
                        continue
                    observations.append({"action_id": mapping["action_id"], "context": context,
                        "variant": operation_variant(operation), "feedback_tier": "evaluated",
                        "browser_mode": mode, "test_name": journey["test_name"],
                        "parent_plan_id": session["accepted_plan_id"], "parent_plan_content_hash": parent_hash,
                        "operation_artifact": path.relative_to(directory).as_posix(),
                        "preview_artifact": preview_path.relative_to(directory).as_posix(),
                        "preview_state": proposal["state"], "child_plan_id": proposal.get("child_plan_id")})
    if "capability-controls" in passed:
        observations.extend(_control_observations(directory, snapshots, passed["capability-controls"], registry))
    if "evaluated-variants" in passed:
        observations.extend(_evaluated_observations(directory, snapshots, passed["evaluated-variants"], registry))
    if "waypoint-pointer" in passed:
        observations.extend(_waypoint_observations(directory, snapshots, passed["waypoint-pointer"], registry))
    if "waypoint-contexts" in passed:
        observations.extend(_waypoint_context_observations(directory, snapshots, passed["waypoint-contexts"], registry))
    if "candidate-day-matrix" in passed:
        observations.extend(_candidate_day_observations(directory, snapshots, passed["candidate-day-matrix"], registry))
    if "chat-closure" in passed:
        observations.extend(_chat_observations(directory, snapshots, passed["chat-closure"], registry))
    return observations


def _observed_mapping(snapshots, registry, context, kind, plan_id, plan_hash):
    if not any(item.get("context") == context and item.get("plan_id") == plan_id
               and item.get("plan_content_hash") == plan_hash for item in snapshots):
        return None
    mappings = [row for row in registry["rows"] if row["operation"] == kind and context in row["contexts"]]
    return mappings[0] if len(mappings) == 1 else None


def _waypoint_context_observations(directory, snapshots, nodes, registry):
    observed = []
    for context in ("original", "accepted_child"):
        test = "test_route_checked_waypoint_contexts_preserve_accepted_plan[" + context + "]"
        if not any(node.endswith("::" + test) for node in nodes):
            continue
        base = directory / "suites/waypoint-contexts/waypoint_contexts" / context
        assertions = json.loads((base / "browser_assertions.json").read_text(encoding="utf-8"))
        rows = json.loads((base / "operation_assertions.json").read_text(encoding="utf-8"))
        if assertions["assertions_passed"] is not True or rows != assertions["operations"]:
            continue
        for index, row in enumerate(rows):
            parent = row["parent"]
            mapping = _observed_mapping(snapshots, registry, context, row["operation"],
                parent["active_plan_id"], parent["active_plan_content_hash"])
            if (not mapping or row["context"] != context or row["assertions_passed"] is not True
                    or row["feedback_tier"] != "route_checked" or row["evaluated_repair"] is not False
                    or row["scope"] != "route_checked_draft_only"):
                continue
            observed.append({"action_id": mapping["action_id"], "context": context, "variant": row["variant"],
                "feedback_tier": "route_checked", "browser_mode": "waypoint-contexts", "test_name": test,
                "parent_plan_id": parent["active_plan_id"], "parent_plan_content_hash": parent["active_plan_content_hash"],
                "operation_artifact": (base / "operation_assertions.json").relative_to(directory).as_posix(),
                "assertion_index": index, "preview_state": "route_checked_draft_only", "child_plan_id": None,
                "evaluated_repair_claimed": False})
    return observed


def _candidate_day_observations(directory, snapshots, nodes, registry):
    test = "test_browser_candidate_day_readiness_in_original_and_accepted_child"
    if not any(node.endswith("::" + test) for node in nodes):
        return []
    base = directory / "suites/candidate-day-matrix/candidate_day_matrix"
    assertions = json.loads((base / "browser_assertions.json").read_text(encoding="utf-8"))
    if assertions["assertions_passed"] is not True or assertions["pair_scope"] != "readiness_and_staging_not_eligibility":
        return []
    observed = []
    for index, row in enumerate(json.loads((base / "candidate_day_cases.json").read_text(encoding="utf-8"))):
        mapping = _observed_mapping(snapshots, registry, row["context"], "add_candidate",
            row["accepted_plan_id"], row["accepted_plan_content_hash"])
        if not mapping or row["assertions_passed"] is not True or row["comparison_eligibility"] != "not_evaluated":
            continue
        ready = row["readiness"]["preview_executable"]
        if row["scope"] not in {"disabled_pair_not_interacted", "real_staging_and_impact_only"}:
            continue
        # A ready Add control plus staged draft is NOT an independently evaluated
        # addition. Keep that requirement open until an actual preview proves it.
        observed.append({"action_id": mapping["action_id"], "context": row["context"],
            "variant": row["candidate_id"] + "@" + str(row["day"]),
            "feedback_tier": "readiness_and_staging" if ready else "unsupported",
            "browser_mode": "candidate-day-matrix", "test_name": test,
            "parent_plan_id": row["accepted_plan_id"], "parent_plan_content_hash": row["accepted_plan_content_hash"],
            "operation_artifact": (base / "candidate_day_cases.json").relative_to(directory).as_posix(),
            "assertion_index": index, "preview_state": "not_evaluated", "child_plan_id": None,
            "scope": row["scope"], "evaluated_repair_claimed": False})
    return observed


def _chat_observations(directory, snapshots, nodes, registry):
    observed = []
    for case_context in ("original", "accepted_child"):
        test = "test_keep_and_chat_families_preserve_runtime_truth[" + case_context + "]"
        if not any(node.endswith("::" + test) for node in nodes):
            continue
        base = directory / "suites/chat-closure/chat_closure" / case_context
        assertions = json.loads((base / "browser_assertions.json").read_text(encoding="utf-8"))
        rows = json.loads((base / "journey_assertions.json").read_text(encoding="utf-8"))
        if assertions["assertions_passed"] is not True or rows != assertions["journeys"]:
            continue
        for index, row in enumerate(rows):
            case = (base / row["case"]).resolve()
            if not case.is_relative_to(base.resolve()) or row["assertions_passed"] is not True:
                continue
            payload = json.loads((case / "preview_response.json").read_text(encoding="utf-8"))
            proposal, parent = payload["proposal"], row["parent"]
            if (proposal["state"] != row["comparison_eligibility"] or proposal["state"] not in {"eligible", "ineligible"}
                    or proposal["parent_plan_id"] != parent["active_plan_id"]
                    or proposal["parent_plan_content_hash"] != parent["active_plan_content_hash"]
                    or payload["session"]["draft"] != row["operations"]):
                continue
            for operation in row["operations"]:
                mapping = _observed_mapping(snapshots, registry, row["context"], operation["type"],
                    parent["active_plan_id"], parent["active_plan_content_hash"])
                if not mapping:
                    continue
                observed.append({"action_id": mapping["action_id"], "context": row["context"],
                    "variant": operation_variant(operation), "feedback_tier": "evaluated", "browser_mode": "chat-closure",
                    "test_name": test, "parent_plan_id": parent["active_plan_id"], "parent_plan_content_hash": parent["active_plan_content_hash"],
                    "operation_artifact": (base / "journey_assertions.json").relative_to(directory).as_posix(),
                    "assertion_index": index, "preview_artifact": (case / "preview_response.json").relative_to(directory.resolve()).as_posix(),
                    "preview_state": proposal["state"], "child_plan_id": proposal.get("child_plan_id"),
                    "interaction_surface": row["interaction_surface"], "scope": row["scope"],
                    "actual_openai_connectivity_claimed": False})
    return observed


def _evaluated_observations(directory, snapshots, nodes, registry):
    observations = []
    for context in ("original", "accepted_child"):
        test = "test_real_evaluated_variants_preview_reload_and_undo[" + context + "]"
        if not any(node.endswith("::" + test) for node in nodes):
            continue
        base = directory / "suites/evaluated-variants/evaluated_variants" / context
        snapshot = json.loads((base / "capability_snapshot.json").read_text(encoding="utf-8"))
        if not any(same_authoritative_parent(item, snapshot) for item in snapshots):
            continue
        for row in json.loads((base / "variant_assertions.json").read_text(encoding="utf-8")):
            mappings = [mapping for mapping in registry["rows"] if mapping["operation"] == row["operation"] and context in mapping["contexts"]]
            if (len(mappings) != 1 or row["context"] != context or row["assertions_passed"] is not True
                    or row["feedback_tier"] != "evaluated" or row["variant_acceptance_claimed"] is not False
                    or row["scope"] != "evaluated_preview_only_then_undo"):
                continue
            case = base / (row["operation"].removeprefix("set_stop_") + "_" + row["variant"])
            if not case.resolve().is_relative_to(base.resolve()):
                continue
            if json.loads((case / "trace.json").read_text(encoding="utf-8")) != row:
                continue
            observations.append({"action_id": mappings[0]["action_id"], "context": context,
                "variant": row["variant"], "feedback_tier": "evaluated", "browser_mode": "evaluated-variants", "test_name": test,
                "parent_plan_id": row["parent_plan_id"], "parent_plan_content_hash": row["parent_plan_content_hash"],
                "operation_artifact": (case / "operation_input.json").relative_to(directory).as_posix(),
                "preview_artifact": (case / "preview_response.json").relative_to(directory).as_posix(),
                "preview_state": row["comparison_eligibility"], "child_plan_id": row["child_plan_id"],
                "scope": "evaluated_preview_reload_undo", "variant_acceptance_claimed": False})
    return observations


def _waypoint_observations(directory, snapshots, nodes, registry):
    test = "test_map_pointer_add_relocate_and_two_undos_preserve_accepted_plan"
    if not any(node.endswith("::" + test) for node in nodes):
        return []
    base = directory / "suites/waypoint-pointer/waypoint_pointer"
    assertions = json.loads((base / "browser_assertions.json").read_text(encoding="utf-8"))
    if (assertions["assertions_passed"] is not True or assertions["evaluated_repair_claimed"] is not False
            or not any(item["context"] == "original" and item["plan_id"] == assertions["accepted_plan_id"]
                and item["plan_content_hash"] == assertions["accepted_plan_content_hash"] for item in snapshots)):
        return []
    observations = []
    for operation in ("add_custom_waypoint", "relocate_custom_waypoint"):
        mappings = [row for row in registry["rows"] if row["operation"] == operation and "original" in row["contexts"]]
        if len(mappings) != 1:
            continue
        observations.append({"action_id": mappings[0]["action_id"], "context": "original", "variant": "default",
            "feedback_tier": "route_checked", "browser_mode": "waypoint-pointer", "test_name": test,
            "parent_plan_id": assertions["accepted_plan_id"], "parent_plan_content_hash": assertions["accepted_plan_content_hash"],
            "operation_artifact": (base / "browser_assertions.json").relative_to(directory).as_posix(),
            "preview_state": "route_checked_draft_only", "child_plan_id": None, "evaluated_repair_claimed": False})
    return observations


def _control_observations(directory, snapshots, nodes, registry):
    """Project only the dedicated independently verified draft/disabled journeys."""
    observations = []
    for context in ("original", "accepted_child"):
        test = "test_real_control_variants_preserve_draft_only_and_disabled_semantics[" + context + "]"
        if not any(node.endswith("::" + test) for node in nodes):
            continue
        base = directory / "suites/capability-controls/capability_controls" / context
        snapshot = json.loads((base / "capability_snapshot.json").read_text(encoding="utf-8"))
        if not any(all(item.get(key) == snapshot.get(key) for key in ("context", "session_id", "plan_id", "plan_content_hash"))
                   for item in snapshots):
            continue
        for filename, tier in (("operation_assertions.json", "draft_only"), ("disabled_control_assertions.json", "unsupported")):
            rows = json.loads((base / filename).read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                mappings = [mapping for mapping in registry["rows"] if mapping["operation"] == row["operation"]
                            and context in mapping["contexts"]]
                if len(mappings) != 1 or row["context"] != context:
                    continue
                if tier == "draft_only" and (row["assertions_passed"] is not True or row["feedback_tier"] != tier):
                    continue
                if tier == "unsupported" and (row["disabled"] is not True or row["draft_request_delta"] != 0):
                    continue
                observations.append({"action_id": mappings[0]["action_id"], "context": context,
                    "variant": row["variant"], "feedback_tier": tier, "browser_mode": "capability-controls", "test_name": test,
                    "parent_plan_id": snapshot["plan_id"], "parent_plan_content_hash": snapshot["plan_content_hash"],
                    "operation_artifact": (base / filename).relative_to(directory).as_posix(), "assertion_index": index,
                    "preview_state": "not_executed", "child_plan_id": None})
    return observations


def coverage_report(snapshots, suites, registry, directory=None):
    if (not isinstance(registry, dict) or not isinstance(registry.get("rows"), list)
            or not isinstance(registry.get("required_contexts"), list)
            or not isinstance(registry.get("unverified_requirements"), list)):
        return {"schema_version": "e2e1-interaction-coverage-v2", "passed": False,
                "enabled_interactions": [], "findings": [{"code": "interaction_registry_invalid"}]}
    from e2e1_runtime_gate_coverage import gate_report

    runtime = gate_report(directory, suites, registry, required_variants)
    findings = list(runtime["findings"])
    contexts = {snapshot.get("context") for snapshot in snapshots if isinstance(snapshot, dict)}
    for context in registry["required_contexts"]:
        if context not in contexts:
            findings.append({"code": "runtime_capability_context_missing", "context": context})
    try:
        observations = collect_verified_operations(directory, snapshots, suites, registry)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        observations = []
        findings.append({"code": "interaction_runtime_evidence_invalid"})
    enabled, disabled = [], []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("capabilities"), dict):
            findings.append({"code": "runtime_capability_snapshot_invalid"})
            continue
        for group, capabilities in snapshot["capabilities"].items():
            if not isinstance(capabilities, dict) or not isinstance(capabilities.get("operations"), dict):
                findings.append({"code": "runtime_capability_snapshot_invalid", "capability_group": group})
                continue
            for operation, capability in capabilities["operations"].items():
                if not isinstance(capability, dict) or type(capability.get("enabled")) is not bool:
                    findings.append({"code": "runtime_capability_operation_invalid", "operation": operation})
                    continue
                is_enabled = capability["enabled"]
                if not is_enabled and registry.get("require_disabled_feedback") is not True:
                    continue
                binding = {"context": snapshot["context"], "capability_group": group, "operation": operation,
                    "plan_id": snapshot["plan_id"], "plan_content_hash": snapshot["plan_content_hash"],
                    "capability_artifact": snapshot["artifact"], "capability": capability}
                matches = [row for row in registry["rows"] if isinstance(row, dict)
                    and row.get("capability_group") == group and row.get("operation") == operation
                    and snapshot["context"] in row.get("contexts", [])]
                variants = required_variants(operation, capability, capabilities)
                mapping_valid = (len(matches) == 1 and all(matches[0].get(field) for field in MAPPING_FIELDS))
                results = []
                for variant in variants:
                    evidence = [item for item in observations if mapping_valid
                        and item["action_id"] == matches[0]["action_id"] and item["context"] == snapshot["context"]
                        and item["parent_plan_id"] == snapshot["plan_id"]
                        and item["parent_plan_content_hash"] == snapshot["plan_content_hash"]
                        and item["variant"] == variant["id"] and item["feedback_tier"] == variant["feedback_tier"]]
                    results.append({**variant, "covered": bool(evidence), "runtime_evidence": evidence})
                binding.update(covered=bool(mapping_valid and results and all(row["covered"] for row in results)),
                               variants=results, mapping_complete=bool(mapping_valid))
                (enabled if is_enabled else disabled).append(binding)
                if not binding["covered"]:
                    findings.append({"code": ("enabled_interaction_without_complete_runtime_row" if is_enabled
                        else "disabled_interaction_without_runtime_feedback"), **binding})
    return {"schema_version": "e2e1-interaction-coverage-v2", "passed": not findings,
            "scope": "One independently verified actual parent per context and variant; not every possible plan permutation.",
            "snapshots_observed": len(snapshots), "enabled_interactions": enabled,
            "disabled_interactions": disabled, "runtime_operation_evidence": observations, "findings": findings,
            "runtime_requirements": runtime["requirements"], "source_control_inventory": runtime["source_control_inventory"],
            "external_gates": runtime["external_gates"], "full_phase_completion_claimed": False}
