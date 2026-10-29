"""Independent real-control draft-only and disabled-variant evidence gate."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _proposal

SCENARIO = "capability_controls"
TEST = "test_real_control_variants_preserve_draft_only_and_disabled_semantics"
CONTEXTS = ("original", "accepted_child")


def require(condition, reason):
    core._require(condition, "e2e1_controls_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def selection_workspace_projection(workspace):
    plans=[{key:plan[key] for key in ('plan_id','content_hash','stops','validated_legs')}
        for plan in workspace['geography']['plans']]
    return {
        'geography':{'plans':plans},
        'alternatives':[{'plan_id':row['plan_id'],'plan_content_hash':row['plan_content_hash']}
            for row in workspace['alternatives']],
        'draft_capabilities':{'candidate_choices':workspace['draft_capabilities']['candidate_choices']},
    }


def selection_prior_sources(row, records):
    boundaries=[index for index,record in enumerate(records) if record['phase']==row['phase']]
    require(boundaries,'selection_action_http_missing')
    responses=[json.loads(record['response_body_utf8']) for record in records[:boundaries[0]]]
    sessions=[response['session'] for response in responses if isinstance(response,dict) and 'session' in response]
    workspaces=[response for response in responses if isinstance(response,dict) and 'workspace' in response]
    try:
        projected_workspace=selection_workspace_projection(workspaces[-1]['workspace']) if workspaces else None
    except (KeyError,TypeError):
        projected_workspace=None
    require(sessions and workspaces and sessions[-1]==row['session_before']
        and projected_workspace==row['workspace_before'],
        'selection_prior_workspace_http_unbound')
    owner=workspaces[-1]['session']
    require(all(owner[key]==row['session_before'][key] for key in ('session_id','accepted_plan_id')),
        'selection_prior_workspace_owner_mismatch')


def expected_drafts(typed):
    operations, words = typed["operations"], typed["vocabularies"]
    rows = [("set_stop_duration", value) for value in operations["set_stop_duration"]["draft_only_modes"]]
    rows += [("set_stop_role", value) for value in operations["set_stop_role"]["draft_only_roles"]]
    rows += [("set_stop_commitment", value) for value in words["commitment_strengths"]
             if value not in operations["set_stop_commitment"]["protected_strengths"]]
    rows += [("set_attribute_constraint", "attribute:" + value) for value in words["attributes"]]
    rows += [("report_route_issue", value) for value in words["route_issue_types"]]
    require(len(rows) == len(set(rows)) == 21, "draft_variant_universe_changed")
    return set(rows)


def variant(operation):
    values = operation["parameters"]
    return {
        "set_stop_duration": lambda: values["duration"]["mode"],
        "set_stop_role": lambda: values["role"],
        "set_stop_commitment": lambda: values["strength"],
        "set_attribute_constraint": lambda: "attribute:" + values["attribute"],
        "report_route_issue": lambda: values["issue_type"],
    }[operation["type"]]()


def _same_map(ack, parent):
    require(all(ack.get(key) == parent[key] for key in ("active_plan_id", "active_plan_content_hash",
        "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids")), "accepted_map_changed")


def cross_family_negative(row, typed):
    first = row["first"]
    require(first in {"duration", "replacement"}, "cross_family_direction_invalid")
    operations = [row["first_operation"], row["second_operation"]]
    expected = ["set_stop_duration", "replace_nearby"] if first == "duration" else ["replace_nearby", "set_stop_duration"]
    require([operation["type"] for operation in operations] == expected, "cross_family_operations_invalid")
    duration = operations[0 if first == "duration" else 1]
    capability = typed["operations"]["set_stop_duration"]
    mode = duration["parameters"]["duration"]["mode"]
    require(mode in capability["draft_only_modes"] and mode not in capability["supported_evaluated_modes"],
        "cross_family_supported_mode_mislabeled")
    control = row["second_control"]["control"]
    require(row["assertions_passed"] is True and row["second_operation_disabled"] is True
        and type(row["second_stage_request_delta"]) is int and row["second_stage_request_delta"] == 0
        and control["visible"] is True and control["disabled"] is True and control["path"] == "/app/itinerary",
        "cross_family_blocked_input_invalid")
    if first == "duration":
        require(control["id"] == "draft-replace"
            and row["selected_candidate_id"] == operations[1]["parameters"]["candidate_id"],
            "cross_family_candidate_input_unbound")
    else:
        from verify_e2e1_control_actions import parameter_fields

        require(control["operation"] == "set_stop_duration", "cross_family_duration_control_unbound")
        parameter_fields("set_stop_duration", duration["parameters"], control["fields"])
    return mode


def _operation(row, records, parent, context):
    expected = row["input"]
    require(row["context"] == context and row["assertions_passed"] is True
        and row["feedback_tier"] == "draft_only" and row["preview_request_delta"] == 0
        and row["operation"] == expected["type"] and row["variant"] == variant(expected), "operation_claim_invalid")
    staged = row["staged"]
    operation = staged["operation"]
    require({key: operation[key] for key in ("type", "target", "parameters")} == expected
        and staged["session"]["draft"] == [operation]
        and staged["session"]["accepted_plan_id"] == parent["active_plan_id"], "staged_operation_mismatch")
    matches = [http for http in records if http["method"] == "POST" and http["url"].endswith("/draft/operations")
        and json.loads(http["response_body_utf8"]) == staged]
    require(len(matches) == 1, "operation_http_unbound")
    request = json.loads(matches[0]["request_body_utf8"])
    require({key: request[key] for key in expected} == expected and request["source"] == "map", "operation_input_mismatch")
    phase = matches[0]["phase"]
    require(not any(http["method"] == "POST" and http["url"].endswith(("/preview", "/accept"))
        and http["phase"] == phase for http in records), "draft_only_executed")
    impact = row["impact"]
    require(any(http["url"].endswith("/draft/impact-preview") and http["phase"] == phase
        and json.loads(http["response_body_utf8"]) == impact for http in records), "impact_http_unbound")
    require(impact["session_revision"] == staged["session"]["revision"]
        and impact["parent_plan_id"] == parent["active_plan_id"]
        and impact["parent_plan_content_hash"] == parent["active_plan_content_hash"]
        and impact["is_certified"] is False and impact["certification_status"] == "not_certified"
        and impact["summary"]["operation_count"] == impact["summary"]["draft_only_count"] == 1
        and impact["summary"]["evaluated_executable_count"] == 0
        and impact["summary"]["can_run_evaluated_preview"] is False
        and len(impact["operations"]) == 1 and impact["operations"][0]["operation_id"] == operation["operation_id"]
        and impact["operations"][0]["feedback_tier"] == "draft_only"
        and impact["operations"][0]["preview_executable"] is False
        and impact["operations"][0]["evaluated_repair"] is False, "false_draft_certification")
    undone = row["undo"]
    require(undone["undone"] == operation and undone["session"]["draft"] == []
        and undone["session"]["accepted_plan_id"] == parent["active_plan_id"]
        and any(http["method"] == "POST" and http["url"].endswith("/draft/undo")
            and http["phase"] == phase and json.loads(http["response_body_utf8"]) == undone for http in records), "undo_http_unbound")
    _same_map(row["rendered"], parent)
    return phase + "_staged"


def _disabled(rows, capabilities, inventory, context):
    typed = capabilities["typed_edit_capabilities"]
    expected = {(operation, "disabled") for group in capabilities.values()
                for operation, capability in group["operations"].items() if capability["enabled"] is False}
    for operation in ("set_stop_commitment", "set_attribute_constraint"):
        expected.update((operation, ("strength:" if operation == "set_attribute_constraint" else "") + strength)
                        for strength in typed["operations"][operation]["protected_strengths"])
    require(len(rows) == len(expected) and {(row["operation"], row["variant"]) for row in rows} == expected,
        "disabled_variant_universe_invalid")
    for row in rows:
        require(row["context"] == context and row["disabled"] is True and row["forced_interaction"] is False
            and row["draft_request_delta"] == 0 and row["label"], "disabled_claim_invalid")
        controls = [control for observation in inventory if observation["operation"] == row["operation"]
                    for control in observation["controls"]]
        if row["variant"] == "disabled":
            require(row["visible"] is True and any(control["disabled"] is True and control["visible"] is True
                and control["label"] == row["label"] for control in controls), "disabled_dom_unbound")
        else:
            strength = row["variant"].removeprefix("strength:")
            options = [option for control in controls for select in control["options"] if select["name"] == "strength"
                       for option in select["options"] if option["value"] == strength]
            require(options and all(option["disabled"] is True and "permission required" in option["label"]
                and option["label"] == row["label"] for option in options), "protected_option_dom_unbound")


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require(len(cases) == 2 and [case.get("name") for case in cases] == [TEST + "[" + context + "]" for context in CONTEXTS]
        and all(case.get("classname") == "tests.product_app.browser.test_capability_controls_live" for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()), "test_inventory_invalid")
    evidence = root / SCENARIO
    require({path.name for path in evidence.iterdir() if path.is_dir()} == set(CONTEXTS), "context_universe_invalid")
    for context in CONTEXTS:
        base = evidence / context
        assertions = core._load(base / "browser_assertions.json")
        parent = assertions["parent"]
        require(assertions["schema_version"] == "capability-controls-browser-v1"
            and assertions["context"] == context and assertions["assertions_passed"] is True
            and assertions["evaluated_variant_coverage_claimed"] is False
            and parent["active_plan_id"] == parent["rendered_map_plan_id"] and parent["preview_state"] == "PARENT_RENDERED",
            "browser_assertions_invalid")
        _same_map(assertions["final"], parent)
        records = _http(base, base_url)
        from verify_e2e1_control_actions import verify_rows
        verify_rows(load(base / 'control_action_receipts.json'),records,context)
        require(all(http.get("phase") for http in records) and not any("/copilot/messages" in http["url"] for http in records),
            "unexpected_chat_or_missing_phase")
        snapshots = core._load(base / "capability_snapshot.json")
        require(snapshots["schema_version"] == "runtime-capability-snapshot-v1" and snapshots["context"] == context
            and snapshots["plan_id"] == parent["active_plan_id"]
            and snapshots["plan_content_hash"] == parent["active_plan_content_hash"], "capability_identity_invalid")
        caps = snapshots["capabilities"]
        from verify_e2e1_control_actions import disabled_day, legacy_keep
        disabled_day(load(base / 'disabled_day_control.json'),caps['typed_edit_capabilities']['operations']['set_stop_day'],records,context,parent)
        legacy_keep(load(base / 'legacy_keep_control.json'),records,context,parent)
        responses = [json.loads(http["response_body_utf8"]) for http in records]
        bindings = [response for response in responses if isinstance(response, dict) and "workspace" in response and "session" in response
            and response["session"]["session_id"] == snapshots["session_id"]
            and response["session"]["accepted_plan_id"] == snapshots["plan_id"]
            and all(response["workspace"].get(group) == contract for group, contract in caps.items())]
        require(bindings and set(caps) == {
            "draft_capabilities", "typed_edit_capabilities", "map_edit_capabilities"},
            "capability_http_unbound")
        plan = next(plan for plan in bindings[0]["workspace"]["geography"]["plans"] if plan["plan_id"] == snapshots["plan_id"])
        require(plan["content_hash"] == snapshots["plan_content_hash"]
            and core._completed_race_geometry_hash(plan) == parent["rendered_geometry_hash"]
            and set(parent["rendered_marker_ids"]) == {row["properties"]["stop_id"] for row in plan["stops"]["features"]}, "parent_geometry_unbound")
        inventory = core._load(base / "runtime_control_inventory.json")
        require(inventory["schema_version"] == "runtime-control-inventory-v1" and inventory["context"] == context
            and inventory["absent_controls_are_not_claimed_tested"] is True, "inventory_scope_invalid")
        registry = core._load(core.ROOT / "configs/e2e1_interaction_coverage.json")
        from e2e1_runtime_gate_coverage import (
            verify_compare_responsive,
            verify_general_journey,
            verify_selection_journey,
        )
        general=core._load(base/'general_control_inventory.json')
        general_rows=core._load(base/'general_control_journeys.json')
        general_events=core._load(base/'general_control_events.json')['events']
        source_bindings={row['binding_id']:row for row in registry['source_bindings']+registry['native_controls']}
        plans={plan['plan_id']:plan for response in responses if isinstance(response,dict) and 'workspace' in response
            for plan in response['workspace']['geography']['plans']}
        for row in general_rows['journeys']:
            verify_general_journey(row,source_bindings[row['binding_id']],general['observations'],general_events,parent)
            verify_selection_journey(row,[core._load(core._safe_path(base,name)) for name in row['http_files']],parent)
            if row['binding_id'] in {'select_day','text_select_stop','text_select_leg','candidate_select','compare_select'}:
                selection_prior_sources(row,records)
            if row['binding_id'] in {'compare_choose','context_choose'}:
                require(row['selected_plan']==plans.get(row['selected_plan']['plan_id']),'choice_plan_not_bound_to_backend')
            for name in row['http_files']:
                recorded=core._load(core._safe_path(base,name))
                require(recorded in records and recorded['phase']==row['phase']
                    and not recorded['url'].endswith(('/draft/operations','/draft/undo','/preview','/accept','/copilot/messages')),
                    'general_control_mutation_or_http_unbound')
        responsive=load(base/'compare_responsive_journeys.json')
        verify_compare_responsive(responsive,context,parent,list(plans.values()))
        for name in responsive[0]['http_files']:
            require(core._load(core._safe_path(base,name)) in records
                and core._load(core._safe_path(base,name))['method']=='GET','responsive_unexpected_api_mutation')
        for observation in inventory["observations"]:
            mapping = [row for row in registry["rows"] if row["action_id"] == observation["action_id"]]
            require(len(mapping) == 1 and observation["context"] == context
                and all(observation[key] == mapping[0][key] for key in ("operation", "capability_group", "selector"))
                and observation["actual_dom_query"] == ", ".join(part.strip() for part in observation["selector"].split(";"))
                and observation["observed_count"] == len(observation["controls"]), "inventory_mapping_invalid")
        operations = load(base / "operation_assertions.json")
        require(len(operations) == assertions["draft_only_variant_count"] == 21
            and {(row["operation"], row["variant"]) for row in operations} == expected_drafts(caps["typed_edit_capabilities"]),
            "draft_variant_coverage_incomplete")
        labels = ["baseline", "disabled_and_protected_controls", "final_unchanged"]
        labels += [_operation(row, records, parent, context) for row in operations]
        disabled = load(base / "disabled_control_assertions.json")
        require(disabled == assertions["disabled_variants"], "disabled_projection_invalid")
        _disabled(disabled, caps, inventory["observations"], context)
        cross = load(base / "cross_family_assertions.json")
        require(cross == assertions["cross_family_cases"]
            and [row["first"] for row in cross] == (["duration", "replacement"] if context == "original" else []), "cross_family_universe_invalid")
        for row in cross:
            cross_family_negative(row, caps["typed_edit_capabilities"])
            phase = "cross_family_" + row["first"] + "_first"
            posts = [http for http in records if http["phase"] == phase and http["method"] == "POST" and http["url"].endswith("/draft/operations")]
            require(row["assertions_passed"] is True and row["context"] == context and row["second_operation_disabled"] is True
                and row["second_stage_request_delta"] == 0 and len(posts) == 1
                and {key: json.loads(posts[0]["request_body_utf8"])[key] for key in row["first_operation"]} == row["first_operation"],
                "cross_family_stage_invalid")
            blocked_kind = "replace_nearby" if row["first"] == "duration" else "set_stop_duration"
            require(any(control["disabled"] and control["visible"] and "combin" in control["title"].lower()
                for observation in inventory["observations"] if observation["phase"] == phase and observation["operation"] == blocked_kind
                for control in observation["controls"]), "cross_family_disabled_dom_unbound")
            labels.append(phase)
        previews = [json.loads(http["response_body_utf8"]) for http in records if http["method"] == "POST" and http["url"].endswith("/preview")]
        accepts = [json.loads(http["response_body_utf8"]) for http in records if http["method"] == "POST" and http["url"].endswith("/accept")]
        require(len(previews) == len(accepts) == (1 if context == "accepted_child" else 0), "preview_accept_count_invalid")
        require(len([http for http in records if http["method"] == "POST" and http["url"].endswith("/draft/operations")])
            == 22 + len(cross) + len(previews), "unobserved_draft_mutation")
        if previews:
            require(previews[0] == core._load(base / "baseline_preview_response.json")
                and accepts[0] == core._load(base / "baseline_acceptance_response.json"), "baseline_http_unbound")
            _proposal(previews[0]["proposal"], parent)
            decision = accepts[0]["result"]["decision"]
            require(decision["selected_plan_id"] == parent["active_plan_id"]
                and decision["selected_plan_content_hash"] == parent["active_plan_content_hash"]
                and decision["certificate_id"] == previews[0]["proposal"]["certificate_id"], "accepted_baseline_lineage_invalid")
            labels.append("baseline_preview")
        core._verify_journey_captures(base, labels, {parent["active_plan_id"]})
        require(load(base / "console_errors.json") == [], "browser_errors")
        from verify_e2e1_network_observations import verify as verify_network
        from verify_e2e1_network_observations import verify_asset_observation

        assets = load(base / "map_asset_sources.json")
        verify_asset_observation(assets, parent, context)
        verify_network(base, base_url, load(base / "network_failures.json"),
            {"sources":assets["vector_sources"],"sprite":assets["sprite"]},asset_observation=assets)
    return {"verified": True, "test_count": 2, "draft_only_variants": 42, "evaluated_variant_coverage_claimed": False}


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, StopIteration, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_controls_evidence_malformed") from error
