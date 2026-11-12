"""Synthetic verified-receipt fixtures; no actual browser/provider claims."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e2e1_interaction_coverage as coverage  # noqa: E402
import e2e1_runtime_gate_coverage as gates  # noqa: E402


def bundle(directory, mode, artifacts):
    root = directory / "suites" / mode
    files = []
    for name, value in artifacts.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value).encode()
        path.write_bytes(data)
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    data = json.dumps({"provider_calls": 0, "files": files}).encode()
    (root / "file_io_manifest.json").write_bytes(data)
    return {
        "mode": mode,
        "status": "passed",
        "verification": {"verified": True, "provider_calls": 0, "manifest_sha256": hashlib.sha256(data).hexdigest()},
    }


def lifecycle(directory):
    artifacts = {}
    original_cases = (
        "clarification_reload",
        "preference_memory",
        "capacity_turns",
        "capacity_bytes",
        "fake_sdk_cancel",
    )
    for case in (
        *original_cases,
        *(name + "_accepted_child" for name in original_cases if name != "clarification_reload"),
    ):
        original_name = case.removesuffix("_accepted_child")
        prefix = "conversation/" + case + "/"
        artifacts[prefix + "trace.json"] = {
            "schema_version": "conversation-browser-v1",
            "scenario": case,
            "assertions_passed": True,
            "blocked_attempt_delta": 0,
            "old_conversation_id": "old",
            "new_conversation_id": "new",
        }
        artifacts[prefix + "provider_scope.json"] = {
            "real_chat_endpoint": True,
            "internal_api_mocked": False,
            "live_provider_call_count": 0,
            "live_openai_connectivity_proven": False,
            "fake_sdk_attempt_count": int(original_name == "fake_sdk_cancel"),
        }
        if original_name != case:
            identity = {"active_plan_id": "child", "active_plan_content_hash": "hash"}
            artifacts[prefix + "trace.json"].update(context="accepted_child", initial=identity, final=identity)
            artifacts[prefix + "accepted_baseline_artifacts.json"] = {
                "context": "accepted_child",
                "accepted_map": identity,
                "original": {"active_plan_id": "parent"},
            }
            artifacts[prefix + "context_control_final.json"] = {
                "accepted_plan_id": "child",
                "accepted_plan_content_hash": "hash",
                "draft_operations": [],
            }
        if original_name == "fake_sdk_cancel":
            for phase in ("cancel", "retry"):
                artifacts[prefix + f"dispatch_after_{phase}.json"] = {"attempt_count": 1, "outcome": "unknown"}
    return [
        bundle(directory, "conversation", artifacts),
        bundle(
            directory,
            "chat-closure",
            {
                "chat_closure/accepted_child/browser_assertions.json": {
                    "assertions_passed": True,
                    "context": "accepted_child",
                },
                "chat_closure/accepted_child/context_accepted_transition.json": {
                    "accepted_plan_id": "child",
                    "accepted_plan_content_hash": "hash",
                    "conversation_memory": {"plan_id": "child", "plan_content_hash": "hash"},
                },
            },
        ),
    ]


def negatives(directory):
    artifacts = {}
    for context in ("original", "accepted_child"):
        prefix = "capability_controls/" + context + "/"
        duration = {
            "type": "set_stop_duration",
            "target": "stop",
            "parameters": {
                "duration": {"mode": "range", "preferred_minutes": None, "minimum_minutes": 45, "maximum_minutes": 90}
            },
        }
        replacement = {"type": "replace_nearby", "target": "other", "parameters": {"candidate_id": "candidate"}}
        cross = [
            {
                "first": first,
                "context": context,
                "assertions_passed": True,
                "second_operation_disabled": True,
                "second_stage_request_delta": 0,
                "first_operation": duration if first == "duration" else replacement,
                "second_operation": replacement if first == "duration" else duration,
                "selected_candidate_id": "candidate",
                "second_control": {
                    "control": {
                        "id": "draft-replace" if first == "duration" else None,
                        "operation": None if first == "duration" else "set_stop_duration",
                        "path": "/app/itinerary",
                        "visible": True,
                        "disabled": True,
                        "fields": None
                        if first == "duration"
                        else {"duration_mode": "range", "duration-minimum": "45", "duration-maximum": "90"},
                    }
                },
            }
            for first in (("duration", "replacement") if context == "original" else ())
        ]
        artifacts[prefix + "browser_assertions.json"] = {"assertions_passed": True, "cross_family_cases": cross}
        artifacts[prefix + "cross_family_assertions.json"] = cross
        artifacts[prefix + "capability_snapshot.json"] = {
            "capabilities": {
                "typed_edit_capabilities": {
                    "operations": {
                        "set_stop_duration": {
                            "enabled": True,
                            "supported_evaluated_modes": ["exact"],
                            "draft_only_modes": ["range"],
                        }
                    }
                }
            }
        }
        artifacts[prefix + "operation_assertions.json"] = [
            {
                "operation": "set_stop_duration",
                "variant": "range",
                "assertions_passed": True,
                "context": context,
                "feedback_tier": "draft_only",
            }
        ]
    return [bundle(directory, "capability-controls", artifacts)]


def report(root, suites, requirement):
    return gates.gate_report(root, suites, {"unverified_requirements": [requirement]}, coverage.required_variants)


@pytest.mark.parametrize(
    "factory,requirement", [(lifecycle, gates.CONVERSATION_GATE), (negatives, gates.NEGATIVE_GATE)]
)
def test_only_complete_bound_runtime_receipts_discharge_requirement(tmp_path, factory, requirement):
    result = report(tmp_path, factory(tmp_path), requirement)
    assert result["findings"] == [] and result["requirements"][0]["passed"] is True
    assert result["requirements"][0]["evidence"]
    assert result["full_phase_completion_claimed"] is False
    assert result["external_gates"][0]["verified"] is False


@pytest.mark.parametrize("fault", ["exact_mode", "missing_input", "enabled_control", "second_post", "foreign_context"])
def test_cross_family_negative_gate_rejects_hash_bound_but_misleading_receipts(tmp_path, fault):
    suites = negatives(tmp_path)
    directory = tmp_path / "suites/capability-controls"
    manifest = json.loads((directory / "file_io_manifest.json").read_text())
    artifacts = {row["path"]: json.loads((directory / row["path"]).read_text()) for row in manifest["files"]}
    prefix = "capability_controls/original/"
    rows = artifacts[prefix + "cross_family_assertions.json"]
    if fault == "exact_mode":
        rows[0]["first_operation"]["parameters"]["duration"]["mode"] = "exact"
    elif fault == "missing_input":
        rows[0].pop("second_operation")
    elif fault == "enabled_control":
        rows[0]["second_control"]["control"]["disabled"] = False
    elif fault == "second_post":
        rows[0]["second_stage_request_delta"] = 1
    else:
        rows[0]["context"] = "accepted_child"
    artifacts[prefix + "browser_assertions.json"]["cross_family_cases"] = rows
    suites[0] = bundle(tmp_path, "capability-controls", artifacts)
    result = report(tmp_path, suites, gates.NEGATIVE_GATE)
    assert result["findings"] and result["requirements"][0]["passed"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "suite_failed",
        "verification_absent",
        "manifest_changed",
        "artifact_changed",
        "artifact_missing",
        "provider_count",
        "boolean_provider_count",
    ],
)
def test_plausible_pass_or_unbound_mutated_receipt_never_clears_requirement(tmp_path, fault):
    suites = lifecycle(tmp_path)
    if fault == "suite_failed":
        suites[0]["status"] = "failed"
    elif fault == "verification_absent":
        suites[0].pop("verification")
    elif fault in {"provider_count", "boolean_provider_count"}:
        suites[0]["verification"]["provider_calls"] = 1 if fault == "provider_count" else False
    elif fault == "manifest_changed":
        (tmp_path / "suites/conversation/file_io_manifest.json").write_text("{}")
    else:
        path = tmp_path / "suites/conversation/conversation/capacity_bytes/trace.json"
        if fault == "artifact_missing":
            path.unlink()
        else:
            path.write_text('{"assertions_passed":true}')
    result = report(tmp_path, suites, gates.CONVERSATION_GATE)
    assert result["requirements"][0]["passed"] is False and result["findings"]


def test_registering_all_modes_is_not_lifecycle_coverage(tmp_path):
    result = report(tmp_path, [{"mode": "conversation", "status": "passed"}], gates.CONVERSATION_GATE)
    assert result["requirements"][0]["reason"] == "independently_passed_suite_missing"


def test_missing_accepted_memory_scope_prevents_original_only_completion(tmp_path):
    suites = lifecycle(tmp_path)
    suites.pop()
    assert report(tmp_path, suites, gates.CONVERSATION_GATE)["findings"]


def test_deterministic_gate_can_pass_without_claiming_live_website_connectivity(tmp_path):
    result = coverage.coverage_report(
        [], [], {"rows": [], "required_contexts": [], "unverified_requirements": []}, tmp_path
    )
    assert result["passed"] is True and result["full_phase_completion_claimed"] is False
    (external,) = result["external_gates"]
    assert external["required_for_full_phase"] is True and external["required_for_deterministic_aggregate"] is False
    assert external["verified"] is False and external["evidence_directory"] is None


def test_source_control_with_no_registry_row_is_reported_not_ignored(tmp_path):
    source = tmp_path / "src/itinerary_system/product_app/static/js"
    source.mkdir(parents=True)
    (source / "app.js").write_text('$("#new-control").addEventListener("click", execute);')
    (source / "copilot.js").write_text('$("#known").addEventListener("submit", submit);')
    inventory = gates.source_controls(tmp_path)
    assert {(row["selector"], row["event"]) for row in inventory} == {("#new-control", "click"), ("#known", "submit")}
    known = source_mapping(inventory[1])
    _, unmatched, stale = gates.binding_inventory(
        {"source_bindings": [known], "required_contexts": ["original"]}, inventory
    )
    assert unmatched == [inventory[0]] and stale == []


def test_missing_control_source_file_has_stable_failure(tmp_path):
    result = gates.gate_report(
        tmp_path, [], {"unverified_requirements": [gates.CONTROL_GATE]}, coverage.required_variants, root=tmp_path
    )
    assert result["requirements"][0]["reason"] == "control_source_inventory_unavailable"


def test_generated_or_variable_control_does_not_disappear_from_source_inventory(tmp_path):
    source = tmp_path / "src/itinerary_system/product_app/static/js"
    source.mkdir(parents=True)
    (source / "app.js").write_text('form.addEventListener("submit", dispatch);')
    (source / "copilot.js").write_text('control.addEventListener("click", action);')
    (source / "compare.js").write_text('container.querySelector("#compare-back").addEventListener("click", onBack);')
    inventory = gates.source_controls(tmp_path)
    assert len(inventory) == 3
    dynamic = [row for row in inventory if row["selector"] is None]
    assert len(dynamic) == 2 and all(row["binding_kind"] == "unresolved_dynamic_binding" for row in dynamic)
    assert any(row["selector"] == "#compare-back" for row in inventory)


def source_mapping(source):
    return {
        "binding_id": "help",
        "source_file": source["source_file"],
        "source_anchor": source["source_anchor"],
        "anchor_occurrence": source["anchor_occurrence"],
        "source_selector": source["selector"],
        "event": source["event"],
        "selectors": [source["selector"] or ".actual-generated-control"],
        "owner": "source-owner",
        "action_semantics": "Open help",
        "availability": "product",
        "kind": "control",
        "contexts": ["original"],
        "execution_credit": "real journey",
    }


def test_all_current_literal_dynamic_and_responsive_bindings_have_exact_explicit_mappings():
    registry = json.loads((gates.ROOT / "configs/e2e1_interaction_coverage.json").read_text())
    rows = gates.source_controls()
    mapped, unmatched, stale = gates.binding_inventory(registry, rows)
    assert len(rows) == len(mapped) == 108 and unmatched == stale == []
    added = {
        row["binding_id"]: row["required_execution_selectors"]
        for row in mapped
        if row["binding_id"]
        in {
            "hosted_sign_out",
            "hosted_sign_in",
            "interest_places_refresh",
            "interest_regions_retry",
            "interest_search_area",
            "interest_search_radius",
            "evidence_download",
        }
    }
    assert added == {
        "hosted_sign_out": ["#hosted-sign-out"],
        "hosted_sign_in": ["form.hosted-login"],
        "interest_places_refresh": ["#refresh-interest-place-data"],
        "interest_regions_retry": ["#retry-interest-regions"],
        "interest_search_area": ["#interest-search-area"],
        "interest_search_radius": ["#interest-radius"],
        "evidence_download": ["#evidence-download"],
    }
    recovery = {
        row["binding_id"]: row
        for row in mapped
        if row["binding_id"]
        in [
            "refresh_current_workspace",
            "refresh_protected_workspace",
            "check_saved_action",
            "retry_saved_action",
            "cancel_pending_action",
            "retry_draft_impact",
        ]
    }
    assert {key: row["source_selector"] for key, row in recovery.items()} == {
        "refresh_current_workspace": "[data-refresh-current-workspace]",
        "refresh_protected_workspace": "[data-repair-pending-mutation]",
        "check_saved_action": "[data-check-pending-mutation]",
        "retry_saved_action": "[data-retry-pending-mutation]",
        "cancel_pending_action": "[data-cancel-pending-mutation]",
        "retry_draft_impact": "[data-retry-draft-impact]",
    }
    assert all(row["required_execution_selectors"] == [row["source_selector"]] for row in recovery.values())
    dynamic = [row for row in mapped if row["source_selector"] is None]
    assert len(dynamic) == 15
    assert next(row for row in dynamic if row["binding_id"] == "compare_responsive")["kind"] == "responsive_lifecycle"
    assert next(row for row in mapped if row["binding_id"] == "navigate_global")["required_execution_selectors"] == [
        prefix + '[data-route="/app/' + route + '"]'
        for prefix in (".nav-item", ".mobile-nav button")
        for route in ("itinerary", "map", "repairs", "compare", "evidence")
    ]


@pytest.mark.parametrize("fault", ["anchor", "owner", "selector", "duplicate"])
def test_source_binding_registration_cannot_guess_a_new_alias_or_drop_owner(fault):
    source = {
        "source_file": "app.js",
        "source_anchor": "actual listener",
        "anchor_occurrence": 0,
        "selector": "#help-button",
        "event": "click",
    }
    mapping = source_mapping(source)
    if fault == "anchor":
        mapping["source_anchor"] = "different listener"
    elif fault == "selector":
        mapping["source_selector"] = "#different"
    elif fault == "owner":
        mapping["owner"] = ""
    registry = {"source_bindings": [mapping] * (2 if fault == "duplicate" else 1), "required_contexts": ["original"]}
    if fault in {"owner", "duplicate"}:
        with pytest.raises(gates.ReceiptError):
            gates.binding_inventory(registry, [source])
    else:
        assert gates.binding_inventory(registry, [source])[1]


def general_fixture():
    source = {
        "source_file": "app.js",
        "source_anchor": "actual listener",
        "anchor_occurrence": 0,
        "selector": "#help-button",
        "event": "click",
    }
    binding = source_mapping(source)
    parent = {"active_plan_id": "parent", "active_plan_content_hash": "p-hash"}
    event = {
        "trusted": True,
        "type": "click",
        "at": 1000,
        "matches": [{"binding_id": "help", "selector": "#help-button"}],
        "ancestors": [{"id": "help-button", "attributes": {}}],
    }
    observation = {
        "phase": "help_before",
        "context": "original",
        "binding_id": "help",
        "selector": "#help-button",
        "observed_count": 1,
        "controls": [{"visible": True, "disabled": False, "attributes": {}}],
    }
    journey = {
        "phase": "help",
        "binding_id": "help",
        "selector": "#help-button",
        "context": "original",
        "outcome": {"selector": "#info-dialog", "visible": True},
        "dom_result": {"visible": True},
        "trusted_events": [event],
        "accepted_plan_id": "parent",
        "accepted_plan_content_hash": "p-hash",
        "draft": [],
        "assertions_passed": True,
        "http_files": [],
    }
    return source, binding, parent, event, observation, journey


def selection_fixture(kind):
    from copy import deepcopy

    import verify_e2e1_vertical_slice as core

    parent = {
        "active_plan_id": "a",
        "active_plan_content_hash": "hash-a",
        "rendered_map_plan_id": "a",
        "rendered_geometry_hash": "g-a",
        "rendered_marker_ids": ["stop"],
    }
    plans = [
        {
            "plan_id": name,
            "content_hash": "hash-" + name,
            "stops": {"features": [{"properties": {"stop_id": "stop", "day": 6}}]},
            "validated_legs": {"features": [{"properties": {"route_leg_id": "leg", "day": 6}}]},
        }
        for name in ("a", "b", "c")
    ]
    before = {
        "session_id": "session",
        "revision": 4,
        "accepted_plan_id": "a",
        "draft": [],
        "selected_stop_id": None,
        "selected_segment_id": None,
        "selected_candidate_id": None,
        "selected_alternative_id": "b",
        "selected_day": 5,
    }
    attributes = {
        "select_day": {"data-day": "6", "data-stop": "stop"},
        "text_select_stop": {"data-geo-day": "6", "data-geo-stop": "stop"},
        "text_select_leg": {"data-geo-route-leg": "leg"},
        "candidate_select": {"id": "registered-candidate"},
        "compare_select": {"data-compare-option": "c"},
    }[kind]
    change = (
        {"selected_day": 6, "selected_stop_id": "stop", "selected_segment_id": None}
        if kind in {"select_day", "text_select_stop"}
        else {"selected_day": 6, "selected_segment_id": "leg", "selected_stop_id": None}
        if kind == "text_select_leg"
        else {"selected_candidate_id": "candidate"}
        if kind == "candidate_select"
        else {"selected_alternative_id": "c"}
    )
    after = {**before, **change, "revision": 5}
    request = {"expected_revision": 4, **change}
    response = {"session": after}
    row = {
        "binding_id": kind,
        "input_attributes": attributes,
        "input": {"select_value": "candidate"},
        "selection": {"request": request, "response": response},
        "session_before": before,
        "session_after": after,
        "workspace_before": {
            "geography": {"plans": plans},
            "draft_capabilities": {"candidate_choices": [{"candidate_id": "candidate"}]},
            "alternatives": [{"plan_id": name, "plan_content_hash": "hash-" + name} for name in ("b", "c")],
        },
        "selection_map": deepcopy(parent),
        "compare_maps": [
            {
                "renderedMapPlanId": plan["plan_id"],
                "renderedPlanContentHash": plan["content_hash"],
                "renderedGeometryHash": core._completed_race_geometry_hash(plan),
                "renderedMarkerIds": '["stop"]',
                "mapReady": "true",
                "mapTileCoverage": "1",
            }
            for plan in (plans[0], plans[2])
        ],
    }
    record = {
        "method": "POST",
        "url": "http://local/api/sessions/session/selection",
        "request_body_utf8": json.dumps(request),
        "response_body_utf8": json.dumps(response),
    }
    return row, [record], parent


@pytest.mark.parametrize(
    "kind", ["select_day", "text_select_stop", "text_select_leg", "candidate_select", "compare_select"]
)
@pytest.mark.parametrize(
    "fault",
    [None, "missing_http", "wrong_request", "wrong_response", "wrong_revision", "foreign_session", "draft_mutation"],
)
def test_selection_receipt_requires_exact_canonical_http(kind, fault):
    row, records, parent = selection_fixture(kind)
    if fault == "missing_http":
        records = []
    elif fault == "wrong_request":
        records[0]["request_body_utf8"] = "{}"
    elif fault == "wrong_response":
        records[0]["response_body_utf8"] = "{}"
    elif fault == "wrong_revision":
        row["session_before"]["revision"] = 3
    elif fault == "foreign_session":
        row["session_before"]["session_id"] = "foreign"
    elif fault == "draft_mutation":
        row["session_before"]["draft"] = ["unexpected"]
    if fault:
        with pytest.raises(gates.ReceiptError):
            gates.verify_selection_journey(row, records, parent)
    else:
        gates.verify_selection_journey(row, records, parent)


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_stop",
        "foreign_leg",
        "foreign_candidate",
        "same_compare_option",
        "wrong_map_id",
        "wrong_map_hash",
        "wrong_map_markers",
        "gray_compare_map",
    ],
)
def test_selection_cannot_substitute_unrelated_canonical_objects(fault):
    kind = {
        "foreign_stop": "select_day",
        "foreign_leg": "text_select_leg",
        "foreign_candidate": "candidate_select",
        "same_compare_option": "compare_select",
        "gray_compare_map": "compare_select",
    }.get(fault, "select_day")
    row, records, parent = selection_fixture(kind)
    if fault == "foreign_stop":
        row["input_attributes"]["data-stop"] = "foreign"
    elif fault == "foreign_leg":
        row["input_attributes"]["data-geo-route-leg"] = "foreign"
    elif fault == "foreign_candidate":
        row["input"]["select_value"] = "foreign"
    elif fault == "same_compare_option":
        row["session_before"]["selected_alternative_id"] = "c"
    elif fault == "gray_compare_map":
        row["compare_maps"][0]["mapTileCoverage"] = "0.3"
    else:
        key = {
            "wrong_map_id": "rendered_map_plan_id",
            "wrong_map_hash": "rendered_geometry_hash",
            "wrong_map_markers": "rendered_marker_ids",
        }[fault]
        row["selection_map"][key] = [] if fault == "wrong_map_markers" else "foreign"
    with pytest.raises(gates.ReceiptError):
        gates.verify_selection_journey(row, records, parent)


def test_real_input_plus_specific_semantic_result_is_required():
    _, binding, parent, event, observation, journey = general_fixture()
    gates.verify_general_journey(journey, binding, [observation], [event], parent)


def test_native_dialog_close_selector_binds_real_ancestor_not_pseudo_id():
    _, binding, parent, event, observation, journey = general_fixture()
    selector = '#info-dialog button[value="close"]'
    binding.update(binding_id="about_close", selectors=[selector])
    event["matches"] = [{"binding_id": "about_close", "selector": selector}]
    event["ancestors"] = [{"id": "", "attributes": {"value": "close"}}, {"id": "info-dialog", "attributes": {}}]
    observation.update(binding_id="about_close", selector=selector)
    journey.update(
        binding_id="about_close",
        selector=selector,
        outcome={"selector": "#info-dialog", "visible": False},
        dom_result={"visible": False},
    )
    gates.verify_general_journey(journey, binding, [observation], [event], parent)


@pytest.mark.parametrize("variant,open_state", [("open", True), ("closed", False)])
@pytest.mark.parametrize("fault", [None, "wrong_open", "hidden_summary", "foreign_plan"])
def test_text_route_details_bind_native_open_state_while_summary_remains_visible(variant, open_state, fault):
    _, binding, parent, event, observation, journey = general_fixture()
    selector = ".textual-map-alternative > summary"
    binding.update(binding_id="text_route_details", selectors=[selector], event="click")
    event.update(type="click", matches=[{"binding_id": "text_route_details", "selector": selector}])
    observation.update(binding_id="text_route_details", selector=selector, phase="text_route_before")
    journey.update(
        binding_id="text_route_details",
        selector=selector,
        phase="text_route",
        variant=variant,
        outcome={"selector": ".text-route-summary", "open": open_state},
        dom_result={
            "visible": True,
            "open": open_state,
            "attributes": {"data-plan-id": "parent", "data-plan-content-hash": "p-hash"},
        },
    )
    if fault == "wrong_open":
        journey["dom_result"]["open"] = not open_state
    elif fault == "hidden_summary":
        journey["dom_result"]["visible"] = False
    elif fault == "foreign_plan":
        journey["dom_result"]["attributes"]["data-plan-id"] = "foreign"
    if fault in {"wrong_open", "hidden_summary", "foreign_plan"}:
        with pytest.raises(gates.ReceiptError):
            gates.verify_general_journey(journey, binding, [observation], [event], parent)
    else:
        gates.verify_general_journey(journey, binding, [observation], [event], parent)


@pytest.mark.parametrize("kind", ["evidence_back_exact", "evidence_back_unavailable"])
@pytest.mark.parametrize("fault", [None, "heading", "wrong_branch", "wrong_plan", "wrong_hash", "wrong_certificate"])
def test_evidence_back_receipt_requires_actual_branch_and_exact_artifact(kind, fault):
    _, binding, parent, event, observation, journey = general_fixture()
    binding.update(binding_id=kind, selectors=["#evidence-back"])
    event["matches"] = [{"binding_id": kind, "selector": "#evidence-back"}]
    event["ancestors"] = [{"id": "evidence-back", "attributes": {}}]
    observation.update(binding_id=kind, selector="#evidence-back")
    exact = {"evidencePlanId": "child", "planContentHash": "child-hash", "evaluationId": "certificate"}
    evidence = {
        "heading": "Evidence and lineage" if kind == "evidence_back_exact" else "Evidence unavailable",
        "evidence": [exact] if kind == "evidence_back_exact" else [],
        "code": "evidence_option_not_selected",
    }
    journey.update(
        binding_id=kind,
        selector="#evidence-back",
        outcome=gates.GENERAL_OUTCOMES[kind],
        path=gates.GENERAL_OUTCOMES[kind]["path"],
        evidence_before=evidence,
        workspace_before={
            "alternative_evidence": {
                "child": {"plan_id": "child", "plan_content_hash": "child-hash", "certificate_id": "certificate"}
            }
        },
    )
    if fault == "heading":
        evidence["heading"] = "Other page"
    elif fault == "wrong_branch":
        evidence["evidence"] = [] if kind == "evidence_back_exact" else [exact]
    elif fault and kind == "evidence_back_exact":
        key = {"wrong_plan": "evidencePlanId", "wrong_hash": "planContentHash", "wrong_certificate": "evaluationId"}[
            fault
        ]
        exact[key] = "wrong"
    elif fault:
        evidence["code"] = None
    if fault:
        with pytest.raises(gates.ReceiptError):
            gates.verify_general_journey(journey, binding, [observation], [event], parent)
    else:
        gates.verify_general_journey(journey, binding, [observation], [event], parent)


@pytest.mark.parametrize(
    "fault",
    [
        "untrusted",
        "wrong_event",
        "wrong_binding",
        "wrong_selector",
        "wrong_target",
        "raw_log_missing",
        "outcome_lie",
        "hidden_result",
        "disabled_input",
        "invisible_input",
        "changed_plan",
        "changed_hash",
        "draft_mutation",
        "unknown_action",
    ],
)
def test_plausible_general_control_receipt_cannot_claim_false_execution(fault):
    _, binding, parent, event, observation, journey = general_fixture()
    events = [event]
    if fault == "untrusted":
        event["trusted"] = False
    elif fault == "wrong_event":
        event["type"] = "change"
    elif fault == "wrong_binding":
        event["matches"][0]["binding_id"] = "layers"
    elif fault == "wrong_selector":
        journey["selector"] = "#wrong"
    elif fault == "wrong_target":
        event["ancestors"][0]["id"] = "wrong"
    elif fault == "raw_log_missing":
        events = []
    elif fault == "outcome_lie":
        journey["outcome"]["visible"] = False
    elif fault == "hidden_result":
        journey["dom_result"]["visible"] = False
    elif fault == "disabled_input":
        observation["controls"][0]["disabled"] = True
    elif fault == "invisible_input":
        observation["controls"][0]["visible"] = False
    elif fault == "changed_plan":
        journey["accepted_plan_id"] = "other"
    elif fault == "changed_hash":
        journey["accepted_plan_content_hash"] = "other"
    elif fault == "draft_mutation":
        journey["draft"] = ["unexpected"]
    elif fault == "unknown_action":
        journey["binding_id"] = "unknown"
    with pytest.raises(gates.ReceiptError):
        gates.verify_general_journey(journey, binding, [observation], events, parent)


@pytest.mark.parametrize("fault", [None, "inventory_only", "disabled_without_capability", "one_alias_only"])
def test_control_gate_requires_bound_executed_journeys_not_presence(tmp_path, fault):
    source, binding, parent, event, observation, journey = general_fixture()
    if fault == "one_alias_only":
        binding["required_execution_selectors"] = ["#help-button", "#other-help"]
    completed = [journey] if fault not in {"inventory_only", "disabled_without_capability"} else []
    if fault == "disabled_without_capability":
        observation["controls"][0]["disabled"] = True
    prefix = "capability_controls/original/"
    suites = [
        bundle(
            tmp_path,
            "capability-controls",
            {
                prefix + "general_control_inventory.json": {
                    "schema_version": "general-control-inventory-v1",
                    "context": "original",
                    "presence_is_not_execution": True,
                    "absent_is_not_unsupported": True,
                    "observations": [observation],
                },
                prefix + "general_control_journeys.json": {
                    "schema_version": "general-control-journeys-v1",
                    "context": "original",
                    "inventory_is_not_execution": True,
                    "journeys": completed,
                },
                prefix + "general_control_events.json": {
                    "schema_version": "general-control-events-v1",
                    "context": "original",
                    "trusted_event_is_not_semantic_completion": True,
                    "events": [event],
                },
                prefix + "capability_snapshot.json": {"capabilities": {"draft_capabilities": {"operations": {}}}},
                prefix + "browser_assertions.json": {"parent": parent},
            },
        )
    ]
    registry = {
        "controls": [{key: "legacy source metadata" for key in gates.CONTROL_FIELDS}],
        "required_contexts": ["original"],
        "source_bindings": [binding],
    }
    result = gates.control_gate(gates.Receipts(tmp_path, suites), registry, [source])
    assert result["passed"] is (fault is None)
    if fault:
        assert result["missing_control_receipts"] and result["disabled_controls"] == []


def existing_proof_bundles(directory, fault=None):
    parent = {"active_plan_id": "parent"}
    child = {
        "active_plan_id": "child",
        "rendered_map_plan_id": "child",
        "rendered_marker_ids": ["pfeiffer_big_sur_state_park"],
    }
    url = "http://127.0.0.1:8127/api/sessions/s/map/poi-candidates?day=5"
    candidates = ["pfeiffer_big_sur_state_park", "point_lobos_state_natural_reserve"]
    poi = {
        "trace.json": {
            "assertions_passed": True,
            "parent_plan_id": "parent",
            "child_plan_id": "child",
            "child_plan_content_hash": "ch",
            "geometry_hash": "geometry",
        },
        "browser_assertions.json": {
            "assertions_passed": True,
            "candidate_request_url": url,
            "candidate_status": 200,
            "candidate_ids": candidates,
            "llm_call_made": False,
            "parent_map": parent,
            "staged_map": parent,
            "rendered_map": child,
        },
        "browser_preview_response.json": {
            "proposal": {"state": "eligible", "child_plan_id": "child", "child_plan_content_hash": "ch"}
        },
        "browser_poi_candidate_request.json": {"method": "GET", "url": url, "llm_call_made": False},
        "browser_poi_candidate_response.json": {"candidates": [{"candidate_id": name} for name in candidates]},
    }
    recovery = {
        "trace.json": {
            "schema_version": "map-config-recovery-browser-v2",
            "context": "accepted_child",
            "assertions_passed": True,
            "active_plan_id": "child",
            "active_plan_content_hash": "ch",
            "child_plan_id": "child",
            "child_plan_content_hash": "ch",
            "rendered_geometry_hash": "geometry",
        },
        "persistent_failure_no_ready_claim_diagnostics.json": {
            "retry_visible": True,
            "map_stage": {},
            "geographic_map": {},
        },
        "recovered_after_explicit_retry_diagnostics.json": {
            "map_stage": {
                "renderedMapPlanId": "child",
                "activePlanId": "child",
                "activePlanContentHash": "ch",
                "renderedGeometryHash": "geometry",
            }
        },
        "accept_response.json": {"session": {"accepted_plan_id": "child"}},
        "browser_requests.json": [{"method": "GET", "phase": "explicit_retry_after_fault_removed"}],
    }
    build = {
        "browser_assertions.json": {
            "schema_version": "build-coherence-browser-v1",
            "assertions_passed": True,
            "explicit_reload_session_preserved": True,
            "blocked_mutation_state_unchanged": True,
            "first_build_id": "a",
            "second_build_id": "b",
            "accepted_plan_id": "child",
            "session": {"session_id": "s"},
        },
        "map_checkpoints.json": {"accepted_child": child, "accepted_child_after_update": child},
        "http_index.json": [
            {
                "phase": "generation_b_reloaded",
                "method": "GET",
                "url": "http://127.0.0.1:8127/api/sessions/s",
                "file": "restored.json",
            }
        ],
        "stale_write_state_hashes.json": {"before": {"a": "hash"}, "after": {"a": "hash"}},
        "reload_control_input.json": {
            "trusted": True,
            "selector": "#reload-current-build",
            "client_build": "a",
            "path": "/app/itinerary",
            "title": "Reload",
            "clicked_at": "2026-09-08T10:00:01Z",
        },
        "product_client_build_mismatch_visible_recovery.json": {
            "text": "Reload",
            "captured_at": "2026-09-08T10:00:00Z",
        },
        "generation_b/active_plan_after_update.readiness.json": {"screenshot_started_at": "2026-09-08T10:00:02Z"},
        "restored.json": {
            "response_status_code": 200,
            "response_body_utf8": json.dumps({"session": {"accepted_plan_id": "child"}}),
        },
    }
    if fault == "discovery_staged_parent_changed":
        poi["browser_assertions.json"]["staged_map"] = {"active_plan_id": "other"}
    elif fault == "discovery_candidate_unbound":
        poi["browser_poi_candidate_response.json"]["candidates"] = []
    elif fault == "retry_mutated":
        recovery["browser_requests.json"][0]["method"] = "POST"
    elif fault == "retry_old_map":
        recovery["recovered_after_explicit_retry_diagnostics.json"]["map_stage"]["renderedMapPlanId"] = "parent"
    elif fault == "reload_same_build":
        build["browser_assertions.json"]["second_build_id"] = "a"
    elif fault == "reload_unbound_http":
        build["restored.json"]["response_body_utf8"] = json.dumps({"session": {"accepted_plan_id": "parent"}})
    build["map_checkpoints.json"]["initial_parent"] = child
    build["map_checkpoints.json"]["active_plan_after_update"] = child
    build_contexts = {
        context + "/" + name: ({**value, "context": context} if name == "browser_assertions.json" else value)
        for context in ("original", "accepted_child")
        for name, value in build.items()
    }
    return [
        bundle(directory, mode, {prefix + name: value for name, value in values.items()})
        for mode, prefix, values in (
            ("vertical-slice", "poi_discovery/", poi),
            ("map-recovery", "map_initialization_recovery/accepted_child/", recovery),
            ("build-coherence", "build_coherence/", build_contexts),
        )
    ]


def test_existing_verified_journeys_project_only_exact_original_or_accepted_controls(tmp_path):
    proofs, missing = gates.existing_control_proofs(gates.Receipts(tmp_path, existing_proof_bundles(tmp_path)))
    assert len(proofs) == 7
    assert missing and all(
        row["bindings"] != ["reload_build"]
        and not (row["bindings"] == ["retry_map_runtime"] and row["context"] == "accepted_child")
        for row in missing
    )
    assert any(row["bindings"] == ["retry_map_runtime"] and row["context"] == "original" for row in missing)
    assert ("retry_map_runtime", "accepted_child") in proofs and ("retry_map_runtime", "original") not in proofs
    assert ("reload_build", "accepted_child") in proofs and ("reload_build", "original") in proofs
    assert ("use_replacement", "original") in proofs and ("use_replacement", "accepted_child") not in proofs
    assert all(row["evidence"] for row in proofs.values())


@pytest.mark.parametrize(
    "fault,mode",
    [
        ("discovery_staged_parent_changed", "vertical-slice"),
        ("discovery_candidate_unbound", "vertical-slice"),
        ("retry_mutated", "map-recovery"),
        ("retry_old_map", "map-recovery"),
        ("reload_same_build", "build-coherence"),
        ("reload_unbound_http", "build-coherence"),
    ],
)
def test_existing_suite_registration_does_not_hide_broken_action_result(tmp_path, fault, mode):
    proofs, missing = gates.existing_control_proofs(gates.Receipts(tmp_path, existing_proof_bundles(tmp_path, fault)))
    assert mode in {row["mode"] for row in missing}
    assert not any(row["mode"] == mode for row in proofs.values())
