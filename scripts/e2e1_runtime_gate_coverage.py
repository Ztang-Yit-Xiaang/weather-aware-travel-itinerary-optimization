"""Receipt-bound aggregate requirements; no browser/provider dispatch or inferred passes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_GATE = "runtime_enabled_control_inventory_receipt_missing"
CONVERSATION_GATE = "conversation_memory_cancel_retry_capacity_runtime_journeys_pending"
NEGATIVE_GATE = "cross_family_and_draft_only_negative_runtime_journeys_pending"
GEOGRAPHY_GATE = "eligible_preview_missing_or_foreign_geography_runtime_pending"
LIVE_GATE = "billed_website_chat_journey_not_part_of_deterministic_suite"
CONTROL_FIELDS = {
    "action_id",
    "selector",
    "handler",
    "state_action",
    "endpoint",
    "service",
    "artifacts",
    "ui_selector",
    "map_effect",
}


class ReceiptError(ValueError):
    pass


class Receipts:
    """Use only artifacts still matching the independently verified child manifest."""

    def __init__(self, directory, suites):
        self.directory = directory
        self.suites = {row["mode"]: row for row in suites}
        self.used = []

    def read(self, mode, relative):
        suite = self.suites.get(mode, {})
        verification = suite.get("verification") or {}
        if self.directory is None or suite.get("status") != "passed" or verification.get("verified") is not True:
            raise ReceiptError("independently_passed_suite_missing")
        root = (self.directory / "suites" / mode).resolve()
        manifest_path = root / "file_io_manifest.json"
        data = manifest_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != verification.get("manifest_sha256"):
            raise ReceiptError("verified_child_manifest_changed")
        manifest = json.loads(data)
        if (
            type(manifest.get("provider_calls")) is not int
            or manifest["provider_calls"] != 0
            or type(verification.get("provider_calls")) is not int
            or verification["provider_calls"] != 0
        ):
            raise ReceiptError("deterministic_receipt_provider_scope_invalid")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ReceiptError("receipt_path_escape")
        records = [row for row in manifest["files"] if row["path"] == relative]
        value = path.read_bytes()
        if (
            len(records) != 1
            or records[0]["sha256"] != hashlib.sha256(value).hexdigest()
            or records[0]["size_bytes"] != len(value)
        ):
            raise ReceiptError("receipt_artifact_binding_invalid")
        self.used.append(
            {
                "mode": mode,
                "artifact": path.relative_to(self.directory.resolve()).as_posix(),
                "sha256": records[0]["sha256"],
                "verified_manifest_sha256": verification["manifest_sha256"],
            }
        )
        return json.loads(value)


def require(condition, code):
    if not condition:
        raise ReceiptError(code)


def source_controls(root=ROOT):
    """Literal event bindings are source requirements, not proof of UI coverage."""
    patterns = [
        re.compile(r"""\$\(\s*(['"])([^'"]+)\1\s*\)\??\.addEventListener\(\s*(['"])(click|submit|change)\3"""),
        re.compile(
            r"""\$\$\(\s*(['"])([^'"]+)\1\s*\)\.forEach\([^\n]+?addEventListener\(\s*(['"])(click|submit|change)\3"""
        ),
        re.compile(
            r"""\.querySelector\(\s*(['"])([^'"]+)\1\s*\)\??\.addEventListener\(\s*(['"])(click|submit|change)\3"""
        ),
    ]
    found = []
    directory = root / "src/itinerary_system/product_app/static/js"
    if not (directory / "app.js").is_file() or not (directory / "copilot.js").is_file():
        raise ValueError("control_source_inventory_unavailable")
    event = re.compile(r"""addEventListener\(\s*(['"])(click|submit|change)\1""")
    for path in sorted(directory.glob("*.js")):
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            matched = False
            for pattern in patterns:
                for match in pattern.finditer(line):
                    matched = True
                    found.append(
                        {
                            "source_file": relative,
                            "line": number,
                            "selector": match[2],
                            "event": match[4],
                            "source_anchor": line.strip(),
                        }
                    )
            # Variable-held controls and generated action buttons must be mapped
            # explicitly, not silently omitted by a literal-selector extractor.
            if not matched:
                for match in event.finditer(line):
                    found.append(
                        {
                            "source_file": relative,
                            "line": number,
                            "selector": None,
                            "event": match[2],
                            "binding_kind": "unresolved_dynamic_binding",
                            "source_text": line.strip(),
                            "source_anchor": line.strip(),
                        }
                    )
    occurrences = {}
    for row in found:
        key = (row["source_file"], row["source_anchor"], row["event"])
        row["anchor_occurrence"] = occurrences.get(key, 0)
        occurrences[key] = row["anchor_occurrence"] + 1
    return found


def binding_inventory(registry, inventory):
    """Match exact source anchors; a hand-written alias is never inferred from CSS."""
    mappings = registry.get("source_bindings", [])
    required = {
        "binding_id",
        "source_file",
        "source_anchor",
        "anchor_occurrence",
        "event",
        "source_selector",
        "selectors",
        "owner",
        "action_semantics",
        "availability",
        "kind",
        "contexts",
        "execution_credit",
    }
    require(
        isinstance(mappings, list)
        and mappings
        and all(
            isinstance(row, dict)
            and required <= row.keys()
            and all(
                row[key]
                for key in ("binding_id", "source_file", "source_anchor", "owner", "action_semantics", "availability")
            )
            and isinstance(row["selectors"], list)
            and row["selectors"]
            and all(isinstance(selector, str) and selector for selector in row["selectors"])
            and row["contexts"] == registry["required_contexts"]
            and row["kind"] in {"control", "responsive_lifecycle"}
            for row in mappings
        ),
        "source_binding_mapping_incomplete",
    )
    require(len({row["binding_id"] for row in mappings}) == len(mappings), "source_binding_id_duplicate")

    def key(row):
        return row["source_file"], row["source_anchor"], row["event"], row["anchor_occurrence"]

    require(len({key(row) for row in mappings}) == len(mappings), "source_binding_anchor_duplicate")
    indexed = {key(row): row for row in mappings}
    actual = {key(row): row for row in inventory}
    unmatched = [
        row for row in inventory if key(row) not in indexed or indexed[key(row)]["source_selector"] != row["selector"]
    ]
    stale = [row for row in mappings if key(row) not in actual]
    return mappings, unmatched, stale


def conversation_gate(receipts):
    cases = {"clarification_reload", "preference_memory", "capacity_turns", "capacity_bytes", "fake_sdk_cancel"}
    cases.update(name + "_accepted_child" for name in tuple(cases) if name != "clarification_reload")
    for case in sorted(cases):
        original_name = case.removesuffix("_accepted_child")
        trace = receipts.read("conversation", f"conversation/{case}/trace.json")
        scope = receipts.read("conversation", f"conversation/{case}/provider_scope.json")
        require(
            trace["schema_version"] == "conversation-browser-v1"
            and trace["scenario"] == case
            and trace["assertions_passed"] is True
            and scope["real_chat_endpoint"] is True
            and scope["internal_api_mocked"] is False
            and scope["live_provider_call_count"] == 0
            and scope["live_openai_connectivity_proven"] is False
            and scope["fake_sdk_attempt_count"] == int(original_name == "fake_sdk_cancel"),
            "conversation_scope_not_bound",
        )
        if original_name != case:
            baseline = receipts.read("conversation", f"conversation/{case}/accepted_baseline_artifacts.json")
            final = receipts.read("conversation", f"conversation/{case}/context_control_final.json")
            require(
                trace["context"] == baseline["context"] == "accepted_child"
                and baseline["accepted_map"] == trace["initial"] == trace["final"]
                and baseline["original"]["active_plan_id"]
                != final["accepted_plan_id"]
                == trace["initial"]["active_plan_id"]
                and final["accepted_plan_content_hash"] == trace["initial"]["active_plan_content_hash"]
                and final["draft_operations"] == [],
                "accepted_conversation_control_context_missing",
            )
        if case.startswith("capacity_"):
            require(
                trace["blocked_attempt_delta"] == 0 and trace["old_conversation_id"] != trace["new_conversation_id"],
                "capacity_dispatched_or_failed_to_reset",
            )
        if original_name == "fake_sdk_cancel":
            before = receipts.read("conversation", f"conversation/{case}/dispatch_after_cancel.json")
            after = receipts.read("conversation", f"conversation/{case}/dispatch_after_retry.json")
            require(
                before == after and before["attempt_count"] == 1 and before["outcome"] == "unknown",
                "cancel_retry_attempt_not_preserved",
            )
    # Keep the distinct accepted-plan chat continuation gate in addition to
    # both-context Delete all, capacity and recovery control journeys above.
    accepted = receipts.read("chat-closure", "chat_closure/accepted_child/browser_assertions.json")
    context = receipts.read("chat-closure", "chat_closure/accepted_child/context_accepted_transition.json")
    require(
        accepted["assertions_passed"] is True
        and accepted["context"] == "accepted_child"
        and context["conversation_memory"]["plan_id"] == context["accepted_plan_id"]
        and context["conversation_memory"]["plan_content_hash"] == context["accepted_plan_content_hash"],
        "accepted_memory_context_missing",
    )
    return {
        "scope": "real website lifecycle; fake SDK transport cancellation; accepted-plan memory binding",
        "live_openai_connectivity_proven": False,
    }


def negative_gate(receipts, variants):
    from verify_e2e1_capability_controls import cross_family_negative
    from verify_e2e1_vertical_slice import E2E1VerificationError

    for context in ("original", "accepted_child"):
        prefix = "capability_controls/" + context + "/"
        assertions = receipts.read("capability-controls", prefix + "browser_assertions.json")
        snapshot = receipts.read("capability-controls", prefix + "capability_snapshot.json")
        rows = receipts.read("capability-controls", prefix + "operation_assertions.json")
        cross = receipts.read("capability-controls", prefix + "cross_family_assertions.json")
        typed = snapshot["capabilities"]["typed_edit_capabilities"]
        expected = {
            (kind, row["id"])
            for kind, cap in typed["operations"].items()
            for row in variants(kind, cap, typed)
            if row["feedback_tier"] == "draft_only"
        }
        require(
            assertions["assertions_passed"] is True
            and rows
            and len(rows) == len(expected)
            and {(row["operation"], row["variant"]) for row in rows} == expected
            and all(
                row["assertions_passed"] is True and row["context"] == context and row["feedback_tier"] == "draft_only"
                for row in rows
            ),
            "draft_only_negative_variants_missing",
        )
        require(
            cross == assertions["cross_family_cases"]
            and [row["first"] for row in cross] == (["duration", "replacement"] if context == "original" else [])
            and all(row["context"] == context for row in cross),
            "cross_family_directions_missing",
        )
        for row in cross:
            try:
                cross_family_negative(row, typed)
            except E2E1VerificationError as error:
                raise ReceiptError(str(error)) from error
    return {
        "scope": "all observed draft-only variants; both original-context unsupported-duration/replacement directions blocked before second staging",
        "forced_mixed_draft_preview_claimed": False,
    }


GENERAL_OUTCOMES = {
    "close_map_editor": {"selector": "#map-edit-dialog", "visible": False},
    "inspect_stop": {"selector": '[data-editor-kind="stop"]', "visible": True},
    "inspect_route": {"selector": '[data-editor-kind="route"]', "visible": True},
    "help": {"selector": "#info-dialog", "visible": True},
    "about_close": {"selector": "#info-dialog", "visible": False},
    "layers": {"selector": "#toast-region", "text": "original route"},
    "map_edit": {"selector": "#map-edit-button", "attribute": "aria-pressed", "value": "true"},
    "map_select": {"selector": "#map-select-button", "attribute": "aria-pressed", "value": "true"},
    "context_edit": {"selector": "#map-edit-button", "attribute": "aria-pressed", "value": "true"},
    "context_select": {"selector": "#map-select-button", "attribute": "aria-pressed", "value": "true"},
    "open_copilot": {"selector": "#copilot-dock", "visible": True},
    "mobile_copilot": {"selector": "#copilot-dock", "visible": True},
    "close_copilot": {"selector": "#copilot-dock", "visible": False},
    "transcript_open": {"selector": "#transcript-settings-dialog", "visible": True},
    "transcript_close": {"selector": "#transcript-settings-dialog", "visible": False},
    "repairs_back": {"path": "/app/itinerary"},
    "compare_back": {"path": "/app/itinerary"},
    "evidence_back_unavailable": {"path": "/app/compare"},
    "evidence_back_exact": {"path": "/app/itinerary"},
    "open_compare": {"path": "/app/compare"},
    "open_evidence": {"path": "/app/evidence"},
    "context_evidence": {"path": "/app/evidence"},
    "compare_evidence": {"path": "/app/evidence"},
    "context_back_compare": {"path": "/app/compare"},
    "compare_choose": {"path": "/app/itinerary"},
    "context_choose": {"path": "/app/itinerary"},
    "destination_day_select": {"selector": "#destination-day", "value_text": "7"},
}


def verify_general_journey(row, binding, observations, events, parent):
    """A trusted click is necessary, not sufficient: verify the specific result."""
    kind, selector = row["binding_id"], row["selector"]
    require(selector in binding["selectors"], "general_journey_selector_unmapped")
    expected = GENERAL_OUTCOMES.get(kind)
    if kind == "navigate_global":
        match = re.fullmatch(
            r'(?:\.nav-item|\.mobile-nav button)\[data-route="(/app/(?:itinerary|map|repairs|compare|evidence))"\]',
            selector,
        )
        require(match is not None, "navigation_selector_not_bound")
        expected = {"path": match[1]}
    elif kind == "compare_mobile_tab":
        match = re.fullmatch(r'\[data-compare-mobile-view="(baseline|option)"\]', selector)
        require(match is not None and row["compare_visible_panes"] == [match[1]], "mobile_compare_visibility_invalid")
        expected = {"selector": selector, "attribute": "aria-pressed", "value": "true"}
    elif kind == "text_route_details":
        require(row["variant"] in {"open", "closed"}, "text_route_action_invalid")
        expected = {"selector": ".text-route-summary", "open": row["variant"] == "open"}
        require(
            row["dom_result"]["attributes"].get("data-plan-id") == parent["active_plan_id"]
            and row["dom_result"]["attributes"].get("data-plan-content-hash") == parent["active_plan_content_hash"],
            "text_route_identity_invalid",
        )
        require(row["dom_result"]["visible"] is True, "text_route_summary_not_visible")
    elif kind == "quick_prompt":
        before = [
            control
            for item in observations
            if item["phase"] == row["phase"] + "_before" and item["binding_id"] == kind
            for control in item["controls"]
            if control["visible"] is True and control["attributes"].get("data-prompt")
        ]
        values = {control["attributes"]["data-prompt"] for control in before}
        require(row["outcome"].get("value_text") in values, "quick_prompt_not_bound")
        expected = {"selector": "#copilot-input", "value_text": row["outcome"]["value_text"]}
    elif kind in {"select_day", "text_select_stop", "text_select_leg", "candidate_select", "compare_select"}:
        attributes = row["input_attributes"]
        if kind == "candidate_select":
            expected = {"selector": "#registered-candidate", "value_text": row["input"]["select_value"]}
        else:
            key = {
                "select_day": "data-day",
                "text_select_stop": "data-geo-stop",
                "text_select_leg": "data-geo-route-leg",
                "compare_select": "data-compare-option",
            }[kind]
            result_selector = (
                f'.day-card[data-day="{attributes[key]}"]' if kind == "select_day" else f'[{key}="{attributes[key]}"]'
            )
            expected = {"selector": result_selector, "attribute": "aria-pressed", "value": "true"}
    if kind in {"compare_choose", "context_choose"}:
        from verify_e2e1_vertical_slice import _completed_race_geometry_hash

        plan, ack = row["selected_plan"], row["selected_map"]
        require(
            ack["active_plan_id"] == ack["rendered_map_plan_id"] == plan["plan_id"]
            and ack["active_plan_content_hash"] == plan["content_hash"]
            and ack["rendered_geometry_hash"] == _completed_race_geometry_hash(plan)
            and set(ack["rendered_marker_ids"])
            == {feature["properties"]["stop_id"] for feature in plan["stops"]["features"]},
            "chosen_map_not_exact_selected_plan",
        )
    if kind.startswith("evidence_back_"):
        evidence = row["evidence_before"]
        if kind == "evidence_back_unavailable":
            require(
                evidence["heading"] == "Evidence unavailable"
                and evidence["evidence"] == []
                and isinstance(evidence["code"], str)
                and evidence["code"].startswith(("evidence_", "compare_")),
                "evidence_unavailable_branch_not_observed",
            )
        else:
            require(
                evidence["heading"] == "Evidence and lineage" and len(evidence["evidence"]) == 1,
                "evidence_exact_branch_not_observed",
            )
            shown = evidence["evidence"][0]
            workspace = row["workspace_before"]
            exact = workspace["alternative_evidence"].get(shown["evidencePlanId"])
            require(
                exact is not None
                and exact["plan_id"] == shown["evidencePlanId"]
                and exact["plan_content_hash"] == shown["planContentHash"]
                and exact["certificate_id"] == shown["evaluationId"],
                "evidence_exact_identity_unbound",
            )
    require(
        expected is not None and row["outcome"] == expected and row["assertions_passed"] is True,
        "general_semantic_contract_missing_or_changed",
    )
    require(
        row["context"] in binding["contexts"]
        and row["accepted_plan_id"] == parent["active_plan_id"]
        and row["accepted_plan_content_hash"] == parent["active_plan_content_hash"]
        and row["draft"] == [],
        "general_control_parent_identity_changed",
    )
    # All event slices must occur unchanged in the separately recorded raw log.
    require(row["trusted_events"] and all(event in events for event in row["trusted_events"]), "general_events_unbound")
    matches = [
        event
        for event in row["trusted_events"]
        if event["trusted"] is True
        and event["type"] == binding["event"]
        and any(match["binding_id"] == kind and match["selector"] == selector for match in event["matches"])
    ]
    require(matches, "general_trusted_input_missing")
    if selector.startswith("#"):
        target_id = re.match(r"#([A-Za-z0-9_-]+)", selector)
        require(
            target_id is not None
            and any(any(ancestor["id"] == target_id[1] for ancestor in event["ancestors"]) for event in matches),
            "general_event_target_mismatch",
        )
    if kind == "quick_prompt":
        require(
            any(
                any(
                    ancestor["attributes"].get("data-prompt") == expected["value_text"]
                    for ancestor in event["ancestors"]
                )
                for event in matches
            ),
            "quick_prompt_event_value_mismatch",
        )
    if kind in {"select_day", "text_select_stop", "text_select_leg", "candidate_select", "compare_select"}:
        require(
            any(
                any(ancestor["attributes"] == row["input_attributes"] for ancestor in event["ancestors"])
                for event in matches
            ),
            "selection_trusted_target_unbound",
        )
    if "path" in expected:
        require(row["path"] == expected["path"], "general_navigation_result_mismatch")
    else:
        result = row["dom_result"]
        if "visible" in expected:
            require(result["visible"] is expected["visible"], "general_visibility_result_mismatch")
        if "attribute" in expected:
            require(
                result["attributes"].get(expected["attribute"]) == expected["value"],
                "general_attribute_result_mismatch",
            )
        if "text" in expected:
            require(expected["text"] in result["text"], "general_text_result_mismatch")
        if "value_text" in expected:
            require(result["value"] == expected["value_text"], "general_composer_result_mismatch")
        if "open" in expected:
            require(result["open"] is expected["open"], "general_open_state_mismatch")
    require(
        any(
            item["binding_id"] == kind
            and item["phase"] == row["phase"] + "_before"
            and any(control["visible"] is True and control["disabled"] is False for control in item["controls"])
            for item in observations
        ),
        "general_visible_enabled_input_missing",
    )


def verify_selection_journey(row, records, parent):
    """Bind native selection to the exact POST, canonical state and rendered map."""
    kind = row["binding_id"]
    if kind not in {"select_day", "text_select_stop", "text_select_leg", "candidate_select", "compare_select"}:
        return
    selection = row["selection"]
    posts = [record for record in records if record["method"] == "POST" and record["url"].endswith("/selection")]
    require(
        len(posts) == 1
        and json.loads(posts[0]["request_body_utf8"]) == selection["request"]
        and json.loads(posts[0]["response_body_utf8"]) == selection["response"],
        "selection_exact_http_unbound",
    )
    before, after, request = row["session_before"], row["session_after"], selection["request"]
    workspace = row["workspace_before"]
    require(
        after == selection["response"]["session"]
        and before["session_id"] == after["session_id"]
        and request["expected_revision"] == before["revision"]
        and after["revision"] > before["revision"]
        and after["accepted_plan_id"] == before["accepted_plan_id"] == parent["active_plan_id"]
        and after["draft"] == before["draft"] == [],
        "selection_session_identity_unbound",
    )
    plans = {plan["plan_id"]: plan for plan in workspace["geography"]["plans"]}
    plan = plans[parent["active_plan_id"]]
    attrs = row["input_attributes"]
    if kind in {"select_day", "text_select_stop"}:
        stop = attrs["data-stop" if kind == "select_day" else "data-geo-stop"]
        day = int(attrs["data-day" if kind == "select_day" else "data-geo-day"])
        require(
            any(
                feature["properties"]["stop_id"] == stop and feature["properties"]["day"] == day
                for feature in plan["stops"]["features"]
            ),
            "selection_stop_not_in_parent",
        )
        expected = {"selected_stop_id": stop, "selected_day": day, "selected_segment_id": None}
    elif kind == "text_select_leg":
        leg = attrs["data-geo-route-leg"]
        features = [
            feature for feature in plan["validated_legs"]["features"] if feature["properties"]["route_leg_id"] == leg
        ]
        require(len(features) == 1, "selection_leg_not_in_parent")
        expected = {
            "selected_segment_id": leg,
            "selected_stop_id": None,
            "selected_day": features[0]["properties"]["day"],
        }
    elif kind == "candidate_select":
        candidate = row["input"]["select_value"]
        require(
            candidate
            in {
                item.get("candidate_id") or item.get("stop_id") or item.get("id")
                for item in workspace["draft_capabilities"]["candidate_choices"]
            },
            "selection_candidate_not_in_capability",
        )
        expected = {"selected_candidate_id": candidate}
    else:
        selected = attrs["data-compare-option"]
        require(
            selected != before["selected_alternative_id"]
            and selected in plans
            and len(
                [
                    option
                    for option in workspace["alternatives"]
                    if option["plan_id"] == selected and option["plan_content_hash"] == plans[selected]["content_hash"]
                ]
            )
            == 1,
            "selection_compare_not_changed_or_canonical",
        )
        expected = {"selected_alternative_id": selected}
    require(
        all(request.get(key) == value and after.get(key) == value for key, value in expected.items()),
        "selection_semantics_invalid",
    )
    from verify_e2e1_vertical_slice import _completed_race_geometry_hash

    if kind == "compare_select":
        maps = row["compare_maps"]
        require(
            len(maps) == 2 and {data.get("renderedMapPlanId") for data in maps} == {parent["active_plan_id"], selected},
            "selection_compare_map_universe_invalid",
        )
        for data in maps:
            shown = plans[data["renderedMapPlanId"]]
            require(
                data["mapReady"] == "true"
                and 0.95 <= float(data["mapTileCoverage"]) <= 1
                and data["renderedPlanContentHash"] == shown["content_hash"]
                and data["renderedGeometryHash"] == _completed_race_geometry_hash(shown)
                and set(json.loads(data["renderedMarkerIds"]))
                == {feature["properties"]["stop_id"] for feature in shown["stops"]["features"]},
                "selection_compare_map_identity_invalid",
            )
    else:
        ack = row["selection_map"]
        require(
            all(
                ack[key] == parent[key]
                for key in (
                    "active_plan_id",
                    "active_plan_content_hash",
                    "rendered_map_plan_id",
                    "rendered_geometry_hash",
                    "rendered_marker_ids",
                )
            ),
            "selection_changed_parent_map",
        )


def verify_compare_responsive(rows, context, parent, plans):
    from verify_e2e1_vertical_slice import _completed_race_geometry_hash

    require(
        len(rows) == 1
        and rows[0]["context"] == context
        and rows[0]["assertions_passed"] is True
        and rows[0]["accepted_plan_id"] == parent["active_plan_id"]
        and rows[0]["accepted_plan_content_hash"] == parent["active_plan_content_hash"],
        "responsive_scope_invalid",
    )
    views = rows[0]["views"]
    require(
        [(view["width"], view["height"]) for view in views] == [(1440, 900), (390, 844), (1440, 900)],
        "responsive_viewport_sequence_invalid",
    )
    known = {plan["plan_id"]: plan for plan in plans}
    identities = []
    for index, view in enumerate(views):
        require(
            view["mobile_toggle_visible"] is (index == 1)
            and [row["pane"] for row in view["maps"]] == ["baseline", "option"]
            and [row["pane"] for row in view["maps"] if row["visible"]]
            == (["option"] if index == 1 else ["baseline", "option"]),
            "responsive_visible_panes_invalid",
        )
        current = {}
        for row in view["maps"]:
            if not row["visible"]:
                continue
            data = row["dataset"]
            plan = known.get(data.get("renderedMapPlanId"))
            require(
                plan is not None
                and data["mapReady"] == "true"
                and 0.95 <= float(data["mapTileCoverage"]) <= 1
                and data["renderedPlanContentHash"] == plan["content_hash"]
                and data["renderedGeometryHash"] == _completed_race_geometry_hash(plan)
                and set(json.loads(data["renderedMarkerIds"]))
                == {feature["properties"]["stop_id"] for feature in plan["stops"]["features"]},
                "responsive_map_not_canonical_or_ready",
            )
            current[row["pane"]] = (plan["plan_id"], plan["content_hash"], data["renderedGeometryHash"])
        identities.append(current)
    require(
        identities[0] == identities[2]
        and identities[1]["option"] == identities[0]["option"]
        and identities[0]["baseline"][0] == parent["active_plan_id"],
        "responsive_plan_changed",
    )


def existing_control_proofs(receipts):
    """Project specific already-verified browser paths, never every control in a suite."""
    proofs, unavailable = {}, []
    # Derive aliases individually from actual native input and the exact HTTP operation.
    # A suite passing does not imply every selector or the opposite plan context ran.
    for mode, folder in (("capability-controls", "capability_controls"), ("evaluated-variants", "evaluated_variants")):
        for context in ("original", "accepted_child"):
            start = len(receipts.used)
            try:
                from verify_e2e1_control_actions import verify_rows

                base = folder + "/" + context + "/"
                rows = receipts.read(mode, base + "control_action_receipts.json")
                records = [
                    receipts.read(mode, base + row["file"]) for row in receipts.read(mode, base + "http_index.json")
                ]
                for detail in verify_rows(rows, records, context):
                    proof = {
                        **detail,
                        "mode": mode,
                        "variants": ["default"],
                        "detail": detail,
                        "evidence": receipts.used[start:],
                    }
                    key = (detail["binding_id"], detail["context"])
                    if key not in proofs:
                        proofs[key] = {**proof, "receipts": []}
                    proofs[key]["receipts"].append(proof)
            except (ReceiptError, OSError, KeyError, TypeError, ValueError, AttributeError, RuntimeError):
                unavailable.append(
                    {
                        "mode": mode,
                        "context": context,
                        "bindings": ["exact_draft_control_aliases"],
                        "reason": "existing_exact_control_receipt_unavailable",
                    }
                )

    def project(mode, context, bindings, verify):
        start = len(receipts.used)
        try:
            detail = verify(lambda name: receipts.read(mode, name))
            for binding_id, selector in bindings.items():
                spec = {"selector": selector, "variants": ["default"]} if isinstance(selector, str) else selector
                proof = {
                    "binding_id": binding_id,
                    "context": context,
                    **spec,
                    "mode": mode,
                    "detail": detail,
                    "evidence": receipts.used[start:],
                }
                key = (binding_id, context)
                if key not in proofs:
                    proofs[key] = {**proof, "receipts": []}
                proofs[key]["receipts"].append(proof)
        except (ReceiptError, OSError, KeyError, TypeError, ValueError, AttributeError, RuntimeError):
            unavailable.append(
                {
                    "mode": mode,
                    "context": context,
                    "bindings": sorted(bindings),
                    "reason": "existing_exact_control_receipt_unavailable",
                }
            )

    def discovery(read):
        base = "poi_discovery/"
        trace, browser, response, request, candidates = (
            read(base + name)
            for name in (
                "trace.json",
                "browser_assertions.json",
                "browser_preview_response.json",
                "browser_poi_candidate_request.json",
                "browser_poi_candidate_response.json",
            )
        )
        child = response["proposal"]
        require(
            trace["assertions_passed"] is True
            and browser["assertions_passed"] is True
            and request["method"] == "GET"
            and "/map/poi-candidates?" in request["url"]
            and request["url"] == browser["candidate_request_url"]
            and browser["candidate_status"] == 200
            and request["llm_call_made"] is False
            and browser["llm_call_made"] is False
            and {row["candidate_id"] for row in candidates["candidates"]} == set(browser["candidate_ids"]),
            "existing_discovery_http_invalid",
        )
        require(
            browser["parent_map"] == browser["staged_map"]
            and child["state"] == "eligible"
            and trace["child_plan_id"] == child["child_plan_id"] == browser["rendered_map"]["rendered_map_plan_id"]
            and trace["parent_plan_id"] == browser["parent_map"]["active_plan_id"]
            and trace["child_plan_content_hash"] == child["child_plan_content_hash"]
            and "pfeiffer_big_sur_state_park" in browser["rendered_map"]["rendered_marker_ids"]
            and "bixby_creek_bridge_viewpoint" not in browser["rendered_map"]["rendered_marker_ids"],
            "existing_discovery_identity_invalid",
        )
        return {
            "scope": "Original-context real route discovery, exact candidate selection, staged replacement, evaluated rendered child",
            "parent_plan_id": trace["parent_plan_id"],
            "child_plan_id": trace["child_plan_id"],
            "map_hash": trace["geometry_hash"],
        }

    def decision_control(read, context, kind):
        from verify_e2e1_control_actions import decision_controls

        base = "addition_continuation/"
        rows = read(base + "decision_control_journeys.json")
        events = read(base + "decision_control_events.json")
        records = [read(base + row["file"]) for row in read(base + "http_index.json")]
        verified = decision_controls(rows, events, records, read(base + "browser_assertions.json"))
        selected = [row for row in verified if row["binding_id"] == kind and row["context"] == context]
        require(len(selected) == 1, "decision_control_context_unbound")
        return selected[0]

    for context in ("original", "accepted_child"):
        project(
            "addition-continuation",
            context,
            {"accept_repair": "#accept-repair"},
            lambda read, c=context: decision_control(read, c, "accept_repair"),
        )
        project(
            "addition-continuation",
            context,
            {"keep_original": "#keep-original"},
            lambda read, c=context: decision_control(read, c, "keep_original"),
        )

    def retry(read, context):
        base = "map_initialization_recovery/" + context + "/"
        trace, blocked, recovered, requests = (
            read(base + name)
            for name in (
                "trace.json",
                "persistent_failure_no_ready_claim_diagnostics.json",
                "recovered_after_explicit_retry_diagnostics.json",
                "browser_requests.json",
            )
        )
        stage = recovered["map_stage"]
        require(
            trace["schema_version"] == "map-config-recovery-browser-v2"
            and trace["context"] == context
            and trace["assertions_passed"] is True
            and blocked["retry_visible"] is True
            and blocked["map_stage"].get("renderedMapPlanId") is None
            and blocked["geographic_map"].get("mapReady") != "true"
            and stage["renderedMapPlanId"] == stage["activePlanId"] == trace["active_plan_id"]
            and stage["activePlanContentHash"] == trace["active_plan_content_hash"]
            and stage["renderedGeometryHash"] == trace["rendered_geometry_hash"]
            and not any(
                row["method"] != "GET" for row in requests if row["phase"] == "explicit_retry_after_fault_removed"
            ),
            "existing_retry_identity_invalid",
        )
        if context == "accepted_child":
            acceptance = read(base + "accept_response.json")
            require(
                trace["active_plan_id"] == trace["child_plan_id"] == acceptance["session"]["accepted_plan_id"],
                "retry_accepted_parent_unbound",
            )
        else:
            require(
                trace["active_plan_id"] == trace["parent_plan_id"] and trace["child_plan_id"] is None,
                "retry_original_parent_unbound",
            )
        return {
            "scope": "Explicit Retry after disclosed map-config fault is removed; real "
            + context
            + " route restored with no mutation",
            "accepted_plan_id": trace["active_plan_id"],
            "map_hash": trace["rendered_geometry_hash"],
        }

    def reload_build(read, context):
        from verify_e2e1_self_managed import stamp

        base = "build_coherence/" + context + "/"
        assertions, maps, index, blocked = (
            read(base + name)
            for name in (
                "browser_assertions.json",
                "map_checkpoints.json",
                "http_index.json",
                "stale_write_state_hashes.json",
            )
        )
        initial = "accepted_child" if context == "accepted_child" else "initial_parent"
        require(
            assertions["schema_version"] == "build-coherence-browser-v1"
            and assertions["context"] == context
            and assertions["assertions_passed"] is True
            and assertions["explicit_reload_session_preserved"] is True
            and assertions["blocked_mutation_state_unchanged"] is True
            and assertions["first_build_id"] != assertions["second_build_id"]
            and blocked["before"] == blocked["after"]
            and maps[initial] == maps["active_plan_after_update"]
            and maps["active_plan_after_update"]["active_plan_id"] == assertions["accepted_plan_id"],
            "existing_reload_identity_invalid",
        )
        clicked = read(base + "reload_control_input.json")
        recovery = read(base + "product_client_build_mismatch_visible_recovery.json")
        ready = read(base + "generation_b/active_plan_after_update.readiness.json")
        require(
            clicked["trusted"] is True
            and clicked["selector"] == "#reload-current-build"
            and clicked["client_build"] == assertions["first_build_id"]
            and clicked["path"] == "/app/itinerary"
            and clicked["title"] == recovery["text"]
            and stamp(recovery["captured_at"]) <= stamp(clicked["clicked_at"]) <= stamp(ready["screenshot_started_at"]),
            "existing_reload_input_unbound",
        )
        restored = [
            read(base + item["file"])
            for item in index
            if item["phase"] == "generation_b_reloaded"
            and item["method"] == "GET"
            and item["url"].endswith("/api/sessions/" + assertions["session"]["session_id"])
        ]
        require(
            len(restored) == 1
            and restored[0]["response_status_code"] == 200
            and json.loads(restored[0]["response_body_utf8"])["session"]["accepted_plan_id"]
            == assertions["accepted_plan_id"],
            "existing_reload_http_unbound",
        )
        return {
            "scope": "Actual two-generation stale-build rejection and explicit reload preserving the "
            + context
            + " plan",
            "accepted_plan_id": assertions["accepted_plan_id"],
            "build_ids": [assertions["first_build_id"], assertions["second_build_id"]],
        }

    project(
        "vertical-slice",
        "original",
        {
            "discover_replacements": "#prepare-replacement-candidates",
            "discover_route_candidates": "#find-route-candidates",
            "inspect_candidate": "[data-poi-candidate]",
            "use_replacement": "#use-candidate-replacement",
        },
        discovery,
    )
    for context in ("original", "accepted_child"):
        project(
            "map-recovery", context, {"retry_map_runtime": "#retry-map-runtime"}, lambda read, c=context: retry(read, c)
        )
    for context in ("original", "accepted_child"):
        project(
            "build-coherence",
            context,
            {"reload_build": "#reload-current-build"},
            lambda read, c=context: reload_build(read, c),
        )

    def cancel(read, context):
        base = f"waypoint_contexts/{context}/"
        value = read(base + "exploratory_cancel.json")
        parent = read(base + "browser_assertions.json")["parent"]
        from verify_e2e1_waypoint_contexts import exploratory_cancel, exploratory_cancel_raw

        exploratory_cancel(value, context, parent)
        exploratory_cancel_raw(value, read(base + "browser_requests.json"), read(base + "exploratory_input_log.json"))
        return {
            "scope": "Actual exploratory canvas click and Cancel; no confirmed waypoint, draft or accepted-plan mutation",
            "accepted_plan_id": parent["active_plan_id"],
            "accepted_plan_content_hash": parent["active_plan_content_hash"],
        }

    for context in ("original", "accepted_child"):
        project(
            "waypoint-contexts",
            context,
            {"cancel_exploratory": "#cancel-exploratory"},
            lambda read, c=context: cancel(read, c),
        )

    def waypoint_controls(read, context):
        from verify_e2e1_waypoint_contexts import control_proofs

        base = f"waypoint_contexts/{context}/"
        records = [read(base + item["file"]) for item in read(base + "http_index.json")]
        return control_proofs(
            read(base + "waypoint_control_receipts.json"),
            read(base + "waypoint_control_input_log.json"),
            read(base + "browser_requests.json"),
            records,
            context,
            read(base + "browser_assertions.json")["parent"],
        )

    from verify_e2e1_waypoint_contexts import CONTROL_SELECTORS

    for context in ("original", "accepted_child"):
        project("waypoint-contexts", context, CONTROL_SELECTORS, lambda read, c=context: waypoint_controls(read, c))

    def conversation_context(read, base, context, identity):
        trace = read(base + "trace.json")
        require(
            trace["context"] == context and trace["initial"] == trace["final"] == identity,
            "conversation_control_context_invalid",
        )
        if context == "accepted_child":
            baseline = read(base + "accepted_baseline_artifacts.json")
            accepted = json.loads(read(base + "accepted_setup_accept_http.json")["response_body_utf8"])
            restored = json.loads(read(base + "accepted_setup_restored_http.json")["response_body_utf8"])
            decision = accepted["result"]["decision"]
            require(
                baseline["context"] == context
                and baseline["assertions_passed"] is True
                and baseline["accepted_map"] == identity
                and baseline["original"]["active_plan_id"] != identity["active_plan_id"]
                and decision["selected_plan_id"]
                == restored["session"]["accepted_plan_id"]
                == identity["active_plan_id"]
                and decision["selected_plan_content_hash"] == identity["active_plan_content_hash"]
                and restored["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"],
                "accepted_conversation_control_unbound",
            )

    def deletion(read, context):
        name = "preference_memory" + ("_accepted_child" if context == "accepted_child" else "")
        base = "conversation/" + name + "/"
        value = read(base + "control_closure.json")
        raw = read(base + "control_clicks.json")
        before, after = (read(base + "context_" + side + "_delete_all.json") for side in ("before", "after"))
        conversation_context(read, base, context, value["accepted_map"])
        require(
            value["schema_version"] == "conversation-control-closure-v1"
            and value["scenario"] == name
            and value["assertions_passed"] is True
            and value["raw_clicks"] == raw[: len(value["raw_clicks"])]
            and any(row["trusted"] is True and row["id"] == "delete-all-conversations" for row in value["raw_clicks"])
            and value["new_chat_request_count"] == 0
            and value["streams_before"] == value["streams_after"]
            and value["visible_turn_count"] == 0
            and value["before"] == before
            and value["after"] == after
            and after["draft_operations"] == []
            and after["conversation_memory"]["proposed_preferences"] == []
            and before["accepted_plan_id"] == after["accepted_plan_id"] == value["accepted_map"]["active_plan_id"]
            and before["accepted_plan_content_hash"]
            == after["accepted_plan_content_hash"]
            == value["accepted_map"]["active_plan_content_hash"],
            "delete_all_control_unbound",
        )
        index = read(base + "http_index.json")
        records = [
            read(base + row["file"])
            for row in index
            if row["method"] == "DELETE" and row["url"].endswith("/api/conversations")
        ]
        require(
            len(records) == 1
            and records[0]["response_status_code"] == 200
            and json.loads(records[0]["request_body_utf8"]) == value["delete_all_request"]
            and value["delete_all_request"]["confirmation"] == "delete_all_conversations"
            and json.loads(records[0]["response_body_utf8"]) == value["delete_all_response"]
            and type(value["delete_all_response"]["deleted_count"]) is int
            and value["delete_all_response"]["deleted_count"] > 0,
            "delete_all_http_unbound",
        )
        return {
            "scope": "Confirmed deletion of all conversations in disposable repository; "
            + context
            + " accepted itinerary unchanged"
        }

    for context in ("original", "accepted_child"):
        project(
            "conversation",
            context,
            {"transcript_delete_all": "#delete-all-conversations"},
            lambda read, c=context: deletion(read, c),
        )

    def submit_and_delete_current(read, context):
        from verify_e2e1_self_managed import _stream

        base = "conversation/preference_memory" + ("_accepted_child" if context == "accepted_child" else "") + "/"
        trace = read(base + "trace.json")
        parent = trace["initial"]
        conversation_context(read, base, context, parent)
        clicks = read(base + "control_clicks.json")
        sends = [row for row in clicks if row["id"] == "copilot-submit"]
        deletes = [row for row in clicks if row["id"] == "delete-current-conversation"]
        turns = read(base + "exact_chat_io.json")
        records = [read(base + row["file"]) for row in read(base + "http_index.json")]
        chats = [row for row in records if row["method"] == "POST" and row["url"].endswith("/copilot/messages")]
        current_deletes = [row for row in records if row["method"] == "DELETE" and row["url"].endswith("/conversation")]
        require(
            trace["assertions_passed"] is True
            and len(sends) == len(turns) == len(chats) == trace["chat_turn_count"] == 11
            and all(row["trusted"] is True and row["aria_label"] == "Send message" for row in sends)
            and len(deletes) == len(current_deletes) == 1
            and deletes[0]["trusted"] is True
            and sends[9]["at"] < deletes[0]["at"] < sends[10]["at"],
            "conversation_submit_delete_input_unbound",
        )
        for recorded, http in zip(turns, chats, strict=True):
            request, events = _stream(http)
            require(
                http["response_status_code"] == 200
                and http["response_capture_complete"] is True
                and request == recorded["input"]
                and events == recorded["events"]
                and events[-1] == recorded["output"]
                and events[-1]["type"] == "turn.completed",
                "conversation_submit_http_unbound",
            )
            payload = events[-1]["payload"]
            turn, session = payload["turn"], payload["session"]
            require(
                turn["user_message"] == request["message"]
                and turn["client_message_id"] == request["client_message_id"]
                and turn["context_revision"] == request["expected_revision"]
                and session["accepted_plan_id"] == parent["active_plan_id"]
                and session["draft"] == []
                and session["proposal"] is None
                and http["url"].endswith("/sessions/" + session["session_id"] + "/copilot/messages"),
                "conversation_submit_result_unbound",
            )
        prior = turns[9]["output"]["payload"]["session"]
        fresh = turns[10]["output"]["payload"]["session"]
        http = current_deletes[0]
        request = json.loads(http["request_body_utf8"])
        result = json.loads(http["response_body_utf8"])
        before, after = (read(base + "context_" + name + ".json") for name in ("corrected", "reset"))
        require(
            http["response_status_code"] == 200
            and http["response_capture_complete"] is True
            and result["deleted"] is True
            and http["url"].endswith("/sessions/" + prior["session_id"] + "/conversation")
            and records.index(chats[9]) < records.index(http) < records.index(chats[10])
            and request == {"expected_revision": prior["revision"]}
            and result["session"] == {**prior, "conversation_id": None}
            and fresh["conversation_id"]
            and prior["conversation_id"]
            and fresh["conversation_id"] != prior["conversation_id"]
            and fresh["session_id"] == prior["session_id"]
            and before["accepted_plan_id"] == after["accepted_plan_id"] == parent["active_plan_id"]
            and before["accepted_plan_content_hash"]
            == after["accepted_plan_content_hash"]
            == parent["active_plan_content_hash"]
            and before["conversation_memory"]["proposed_preferences"]
            and after["conversation_memory"]["proposed_preferences"] == []
            and after["conversation_memory"]["pending_clarification"] is None
            and after["draft_operations"] == [],
            "conversation_delete_current_result_unbound",
        )
        return {
            "scope": "Actual native Send clicks with exact SSE turns, followed by confirmed current-transcript deletion and empty-memory new chat",
            "accepted_plan_id": parent["active_plan_id"],
            "accepted_plan_content_hash": parent["active_plan_content_hash"],
            "old_conversation_id": prior["conversation_id"],
            "new_conversation_id": fresh["conversation_id"],
        }

    for context in ("original", "accepted_child"):
        project(
            "conversation",
            context,
            {"chat_submit": "#copilot-form", "transcript_delete_current": "#delete-current-conversation"},
            lambda read, c=context: submit_and_delete_current(read, c),
        )

    def candidate_add(read):
        from verify_e2e1_candidate_day_matrix import legacy_add_control_receipt

        base = "candidate_day_matrix/"
        cases = read(base + "candidate_day_cases.json")
        selected = [
            row
            for row in cases
            if row["context"] == "accepted_child"
            and row["scope"] == "real_staging_and_impact_only"
            and row["readiness"]["preview_executable"] is True
        ]
        require(len(selected) == 1, "candidate_add_control_case_invalid")
        index = read(base + "http_index.json")
        records = [read(base + row["file"]) for row in index]
        return legacy_add_control_receipt(selected[0], records)

    project("candidate-day-matrix", "accepted_child", {"legacy_add": "#draft-add"}, candidate_add)

    def generated_chat(read, name):
        base = "conversation/" + name + "/"
        value, raw, trace = (
            read(base + filename) for filename in ("control_closure.json", "control_clicks.json", "trace.json")
        )
        context = "accepted_child" if name.endswith("_accepted_child") else "original"
        recovery = name.removesuffix("_accepted_child") == "fake_sdk_cancel"
        conversation_context(read, base, context, value["accepted_map"])
        group = "Message recovery actions" if recovery else "Conversation capacity actions"
        require(
            value["assertions_passed"] is True
            and value["scenario"] == name
            and value["raw_clicks"] == raw[: len(value["raw_clicks"])]
            and value["new_chat_request_count"] == 0
            and value["streams_before"] == value["streams_after"]
            and trace["initial"] == trace["final"] == value["accepted_map"],
            "generated_chat_state_changed",
        )

        def clicked(label):
            return any(row["trusted"] is True and row["label"] == label and row["action_group"] == group for row in raw)

        before = read(base + "dispatch_after_cancel.json") if recovery else None
        expected_edit = before["user_message"] if recovery else "Review the safer repair."
        require(
            clicked("Edit message") and value["edit_message_composer"] == expected_edit, "generated_edit_result_missing"
        )
        if recovery:
            after = read(base + "dispatch_after_retry.json")
            require(
                clicked("Retry")
                and before == after
                and before["attempt_count"] == 1
                and before["outcome"] == "unknown"
                and value["fake_attempts_before"] == value["fake_attempts_after"] == 1
                and trace["live_provider_call_count"] == 0,
                "generated_retry_redispatched",
            )
        else:
            require(
                clicked("Conversation settings")
                and value["settings_opened"] is True
                and value["settings_closed"] is True,
                "generated_capacity_settings_missing",
            )
        return {
            "scope": "Exact "
            + ("recovery Edit and no-redispatch Retry" if recovery else "capacity Edit and Conversation settings")
            + " controls; no other generated proposal action credited"
        }

    for context in ("original", "accepted_child"):
        suffix = "_accepted_child" if context == "accepted_child" else ""
        project(
            "conversation",
            context,
            {
                "chat_generated_action": {
                    "selector": ".copilot-recovery-actions button",
                    "variants": ["recovery_retry", "recovery_edit"],
                }
            },
            lambda read, s=suffix: generated_chat(read, "fake_sdk_cancel" + s),
        )
        for case in ("capacity_turns", "capacity_bytes"):
            project(
                "conversation",
                context,
                {
                    "chat_generated_action": {
                        "selector": '.copilot-recovery-actions[aria-label="Conversation capacity actions"] button',
                        "variants": ["capacity_settings", "capacity_edit"],
                    }
                },
                lambda read, name=case + suffix: generated_chat(read, name),
            )
        for one_click in (False, True):
            name = "combined_chat" + ("_one_click" if one_click else "") + suffix
            variants = (
                ["proposal_combined_preview_empty", "proposal_combined_added_disabled"]
                if one_click
                else ["proposal_combined_add", "proposal_combined_preview_staged", "proposal_combined_added_disabled"]
            )

            def combined_proposal(read, case=name):
                from verify_e2e1_self_managed import combined_chat_control_receipt

                return combined_chat_control_receipt(
                    lambda filename: read("conversation/" + case + "/" + filename), name=case
                )

            project(
                "conversation",
                context,
                {"chat_generated_action": {"selector": ".copilot-proposal-actions > button", "variants": variants}},
                combined_proposal,
            )

    def aliases(read, context, position):
        from verify_e2e1_evaluated_variants import VARIANTS, control_aliases

        kind, variant, _ = VARIANTS[position]
        base = f"evaluated_variants/{context}/{kind}_{variant}/"
        value, timing, payload, staged, undo, trace = (
            read(base + name)
            for name in (
                "action_aliases.json",
                "preview_timing.json",
                "preview_response.json",
                "stage_response.json",
                "undo_response.json",
                "trace.json",
            )
        )
        control_aliases(value, timing, context, position, payload["proposal"], staged["operation"])
        require(
            undo["undone"] == staged["operation"]
            and undo["session"]["draft"] == []
            and undo["session"]["accepted_plan_id"] == trace["parent_plan_id"]
            and trace["assertions_passed"] is True,
            "alias_undo_not_canonical",
        )
        return {
            "scope": "Existing evaluated variant actual Preview/Undo alias, canonical request/result and map checks",
            "request_id": value["request_id"],
            "operation_id": value["operation_id"],
            "eligibility": payload["proposal"]["state"],
        }

    for context in ("original", "accepted_child"):
        for position, prefix in enumerate(("inline", "editor", "context")):
            control_prefix = "draft" if prefix == "editor" else prefix
            project(
                "evaluated-variants",
                context,
                {
                    prefix + "_preview": "#" + control_prefix + "-preview",
                    prefix + "_undo": "#" + control_prefix + "-undo",
                },
                lambda read, c=context, p=position: aliases(read, c, p),
            )

    def poi_controls(read):
        from verify_e2e1_poi_controls import verify_rows

        base = "poi_discovery/"
        return verify_rows(
            read(base + "poi_control_journeys.json"),
            read(base + "poi_control_input_log.json"),
            read(base + "browser_assertions.json")["parent_map"],
            read(base + "browser_poi_candidate_response.json")["candidates"],
        )

    project(
        "vertical-slice",
        "original",
        {
            "filter_candidates": "#poi-candidate-filter",
            "close_candidates": "#close-poi-candidates",
            "candidate_layer": "#poi-layer-toggle",
            "layers": {"selector": "#layers-button", "variants": ["candidate_layer_toggle"]},
        },
        poi_controls,
    )

    def accepted_poi_controls(read):
        from verify_e2e1_poi_controls import accepted_rows

        base = "candidate_day_matrix/"
        records = [read(base + row["file"]) for row in read(base + "http_index.json")]
        return accepted_rows(
            read(base + "accepted_poi_control_journeys.json"),
            read(base + "accepted_poi_control_input_log.json"),
            read(base + "capability_snapshots.json")["accepted_child"],
            records,
        )

    project(
        "candidate-day-matrix",
        "accepted_child",
        {
            "filter_candidates": "#poi-candidate-filter",
            "close_candidates": "#close-poi-candidates",
            "candidate_layer": "#poi-layer-toggle",
            "layers": {"selector": "#layers-button", "variants": ["candidate_layer_toggle"]},
            "discover_replacements": "#prepare-replacement-candidates",
            "discover_route_candidates": "#find-route-candidates",
        },
        accepted_poi_controls,
    )

    def inspected_accepted(read):
        from verify_e2e1_poi_controls import accepted_inspection

        base = "candidate_day_matrix/"
        return accepted_inspection(
            read(base + "accepted_candidate_inspection.json"),
            read(base + "capability_snapshots.json")["accepted_child"],
            [read(base + row["file"]) for row in read(base + "http_index.json")],
        )

    project(
        "candidate-day-matrix",
        "accepted_child",
        {
            "inspect_candidate": "[data-poi-candidate]",
            "use_replacement": {
                "selector": "#use-candidate-replacement",
                "variants": ["default"],
                "status": "disabled_by_exact_server_guard",
            },
        },
        inspected_accepted,
    )

    def guarded_day(read, context):
        from verify_e2e1_control_actions import disabled_day

        base = "capability_controls/" + context + "/"
        snapshot = read(base + "capability_snapshot.json")
        return disabled_day(
            read(base + "disabled_day_control.json"),
            snapshot["capabilities"]["typed_edit_capabilities"]["operations"]["set_stop_day"],
            [read(base + row["file"]) for row in read(base + "http_index.json")],
            context,
            read(base + "browser_assertions.json")["parent"],
        )

    for context in ("original", "accepted_child"):
        project(
            "capability-controls",
            context,
            {
                "typed_combination": '.typed-operation-form[data-typed-operation="set_stop_day"]',
                "typed_submit": {
                    "selector": '.typed-operation-form[data-typed-operation="set_stop_day"]',
                    "variants": ["default"],
                    "status": "disabled_by_exact_server_guard",
                },
            },
            lambda read, c=context: guarded_day(read, c),
        )

    def generated_proposals(read, context):
        from verify_e2e1_chat_closure import generated_controls

        base = "chat_closure/" + context + "/"
        return {
            "scope": "Real generated proposal controls, with disabled states kept distinct from executed actions",
            "actions": generated_controls(
                read(base + "generated_control_actions.json"),
                read(base + "generated_control_observer.json"),
                read(base + "exact_chat_io.json"),
                [read(base + row["file"]) for row in read(base + "http_index.json")],
                read(base + "journey_assertions.json"),
                context,
                read(base + "browser_assertions.json")["parent"],
            ),
        }

    for context in ("original", "accepted_child"):
        for selector, variants in (
            (".copilot-proposal-actions > button", ["proposal_show_map", "proposal_compare", "proposal_evidence"]),
            (
                ".copilot-proposal-action-row button",
                ["proposal_add", "proposal_preview", "proposal_added_disabled", "proposal_cannot_add_disabled"],
            ),
        ):
            project(
                "chat-closure",
                context,
                {"chat_generated_action": {"selector": selector, "variants": variants}},
                lambda read, c=context: generated_proposals(read, c),
            )

    def responsive(read, context):
        base = f"capability_controls/{context}/"
        rows = read(base + "compare_responsive_journeys.json")
        parent = read(base + "browser_assertions.json")["parent"]
        index = read(base + "http_index.json")
        records = [read(base + row["file"]) for row in index]
        workspaces = [
            json.loads(row["response_body_utf8"])["workspace"]
            for row in records
            if "workspace" in json.loads(row["response_body_utf8"])
        ]
        plans = {plan["plan_id"]: plan for workspace in workspaces for plan in workspace["geography"]["plans"]}
        verify_compare_responsive(rows, context, parent, list(plans.values()))
        for name in rows[0]["http_files"]:
            require(read(base + name)["method"] == "GET", "responsive_api_mutation")
        return {
            "scope": "Actual active Compare 1440 to 390 to 1440; same canonical map identities, visible panes and readiness"
        }

    for context in ("original", "accepted_child"):
        project(
            "capability-controls",
            context,
            {"compare_responsive": "[data-compare-mobile-toggle]"},
            lambda read, c=context: responsive(read, c),
        )

    def bootstrap(read, case, context):
        from verify_e2e1_bootstrap_recovery import blocked_result, retry_observer

        base = "bootstrap_recovery/" + case + "/" + context + "/"
        blocked = read(base + "blocked_dom.json")
        result = read(base + "browser_assertions.json")
        blocked_result(blocked, blocked["fault_health"], case, context)
        retry_observer(
            read(base + "retry_observer.json"),
            blocked,
            result,
            read(base + "recovered/provider_isolation.json"),
            read(base + "recovered/recovered_parent.readiness.json"),
        )
        require(
            result["assertions_passed"] is True
            and result["case"] == case
            and result["context"] == context
            and result["retry"]["trusted"] is True
            and result["retry"]["id"] == "retry-runtime"
            and result["retry"]["title"] == blocked["title"]
            and result["parent"]["preview_state"] == "PARENT_RENDERED"
            and result["blocked_screenshot_taken"] is False
            and result["frontend_fetch_mocked"] is False,
            "bootstrap_control_not_bound",
        )
        if context == "accepted_child":
            transition = read("bootstrap_recovery/" + case + "/original/accepted_transition.json")
            require(
                result["accepted_transition"] == transition
                and result["parent"] == transition["accepted_map"]
                and result["session"] == blocked["session"] == transition["session"],
                "accepted_bootstrap_not_bound",
            )
        return {
            "scope": "Actual "
            + case
            + " bootstrap rejection; real Retry renders the same canonical "
            + context
            + " plan without creating a replacement session.",
            "parent_plan_id": result["parent"]["active_plan_id"],
        }

    for case, binding in (("core_not_ready", "retry_runtime_unavailable"), ("invalid_health", "retry_runtime_failed")):
        for context in ("original", "accepted_child"):
            project(
                "bootstrap-recovery",
                context,
                {binding: "#retry-runtime"},
                lambda read, c=case, p=context: bootstrap(read, c, p),
            )

    def optional_evidence_failure(read):
        from verify_e2e1_optional_evidence_fault import semantic_contract

        base = "optional_evidence_fault/"
        return semantic_contract(
            read(base + "browser_assertions.json"),
            read(base + "control_inputs.json"),
            [read(base + row["file"]) for row in read(base + "http_index.json")],
            read(base + "fault_patch.json"),
        )

    project(
        "optional-evidence-fault",
        "accepted_child",
        {
            "evidence_back_unavailable": {
                "selector": "#evidence-back",
                "variants": ["default"],
                "coverage_category": "declared_backend_projection_failure_path",
                "natural_interaction_claimed": False,
            }
        },
        optional_evidence_failure,
    )
    return proofs, unavailable


def control_gate(receipts, registry, inventory):
    mappings = registry.get("controls")
    require(
        isinstance(mappings, list)
        and mappings
        and all(isinstance(row, dict) and all(row.get(key) for key in CONTROL_FIELDS) for row in mappings),
        "control_mapping_incomplete",
    )
    require(len({row["action_id"] for row in mappings}) == len(mappings), "control_mapping_duplicate")
    source, unmapped, stale = binding_inventory(registry, inventory)
    source = source + registry.get("native_controls", [])
    missing, executed, disabled, observed_only = [], [], [], []
    external, external_unavailable = existing_control_proofs(receipts)
    for context in registry["required_contexts"]:
        prefix = f"capability_controls/{context}/"
        value = receipts.read("capability-controls", prefix + "general_control_inventory.json")
        journeys = receipts.read("capability-controls", prefix + "general_control_journeys.json")
        raw = receipts.read("capability-controls", prefix + "general_control_events.json")
        snapshot = receipts.read("capability-controls", prefix + "capability_snapshot.json")
        parent = receipts.read("capability-controls", prefix + "browser_assertions.json")["parent"]
        require(
            value["schema_version"] == "general-control-inventory-v1"
            and value["context"] == context
            and value["presence_is_not_execution"] is True
            and value["absent_is_not_unsupported"] is True
            and journeys["schema_version"] == "general-control-journeys-v1"
            and journeys["context"] == context
            and journeys["inventory_is_not_execution"] is True
            and raw["schema_version"] == "general-control-events-v1"
            and raw["context"] == context
            and raw["trusted_event_is_not_semantic_completion"] is True,
            "general_inventory_scope_invalid",
        )
        observations = value["observations"]
        indexed = {row["binding_id"]: row for row in source}
        for item in observations:
            require(
                item["binding_id"] in indexed
                and item["context"] == context
                and item["selector"] in indexed[item["binding_id"]]["selectors"]
                and item["observed_count"] == len(item["controls"]),
                "general_inventory_mapping_invalid",
            )
        for journey in journeys["journeys"]:
            require(journey["binding_id"] in indexed, "general_journey_unregistered")
            verify_general_journey(journey, indexed[journey["binding_id"]], observations, raw["events"], parent)
            selection_records = []
            for name in journey["http_files"]:
                http = receipts.read("capability-controls", prefix + name)
                selection_records.append(http)
                require(
                    http["phase"] == journey["phase"]
                    and http["response_status_code"] < 400
                    and not http["url"].endswith(
                        ("/draft/operations", "/draft/undo", "/preview", "/accept", "/copilot/messages")
                    ),
                    "general_navigation_mutated_or_http_unbound",
                )
                payload = json.loads(http["response_body_utf8"])
                if isinstance(payload, dict) and "session" in payload:
                    require(
                        payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
                        and payload["session"]["draft"] == [],
                        "general_http_parent_or_draft_changed",
                    )
            verify_selection_journey(journey, selection_records, parent)
        for row in source:
            descriptor = {"binding_id": row["binding_id"], "context": context, "selectors": row["selectors"]}
            matches = [
                item
                for item in observations
                if item["binding_id"] == row["binding_id"]
                and any(control["visible"] is True for control in item["controls"])
            ]
            completed = [item for item in journeys["journeys"] if item["binding_id"] == row["binding_id"]]
            proof = external.get((row["binding_id"], context))
            proof_rows = proof.get("receipts", []) if proof else []
            completed_selectors = {item["selector"] for item in completed} | {item["selector"] for item in proof_rows}
            completed_variants = {item.get("variant", "default") for item in completed} | {
                variant for item in proof_rows for variant in item["variants"]
            }
            operation = row.get("operation")
            cap = snapshot["capabilities"]["draft_capabilities"]["operations"].get(operation, {})
            if (
                matches
                and operation
                and cap.get("enabled") is False
                and all(
                    control["disabled"] is True
                    for item in matches
                    for control in item["controls"]
                    if control["visible"] is True
                )
            ):
                disabled.append(
                    {**descriptor, "status": "disabled_by_current_capability", "unsupported_claimed": False}
                )
            elif completed_selectors >= set(
                row.get("required_execution_selectors", row["selectors"])
            ) and completed_variants >= set(row.get("required_journey_variants", ["default"])):
                if (
                    not completed
                    and proof_rows
                    and all(item.get("status") == "disabled_by_exact_server_guard" for item in proof_rows)
                ):
                    disabled.append(
                        {
                            **descriptor,
                            "status": "disabled_by_exact_server_guard",
                            "unsupported_claimed": False,
                            "external_proof": proof,
                        }
                    )
                else:
                    executed.append(
                        {
                            **descriptor,
                            "journeys": [item["phase"] for item in completed],
                            "external_proof": proof,
                            "independently_disabled_selectors": [
                                item["selector"]
                                for item in proof_rows
                                if item.get("status") == "disabled_by_exact_server_guard"
                            ],
                        }
                    )
            else:
                finding = {
                    **descriptor,
                    "reason": "observed_but_semantic_journey_missing"
                    if matches
                    else "conditional_control_not_observed",
                    "missing_executed_selectors": sorted(
                        set(row.get("required_execution_selectors", row["selectors"])) - completed_selectors
                    ),
                    "missing_executed_variants": sorted(
                        set(row.get("required_journey_variants", ["default"])) - completed_variants
                    ),
                    "availability": row.get("availability", "Native product control"),
                }
                missing.append(finding)
                if matches:
                    observed_only.append(descriptor)
    return {
        "passed": not unmapped and not stale and not missing,
        "unmapped_source_bindings": unmapped,
        "stale_source_bindings": stale,
        "missing_control_receipts": missing,
        "executed_controls": executed,
        "disabled_controls": disabled,
        "observed_only_controls": observed_only,
        "existing_control_proofs_unavailable": external_unavailable,
        "scope": "Exact source bindings; separate visible/disabled DOM evidence and independently checked trusted-action results. Conditional absence never means unsupported.",
    }


def geography_fault_gate(receipts):
    from verify_e2e1_preview_geography_fault import semantic_contract
    from verify_e2e1_vertical_slice import E2E1VerificationError

    cases = []
    for kind in ("missing", "foreign"):
        base = "preview_geography_fault/" + kind + "/"
        value = receipts.read("preview-geography-fault", base + "browser_assertions.json")
        boundary = receipts.read("preview-geography-fault", base + "backend_boundary.json")
        persisted = receipts.read("preview-geography-fault", base + "persisted_session.json")
        index = receipts.read("preview-geography-fault", base + "http_index.json")
        records = [receipts.read("preview-geography-fault", base + row["file"]) for row in index]
        require(value["kind"] == kind, "geometry_fault_case_mismatch")
        try:
            cases.append(semantic_contract(value, boundary, records, persisted))
        except E2E1VerificationError as error:
            raise ReceiptError(str(error)) from error
    return {
        "cases": cases,
        "scope": "Real eligible child rejected under declared missing/foreign response geography; no live provider claim.",
    }


def gate_report(directory, suites, registry, variants, *, root=ROOT):
    receipts = Receipts(directory, suites)
    findings, gates = [], []
    source_error = False
    try:
        inventory = source_controls(root) if CONTROL_GATE in registry["unverified_requirements"] else []
    except (OSError, ValueError, TypeError):
        inventory, source_error = [], True
    functions = {
        CONVERSATION_GATE: lambda: conversation_gate(receipts),
        NEGATIVE_GATE: lambda: negative_gate(receipts, variants),
        CONTROL_GATE: lambda: control_gate(receipts, registry, inventory),
        GEOGRAPHY_GATE: lambda: geography_fault_gate(receipts),
    }
    for code in registry["unverified_requirements"]:
        start = len(receipts.used)
        try:
            require(code in functions, "runtime_requirement_has_no_verifier")
            require(code != CONTROL_GATE or not source_error, "control_source_inventory_unavailable")
            value = functions[code]()
            passed = value.get("passed", True) is True
            gates.append({"requirement": code, **value, "passed": passed, "evidence": receipts.used[start:]})
            if not passed:
                findings.append({"code": code})
        except (ReceiptError, OSError, KeyError, TypeError, ValueError, AttributeError) as error:
            detail = str(error) if isinstance(error, ReceiptError) else "runtime_requirement_evidence_malformed"
            gates.append({"requirement": code, "passed": False, "reason": detail, "evidence": receipts.used[start:]})
            findings.append({"code": code, "reason": detail})
    external = [
        {
            "requirement": LIVE_GATE,
            "required_for_full_phase": True,
            "required_for_deterministic_aggregate": False,
            "status": "not_evaluated_by_deterministic_aggregate",
            "verified": False,
            "verifier": "scripts/verify_e2e1_live_copilot_chat.py",
            "evidence_directory": None,
            "scope": "A separately authorized real website chat must use the live provider and independently evaluated exact rendered child.",
        }
    ]
    return {
        "findings": findings,
        "requirements": gates,
        "source_control_inventory": inventory,
        "external_gates": external,
        "full_phase_completion_claimed": False,
    }
