"""Independent exact native draft input -> real request -> canonical operation binding."""

from __future__ import annotations

import hashlib
import json

import verify_e2e1_vertical_slice as core

LEGACY = {"keep_stop": "legacy_keep", "replace_nearby": "legacy_replace", "add_candidate": "legacy_add"}
IDS = {"keep_stop": "draft-keep", "replace_nearby": "draft-replace", "add_candidate": "draft-add"}


def decision_controls(rows, events, records, assertions):
    """Existing A->B->C acceptance and explicit C->A->A Keep-original controls."""
    expected = [
        ("accept_repair", "original", "b"),
        ("accept_repair", "accepted_child", "c"),
        ("keep_original", "accepted_child", "a"),
        ("keep_original", "original", "a"),
    ]
    require(len(rows) == 4 and len(events) == 4, "decision_control_universe_invalid")
    require(
        [event for row in rows for event in row["events"]] == events and len({event["at"] for event in events}) == 4,
        "decision_event_universe_invalid",
    )
    http = [
        record
        for record in records
        if record["method"] == "POST" and record["url"].endswith(("/accept", "/keep-original"))
    ]
    require(len(http) == 4, "decision_http_universe_invalid")
    previous = assertions["a"]["active_plan_id"]
    for row, record, (kind, context, label) in zip(rows, http, expected, strict=True):
        request, payload = row["request"], row["response"]
        before, after, restored = row["session_before"], payload["session"], row["restored"]
        decision, pointer = payload["result"]["decision"], payload["result"]["pointer"]
        accepted = kind == "accept_repair"
        selector = "#accept-repair" if accepted else "#keep-original"
        endpoint = "/accept" if accepted else "/keep-original"
        confirmation = "accept_evaluated_repair" if accepted else "keep_original_plan"
        message = (
            "Accept this independently evaluated repair as the durable trip plan?"
            if accepted
            else "Keep the immutable original plan as the durable trip plan?"
        )
        require(
            row["binding_id"] == kind
            and row["context"] == context
            and row["selector"] == selector
            and row["assertions_passed"] is True
            and row["confirmation_messages"] == [message],
            "decision_input_scope_invalid",
        )
        require(len(row["events"]) == 1 and row["events"][0] in events, "decision_raw_event_unbound")
        event = row["events"][0]
        require(
            event["trusted"] is True
            and event["type"] == "click"
            and event["selector"] == selector
            and event["path"] == "/app/itinerary"
            and event["visible"] is True
            and event["disabled"] is False
            and isinstance(event["at"], (int, float)),
            "decision_native_input_invalid",
        )
        require(
            record["phase"] == row["phase"]
            and record["url"].endswith(endpoint)
            and json.loads(record["request_body_utf8"]) == request
            and json.loads(record["response_body_utf8"]) == payload
            and request["confirmation"] == confirmation
            and request["expected_revision"] == before["revision"]
            and request["idempotency_key"].startswith(endpoint[1:] + ":"),
            "decision_exact_http_invalid",
        )
        require(
            before["session_id"] == after["session_id"] == restored["session"]["session_id"]
            and before["trip_id"] == after["trip_id"] == restored["session"]["trip_id"]
            and before["run_id"] == after["run_id"] == restored["session"]["run_id"]
            and before["accepted_plan_id"] == previous
            and decision["previous_plan_id"] == previous
            and decision["session_revision_before"] == before["revision"]
            and decision["session_revision_after"] == after["revision"]
            and after["revision"] > before["revision"],
            "decision_session_lineage_invalid",
        )
        target = assertions[label]
        require(
            decision["decision_type"] == kind
            and decision["selected_plan_id"] == target["active_plan_id"]
            and decision["selected_plan_content_hash"] == target["active_plan_content_hash"]
            and after["accepted_plan_id"] == restored["session"]["accepted_plan_id"] == target["active_plan_id"]
            and after["draft"] == restored["session"]["draft"] == []
            and after["proposal"] is None,
            "decision_canonical_result_invalid",
        )
        for value in (decision, pointer):
            digest = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in value.items() if key != "content_hash"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            require(value["content_hash"] == digest, "decision_durable_hash_invalid")
        projection = {
            key: pointer[key]
            for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
        }
        require(
            payload["workspace"]["accepted_plan_pointer"]
            == restored["workspace"]["accepted_plan_pointer"]
            == projection
            and pointer["decision_id"] == decision["decision_id"]
            and pointer["revision"] == decision["pointer_revision_after"]
            and pointer["accepted_plan_id"] == target["active_plan_id"]
            and pointer["accepted_plan_content_hash"] == target["active_plan_content_hash"]
            and pointer["original_plan_id"] == assertions["a"]["active_plan_id"]
            and pointer["original_plan_content_hash"] == assertions["a"]["active_plan_content_hash"],
            "decision_pointer_invalid",
        )
        require(
            all(
                row["rendered_after"][key] == target[key]
                for key in (
                    "active_plan_id",
                    "active_plan_content_hash",
                    "rendered_map_plan_id",
                    "rendered_geometry_hash",
                    "rendered_marker_ids",
                )
            ),
            "decision_map_invalid",
        )
        if accepted:
            require(
                all(
                    row["rendered_before"][key] == target[key]
                    for key in (
                        "active_plan_id",
                        "active_plan_content_hash",
                        "rendered_map_plan_id",
                        "rendered_geometry_hash",
                        "rendered_marker_ids",
                    )
                )
                and row["rendered_before"]["preview_state"] == "PREVIEW_RENDERED",
                "accept_not_exact_rendered_preview",
            )
        else:
            prior = next(
                value
                for value in assertions.values()
                if isinstance(value, dict) and value.get("active_plan_id") == previous
            )
            require(
                all(
                    row["rendered_before"][key] == prior[key]
                    for key in (
                        "active_plan_id",
                        "active_plan_content_hash",
                        "rendered_map_plan_id",
                        "rendered_geometry_hash",
                        "rendered_marker_ids",
                    )
                ),
                "keep_not_current_parent",
            )
        previous = target["active_plan_id"]
    return [
        {
            "binding_id": row["binding_id"],
            "selector": row["selector"],
            "context": row["context"],
            "decision_id": row["response"]["result"]["decision"]["decision_id"],
            "phase": row["phase"],
        }
        for row in rows
    ]


def require(value, reason):
    core._require(value, "e2e1_control_action_" + reason)


def parameter_fields(kind, parameters, fields):
    """Only public form values, independent from the production serializer."""
    checks = {
        "set_stop_role": lambda: fields["role"] == parameters["role"],
        "set_stop_day": lambda: int(fields["day"]) == parameters["day"],
        "set_stop_order": lambda: (
            int(fields["sequence_position"]) - 1 == parameters["sequence_index"]
            and int(fields["day"]) == parameters["day"]
        ),
        "set_stop_time_window": lambda: (
            (fields["earliest_arrival"] or None) == parameters["earliest_arrival"]
            and (fields["latest_departure"] or None) == parameters["latest_departure"]
        ),
        "set_stop_commitment": lambda: all(fields[key] == parameters[key] for key in ("strength", "scope_lifetime")),
        "report_route_issue": lambda: all(fields[key] == parameters[key] for key in ("issue_type", "note")),
    }
    if kind == "set_stop_duration":
        duration = parameters["duration"]
        require(fields["duration_mode"] == duration["mode"], "duration_mode_unbound")
        for component in ("preferred", "minimum", "maximum"):
            value = duration[component + "_minutes"]
            if value is not None and (duration["mode"] != "exact" or component == "preferred"):
                require(float(fields["duration-" + component]) == value, "duration_value_unbound")
    elif kind == "set_attribute_constraint":
        require(
            fields["attribute"] == parameters["attribute"]
            and all(fields[key] == parameters[key] for key in ("strength", "scope_lifetime")),
            "attribute_kind_unbound",
        )
        attribute, value = parameters["attribute"], parameters["value"]
        if attribute == "duration":
            require(
                fields["attribute_duration_mode"] == value["mode"] == "exact"
                and float(fields["attribute-duration-preferred"]) == value["preferred_minutes"],
                "attribute_duration_unbound",
            )
        elif attribute == "sequence_order":
            require(
                int(fields["attribute_order_day"]) == value["day"]
                and int(fields["attribute_sequence_index"]) == value["sequence_index"],
                "attribute_order_unbound",
            )
        else:
            actual = fields["attribute_" + attribute]
            require((int(actual) if attribute == "day" else actual) == value, "attribute_value_unbound")
    else:
        require(kind in checks and checks[kind](), "form_parameter_unbound")


def verify_rows(rows, records, expected_context=None):
    require(isinstance(rows, list) and rows, "receipts_missing")
    proofs = []
    operation_ids = []
    for row in rows:
        request, response, control = row["request"], row["response"], row["control"]
        operation = response["operation"]
        if expected_context:
            require(
                row["context"] == ("original" if row["phase"] == "baseline_replace_accept" else expected_context),
                "context_unbound",
            )
        require(
            row["assertions_passed"] is True
            and row["context"] in {"original", "accepted_child"}
            and control["visible"] is True
            and control["disabled"] is False
            and request["source"] == "map"
            and all(request[key] == operation[key] for key in ("type", "target", "parameters")),
            "input_semantics_invalid",
        )
        matches = [
            record
            for record in records
            if record["method"] == "POST"
            and record["url"].endswith("/draft/operations")
            and json.loads(record["response_body_utf8"]) == response
        ]
        require(
            len(matches) == 1
            and matches[0]["phase"] == row["phase"]
            and json.loads(matches[0]["request_body_utf8"]) == request,
            "http_unbound",
        )
        events, start, end = row["raw_events"], row["event_start"], row["event_end"]
        require(type(start) is type(end) is int and 0 <= start < end == len(events), "event_range_invalid")
        kind = request["type"]
        selector = '.typed-operation-form[data-typed-operation="' + kind + '"]'
        if kind in LEGACY:
            require(
                control["id"] == IDS[kind]
                and any(
                    event["type"] == "click"
                    and event["trusted"] is True
                    and event["id"] == IDS[kind]
                    and event["path"] == control["path"]
                    for event in events[start:end]
                ),
                "legacy_click_missing",
            )
            aliases = [(LEGACY[kind], "#" + IDS[kind])]
        else:
            require(
                control["operation"] == kind and type(control["form_id"]) is int and control["form_id"] > 0,
                "wrong_form",
            )
            parameter_fields(kind, request["parameters"], control["fields"])
            submitted = [
                event
                for event in events[start:end]
                if event["type"] == "submit"
                and event["trusted"] is True
                and event["operation"] == kind
                and event["form_id"] == control["form_id"]
                and event["fields"] == control["fields"]
                and event["path"] == control["path"]
            ]
            require(len(submitted) == 1, "native_form_submit_missing")
            aliases = [("typed_submit", selector)]
            # Bind a changed field to this exact mounted form, not a same-named earlier editor.
            # Other fields can legitimately be filled later before the native submission.
            changes = [
                event
                for event in events
                if event["type"] == "change"
                and event["trusted"] is True
                and event["operation"] == kind
                and event["form_id"] == control["form_id"]
                and event["at"] <= submitted[0]["at"]
                and event["value"] == control["fields"].get(event["name"])
                and event["path"] == control["path"]
            ]
            if changes:
                aliases.append(("typed_combination", selector))
            for name, binding, alias in (
                ("duration_mode", "duration_mode", '[data-duration-mode][name="duration_mode"]'),
                ("attribute_duration_mode", "duration_mode", '[data-duration-mode][name="attribute_duration_mode"]'),
                ("role", "role_value", "[data-role-select]"),
                (
                    "attribute",
                    "attribute_value",
                    '[data-typed-operation="set_attribute_constraint"] [name="attribute"]',
                ),
            ):
                if any(event["name"] == name and event["value"] == control["fields"].get(name) for event in changes):
                    aliases.append((binding, alias))
        operation_ids.append(operation["operation_id"])
        proofs.extend(
            {
                "binding_id": binding,
                "selector": alias,
                "context": row["context"],
                "phase": row["phase"],
                "operation_id": operation["operation_id"],
                "status": "executed",
                "scope": "Real native input produced this exact canonical staged operation, not an evaluated-repair claim",
            }
            for binding, alias in aliases
        )
    require(len(operation_ids) == len(set(operation_ids)), "duplicate_operation_receipt")
    expected = [
        json.loads(record["response_body_utf8"])["operation"]["operation_id"]
        for record in records
        if record["method"] == "POST" and record["url"].endswith("/draft/operations")
    ]
    require(
        set(expected) == set(operation_ids) and len(expected) == len(operation_ids),
        "operation_receipt_universe_incomplete",
    )
    return proofs


def disabled_day(value, capability, records, context, parent):
    before, after = value["before"], value["after"]
    control = after["control"]
    events = value["events"]
    require(
        value["context"] == context
        and value["assertions_passed"] is True
        and value["capability"] == capability
        and capability["enabled"] is False
        and value["parent"] == value["map"] == parent,
        "disabled_day_scope_unbound",
    )
    require(
        before["control"]["operation"] == control["operation"] == "set_stop_day"
        and before["control"]["form_id"] == control["form_id"]
        and type(control["form_id"]) is int
        and before["control"]["disabled"] is control["disabled"] is True
        and before["control"]["visible"] is control["visible"] is True
        and control["fields"]["day"] == "6",
        "disabled_day_form_unbound",
    )
    require(
        type(before["event_start"]) is type(after["event_start"]) is int
        and 0 <= before["event_start"] < after["event_start"] == len(events)
        and any(
            event["type"] == "change"
            and event["trusted"] is True
            and event["operation"] == "set_stop_day"
            and event["form_id"] == control["form_id"]
            and event["name"] == "day"
            and event["value"] == "6"
            and event["path"] == control["path"]
            for event in events[before["event_start"] :]
        ),
        "disabled_day_change_missing",
    )
    require(
        not any(
            record["phase"] == value["phase"]
            and record["method"] == "POST"
            and record["url"].endswith(("/draft/operations", "/preview", "/accept"))
            for record in records
        ),
        "disabled_day_mutated",
    )
    return {
        "scope": "Enabled day selection actually changed; submit remained visible and disabled by the current server capability",
        "unsupported_claimed": False,
    }


def legacy_keep(value, records, context, parent):
    require(
        value["context"] == context
        and value["phase"] == "legacy_keep_control"
        and value["parent"] == parent
        and value["assertions_passed"] is True
        and value["evaluated_claimed"] is False,
        "legacy_keep_scope_invalid",
    )
    operation = value["stage"]["operation"]
    require(
        operation["type"] == "keep_stop"
        and operation["target"] == "griffith_observatory"
        and operation["parameters"] == {}
        and value["stage"]["session"]["draft"] == [operation]
        and value["undo"]["undone"] == operation
        and value["undo"]["session"]["draft"] == []
        and value["stage"]["session"]["accepted_plan_id"]
        == value["undo"]["session"]["accepted_plan_id"]
        == parent["active_plan_id"]
        and all(
            value[label][key] == parent[key]
            for label in ("staged_map", "undone_map")
            for key in (
                "active_plan_id",
                "active_plan_content_hash",
                "rendered_map_plan_id",
                "rendered_geometry_hash",
                "rendered_marker_ids",
            )
        ),
        "legacy_keep_state_changed",
    )
    actual = [
        record
        for record in records
        if record["phase"] == value["phase"]
        and record["method"] == "POST"
        and record["url"].endswith(("/draft/operations", "/draft/undo", "/preview", "/accept"))
    ]
    require(
        len(actual) == 2
        and actual[0]["url"].endswith("/draft/operations")
        and actual[1]["url"].endswith("/draft/undo")
        and json.loads(actual[0]["response_body_utf8"]) == value["stage"]
        and json.loads(actual[1]["response_body_utf8"]) == value["undo"],
        "legacy_keep_http_unbound",
    )
