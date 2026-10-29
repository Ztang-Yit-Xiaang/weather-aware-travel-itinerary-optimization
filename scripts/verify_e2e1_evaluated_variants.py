"""Independent typed variant preview/reload/Undo gate; never claims per-variant acceptance."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _proposal

SCENARIO = "evaluated_variants"
TEST = "test_real_evaluated_variants_preview_reload_and_undo"
CONTEXTS = ("original", "accepted_child")
VARIANTS = (
    *(("role", role, "stearns_wharf") for role in ("attraction", "activity", "meal", "rest_stop", "scenic_stop")),
    *(("time_window", mode, "stearns_wharf") for mode in ("earliest_only", "latest_only", "both")),
    ("duration", "exact", "griffith_observatory"),
    ("order", "default", "sequoia_national_park"),
)


def require(condition, reason):
    core._require(condition, "e2e1_variants_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def control_aliases(value, timing, context, position, proposal, operation):
    require(
        value["schema_version"] == "evaluated-action-aliases-v1"
        and value["context"] == context
        and value["assertions_passed"] is True
        and value["preview"] == timing["control"]
        and value["request_id"] == proposal["compiled_request"]["source_request_id"]
        and value["operation_id"] == operation["operation_id"],
        "alias_artifact_unbound",
    )
    for action in ("preview", "undo"):
        expected = (
            ("#inline-", "#draft-", "#context-")[position % 3] + action
            if action == "preview"
            else ("#inline-", "#context-")[position % 2] + action
        )
        row = value[action]
        require(
            row["selector"] == expected
            and row["visible"] is True
            and row["disabled"] is False
            and row["raw_clicks"]
            and row["event"] == row["raw_clicks"][-1]
            and row["event"]["trusted"] is True
            and row["event"]["id"] == expected[1:]
            and row["event"]["path"]
            == row["path"]
            == (
                "/app/repairs"
                if (action == "preview" and position % 3 == 2) or (action == "undo" and position % 2 == 1)
                else "/app/itinerary"
            ),
            "alias_control_not_executed",
        )


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp_timezone_missing")
    return result


def identity(first, second):
    return all(
        first.get(key) == second.get(key)
        for key in (
            "active_plan_id",
            "active_plan_content_hash",
            "rendered_map_plan_id",
            "rendered_geometry_hash",
            "rendered_marker_ids",
        )
    )


def artifact_chain(proposal, staged, trace, outcome):
    parent = trace["parent"]
    certificate = (
        proposal.get("evaluation_certificate")
        or proposal.get("certificate_schedule_evidence", {}).get("certificate_record")
        or proposal.get("certificate_role_evidence", {}).get("certificate_record")
    )
    diff, geography = proposal["plan_diff"], proposal["geography_plan"]
    require(
        proposal["execution_mode"] == "typed_direct_edit_independent_evaluation"
        and proposal["state"] == proposal["eligibility"] == trace["comparison_eligibility"] == outcome
        and proposal["parent_plan_id"] == trace["parent_plan_id"] == parent["active_plan_id"]
        and proposal["parent_plan_content_hash"]
        == trace["parent_plan_content_hash"]
        == parent["active_plan_content_hash"]
        and proposal["child_plan_id"] == trace["child_plan_id"] != parent["active_plan_id"]
        and proposal["child_plan_content_hash"]
        == trace["child_plan_content_hash"]
        != parent["active_plan_content_hash"]
        and proposal["compiled_request"]["source_request_id"] == trace["request_id"]
        and proposal["compiled_request"]["operations"]
        == [{key: value for key, value in staged.items() if key != "created_at"}],
        "canonical_request_mismatch",
    )
    require(
        certificate is not None
        and certificate["comparison_eligibility"] == outcome
        and certificate["plan_id"] == diff["child_plan_id"] == geography["plan_id"] == proposal["child_plan_id"]
        and certificate["plan_content_hash"] == geography["content_hash"] == proposal["child_plan_content_hash"]
        and certificate["certificate_id"] == proposal["certificate_id"] == trace["evaluation_id"]
        and core._artifact_content_hash(certificate)
        == certificate["content_hash"]
        == proposal["certificate_content_hash"]
        and core._artifact_content_hash(diff) == proposal["diff_content_hash"]
        and diff["diff_id"] == proposal["diff_id"]
        and diff["parent_plan_id"] == parent["active_plan_id"],
        "certificate_diff_mismatch",
    )
    failures = [row["code"] for row in certificate["failures"]]
    require(
        failures == trace["failure_codes"] == (["weather_infeasible"] if outcome == "ineligible" else [])
        and proposal["acceptance_eligible"] is (outcome == "eligible"),
        "unexpected_eligibility_reason",
    )
    route = certificate["route_validation"]
    require(
        route["required_leg_count"] > 0
        and route["road_validated_leg_count"] == route["required_leg_count"]
        and route["missing_leg_count"] == route["fallback_leg_count"] == 0
        and geography["coverage"]["complete"] is True
        and geography["coverage"]["gap_count"] == 0
        and trace["child_geometry_hash"] == core._completed_race_geometry_hash(geography),
        "route_not_canonical_complete",
    )
    if outcome == "eligible":
        core._verify_plan_ack(trace["rendered"], proposal)
    else:
        require(identity(trace["rendered"], parent), "ineligible_child_rendered")
    require(
        trace["rendered"]["preview_state"] == ("PREVIEW_RENDERED" if outcome == "eligible" else "PREVIEW_INELIGIBLE")
        and trace["reloaded"] == trace["rendered"]
        and identity(trace["staged"], parent)
        and identity(trace["restored"], parent)
        and trace["restored"]["preview_state"] == "PARENT_RENDERED",
        "render_state_or_reload_mismatch",
    )
    return certificate, diff, geography


def semantics(kind, mode, target, operation, preconditions, proposal, parent_geography, earliest_proposal=None):
    parameters = operation["parameters"]
    before = next(
        row["properties"] for row in parent_geography["stops"]["features"] if row["properties"]["stop_id"] == target
    )
    after = next(
        row["properties"]
        for row in proposal["geography_plan"]["stops"]["features"]
        if row["properties"]["stop_id"] == target
    )
    require(
        preconditions["parent_stop"] == before
        and operation["type"] == "set_stop_" + kind
        and operation["target"] == target
        and operation["source"] == "map",
        "semantic_input_mismatch",
    )
    diff = proposal["plan_diff"]
    if kind == "role":
        require(
            parameters == {"role": mode}
            and before.get("itinerary_role") != mode
            and after["itinerary_role"] == mode
            and len(diff["role_changes"]) == 1
            and diff["role_changes"][0]["stop_id"] == target
            and diff["role_changes"][0]["from_role"] == before.get("itinerary_role")
            and diff["role_changes"][0]["to_role"] == mode
            and proposal["role_impact"]["semantic_scope"]["place_identity_unchanged"] is True
            and proposal["role_impact"]["semantic_scope"]["semantic_fit_claim"] == "unavailable",
            "role_semantics_mismatch",
        )
    elif kind == "duration":
        require(
            parameters
            == {"duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}}
            and after["visit_duration_minutes"] == 60
            and len(diff["duration_changes"]) == 1
            and diff["duration_changes"][0]["stop_id"] == target
            and diff["duration_changes"][0]["to_minutes"] == 60,
            "duration_semantics_mismatch",
        )
    elif kind == "order":
        require(
            parameters == {"day": 4, "sequence_index": 1}
            and after["stop_order"] == 2
            and before["stop_order"] != after["stop_order"],
            "order_semantics_mismatch",
        )
    else:
        earliest = "10:00" if mode != "latest_only" else None
        latest = None
        if mode != "earliest_only":
            require(earliest_proposal is not None, "latest_bound_evaluation_missing")
            source = next(
                row for row in earliest_proposal["schedule_impact"]["child"]["stop_traces"] if row["stop_id"] == target
            )
            minute = math.ceil(source["departure_minute"] / 30) * 30 + 30
            require(0 < minute < 1440, "latest_bound_out_of_day")
            latest = f"{minute // 60:02d}:{minute % 60:02d}"
            require(
                preconditions["latest_bound_source"]
                == {
                    "case": "time_window_earliest_only",
                    "child_plan_id": earliest_proposal["child_plan_id"],
                    "evaluation_id": earliest_proposal["certificate_id"],
                    "trace": source,
                    "chosen_latest_departure": latest,
                },
                "latest_bound_not_evidence_derived",
            )
        else:
            require(preconditions["latest_bound_source"] is None, "unexpected_latest_bound_source")
        changes = diff["time_window_changes"]
        trace = next(row for row in proposal["schedule_impact"]["child"]["stop_traces"] if row["stop_id"] == target)
        require(
            parameters == {"earliest_arrival": earliest, "latest_departure": latest}
            and len(changes) == 1
            and changes[0]["stop_id"] == target
            and changes[0]["to_earliest_arrival"] == earliest
            and changes[0]["to_latest_departure"] == latest
            and (changes[0]["from_earliest_arrival"], changes[0]["from_latest_departure"]) != (earliest, latest)
            and (not earliest or trace["service_start_minute"] >= 600)
            and (
                not latest
                or (
                    trace["departure_minute"] <= int(latest[:2]) * 60 + int(latest[3:])
                    and trace["latest_departure_status"] == "satisfied"
                )
            ),
            "time_window_semantics_mismatch",
        )


def surfaces(record, geography, proposal, evaluation_id):
    for scope in (record["metrics"], record["text_route"]["scope"]):
        require(
            scope["planId"] == geography["plan_id"] and scope["planContentHash"] == geography["content_hash"],
            "visible_plan_scope_mismatch",
        )
    require(
        record["metrics"]["evaluationId"] == evaluation_id and record["text_route"]["text"],
        "visible_evaluation_mismatch",
    )
    expected = []
    for day in range(1, 8):
        stops = sorted(
            (row["properties"] for row in geography["stops"]["features"] if int(row["properties"]["day"]) == day),
            key=lambda stop: stop["stop_order"],
        )
        names = " · ".join(row["name"] for row in stops) or "No recorded stop"
        expected.append({"day": day, "stop_ids": [row["stop_id"] for row in stops], "visible_names": names})
        require(names in record["timeline_text"], "timeline_visible_names_missing")
    require(
        record["timeline"] == expected
        and all(proposal[key] in record["advanced_evidence"] for key in ("child_plan_id", "certificate_id", "diff_id")),
        "timeline_or_advanced_evidence_mismatch",
    )


def _one_http(records, suffix, *, method="POST", request=None, response=None):
    matches = [row for row in records if row["method"] == method and row["url"].endswith(suffix)]
    require(len(matches) == 1, "http_operation_count_invalid")
    row = matches[0]
    require(
        (request is None or json.loads(row["request_body_utf8"]) == request)
        and (response is None or json.loads(row["response_body_utf8"]) == response),
        "exact_http_io_mismatch",
    )
    return row


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require(
        len(cases) == 2
        and [case.get("name") for case in cases] == [TEST + "[" + context + "]" for context in CONTEXTS]
        and all(case.get("classname") == "tests.product_app.browser.test_evaluated_variants_live" for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()),
        "test_inventory_invalid",
    )
    require(
        {path.name for path in (root / SCENARIO).iterdir() if path.is_dir()} == set(CONTEXTS),
        "context_universe_invalid",
    )
    for context in CONTEXTS:
        base = root / SCENARIO / context
        assertions = core._load(base / "browser_assertions.json")
        traces = load(base / "variant_assertions.json")
        require(
            assertions["schema_version"] == "evaluated-variants-browser-v1"
            and assertions["context"] == context
            and assertions["scope"] == "evaluated_preview_only_then_undo"
            and assertions["variant_acceptance_claimed"] is False
            and assertions["baseline_acceptance_through_real_ui"] is (context == "accepted_child")
            and assertions["assertions_passed"] is True
            and assertions["variants"] == traces
            and len(traces) == 10,
            "scope_or_variant_count_invalid",
        )
        require(
            {path.name for path in base.iterdir() if path.is_dir()}
            == {kind + "_" + mode for kind, mode, _ in VARIANTS},
            "variant_directory_universe_invalid",
        )
        records = _http(base, base_url)
        from verify_e2e1_control_actions import verify_rows

        verify_rows(load(base / "control_action_receipts.json"), records, context)
        require(
            all(row.get("phase") for row in records) and not any("/copilot/messages" in row["url"] for row in records),
            "unexpected_chat_or_missing_phase",
        )
        snapshot = core._load(base / "capability_snapshot.json")
        parent = traces[0]["parent"]
        require(
            snapshot["context"] == context
            and snapshot["plan_id"] == parent["active_plan_id"]
            and snapshot["plan_content_hash"] == parent["active_plan_content_hash"]
            and parent["preview_state"] == "PARENT_RENDERED",
            "baseline_identity_invalid",
        )
        responses = [json.loads(row["response_body_utf8"]) for row in records]
        bindings = [
            row
            for row in responses
            if isinstance(row, dict)
            and "workspace" in row
            and "session" in row
            and row["session"]["session_id"] == snapshot["session_id"]
            and row["session"]["accepted_plan_id"] == snapshot["plan_id"]
            and all(row["workspace"].get(group) == contract for group, contract in snapshot["capabilities"].items())
        ]
        require(bindings, "capabilities_http_unbound")
        workspace = bindings[0]["workspace"]
        parent_geography = next(
            plan for plan in workspace["geography"]["plans"] if plan["plan_id"] == parent["active_plan_id"]
        )
        require(
            parent_geography["content_hash"] == parent["active_plan_content_hash"]
            and core._completed_race_geometry_hash(parent_geography) == parent["rendered_geometry_hash"],
            "parent_geography_mismatch",
        )
        parent_certificate = (
            workspace.get("alternative_evidence", {}).get(parent["active_plan_id"], {}).get("certificate")
        )
        parent_evaluation = (
            parent_certificate["certificate_id"]
            if parent_certificate
            and parent_certificate["plan_id"] == parent["active_plan_id"]
            and parent_certificate["plan_content_hash"] == parent["active_plan_content_hash"]
            else ""
        )
        typed = snapshot["capabilities"]["typed_edit_capabilities"]["operations"]
        require(
            set(typed["set_stop_role"]["supported_evaluated_roles"])
            == {mode for kind, mode, _ in VARIANTS if kind == "role"}
            and typed["set_stop_duration"]["supported_evaluated_modes"] == ["exact"]
            and all(
                typed["set_stop_" + kind]["enabled"] is True for kind in ("role", "time_window", "duration", "order")
            ),
            "evaluated_capability_universe_changed",
        )
        inventory = core._load(base / "runtime_control_inventory.json")
        require(
            inventory["context"] == context and len(inventory["observations"]) == 10, "control_inventory_incomplete"
        )
        earliest_proposal = None
        for position, (kind, mode, target) in enumerate(VARIANTS):
            phase = kind + "_" + mode
            case = base / phase
            trace = core._load(case / "trace.json")
            operation = core._load(case / "operation_input.json")
            staged = core._load(case / "stage_response.json")
            payload = core._load(case / "preview_response.json")
            proposal = payload["proposal"]
            require(
                trace == traces[position]
                and trace["schema_version"] == "evaluated-variant-browser-v1"
                and trace["context"] == context
                and trace["operation"] == "set_stop_" + kind
                and trace["variant"] == mode
                and trace["feedback_tier"] == "evaluated"
                and trace["scope"] == "evaluated_preview_only_then_undo"
                and trace["variant_acceptance_claimed"] is False
                and trace["assertions_passed"] is True
                and trace["parent"] == parent
                and trace["operation_input"] == operation
                and trace["operation_id"] == staged["operation"]["operation_id"],
                "variant_trace_mismatch",
            )
            require(
                staged["session"]["draft"] == payload["session"]["draft"] == [staged["operation"]]
                and all(staged["operation"][key] == operation[key] for key in ("type", "target", "parameters"))
                and staged["session"]["accepted_plan_id"]
                == payload["session"]["accepted_plan_id"]
                == parent["active_plan_id"]
                and payload["session"]["proposal"] == proposal,
                "session_draft_binding_invalid",
            )
            outcome = "ineligible" if context == "original" else "eligible"
            certificate, diff, geography = artifact_chain(proposal, staged["operation"], trace, outcome)
            require(
                core._load(case / "evaluation_summary.json") == certificate
                and core._load(case / "plan_diff.json") == diff
                and core._load(case / "map_render_ack.json") == trace["rendered"],
                "canonical_projection_mismatch",
            )
            semantics(
                kind,
                mode,
                target,
                operation,
                core._load(case / "edit_preconditions.json"),
                proposal,
                parent_geography,
                earliest_proposal,
            )
            if phase == "time_window_earliest_only":
                earliest_proposal = proposal
            observation = inventory["observations"][position]
            require(
                observation["context"] == context
                and observation["operation"] == operation["type"]
                and observation["variant"] == mode
                and observation["visible"] is True
                and observation["disabled"] is False
                and observation["selector"] == f'[data-typed-operation="{operation["type"]}"] button[type="submit"]'
                and observation["form_text"]
                and observation["fields"],
                "control_variant_unbound",
            )
            phase_records = [row for row in records if row["phase"] == phase]
            _one_http(phase_records, "/draft/operations", request=operation, response=staged)
            timing = core._load(case / "preview_timing.json")
            _one_http(phase_records, "/preview", request=timing["request_body"], response=payload)
            impact = core._load(case / "draft_impact_response.json")
            require(
                any(
                    row["url"].endswith("/draft/impact-preview") and json.loads(row["response_body_utf8"]) == impact
                    for row in phase_records
                )
                and impact["session_revision"] == staged["session"]["revision"]
                and impact["is_certified"] is False
                and impact["parent_plan_id"] == parent["active_plan_id"]
                and impact["parent_plan_content_hash"] == parent["active_plan_content_hash"]
                and impact["summary"]["evaluated_executable_count"] == 1
                and impact["summary"]["draft_only_count"] == 0
                and impact["summary"]["can_run_evaluated_preview"] is True
                and len(impact["operations"]) == 1
                and impact["operations"][0]["operation_id"] == staged["operation"]["operation_id"],
                "impact_not_bound",
            )
            restored = core._load(case / "undo_response.json")
            control_aliases(
                core._load(case / "action_aliases.json"),
                core._load(case / "preview_timing.json"),
                context,
                position,
                proposal,
                staged["operation"],
            )
            _one_http(phase_records, "/draft/undo", response=restored)
            require(
                restored["undone"] == staged["operation"]
                and restored["session"]["draft"] == []
                and restored["session"]["proposal"] is None
                and restored["session"]["accepted_plan_id"] == parent["active_plan_id"],
                "undo_not_restored",
            )
            reload = _one_http(phase_records, "/api/sessions/" + snapshot["session_id"], method="GET")
            session = json.loads(reload["response_body_utf8"])["session"]
            require(
                session["accepted_plan_id"] == parent["active_plan_id"]
                and session["draft"] == [staged["operation"]]
                and session["proposal"] == proposal,
                "preview_reload_not_persisted",
            )
            displayed = parent_geography if context == "original" else geography
            surfaces(
                core._load(case / "visible_surfaces.json"),
                displayed,
                proposal,
                parent_evaluation if context == "original" else certificate["certificate_id"],
            )
            labels = ["staged_parent", "preview_result", "after_reload", "undo_restored_parent"]
            if context == "accepted_child":
                panel = core._load(case / "evidence_panel.json")
                require(
                    panel["scope"]["evidencePlanId"] == proposal["child_plan_id"]
                    and panel["scope"]["planContentHash"] == proposal["child_plan_content_hash"]
                    and panel["scope"]["evaluationId"] == certificate["certificate_id"],
                    "evidence_panel_mismatch",
                )
                labels.append("compare")
            core._verify_journey_captures(case, labels, {parent["active_plan_id"], proposal["child_plan_id"]})
            ready = core._load(case / "preview_result.readiness.json")
            require(
                stamp(timing["started_at"])
                <= stamp(timing["received_at"])
                <= stamp(ready["screenshot_started_at"])
                <= stamp(ready["screenshot_completed_at"])
                <= stamp(trace["completed_at"]),
                "preview_timing_invalid",
            )
        posts = [row for row in records if row["method"] == "POST"]
        baseline_count = int(context == "accepted_child")
        for suffix, expected_count in (
            ("/api/sessions", 1),
            ("/draft/operations", 10 + baseline_count),
            ("/preview", 10 + baseline_count),
            ("/accept", baseline_count),
            ("/draft/undo", 10),
        ):
            require(
                sum(row["url"].endswith(suffix) for row in posts) == expected_count,
                "mutation_count_or_false_acceptance",
            )
        labels = ["initial", "baseline"]
        if baseline_count:
            baseline = core._load(base / "baseline_preview_response.json")
            accepted = core._load(base / "baseline_acceptance_response.json")
            baseline_records = [row for row in records if row["phase"] == "baseline_replace_accept"]
            _one_http(baseline_records, "/preview", response=baseline)
            _one_http(baseline_records, "/accept", response=accepted)
            _proposal(baseline["proposal"], parent)
            decision = accepted["result"]["decision"]
            require(
                decision["selected_plan_id"] == parent["active_plan_id"]
                and decision["selected_plan_content_hash"] == parent["active_plan_content_hash"]
                and decision["certificate_id"] == baseline["proposal"]["certificate_id"]
                and decision["certificate_content_hash"] == baseline["proposal"]["certificate_content_hash"],
                "baseline_acceptance_not_bound",
            )
            labels.append("baseline_preview")
        initial_plans = {
            row["session"]["accepted_plan_id"] for row in responses if isinstance(row, dict) and "session" in row
        }
        core._verify_journey_captures(base, labels, initial_plans)
        require(load(base / "console_errors.json") == assertions["console_errors"] == [], "browser_errors")
        failures = load(base / "network_failures.json")
        require(failures == assertions["network_failures"], "network_observations_unbound")
        from verify_e2e1_network_observations import verify as verify_network
        from verify_e2e1_network_observations import verify_asset_observation

        assets = load(base / "map_asset_sources.json")
        verify_asset_observation(assets, traces[-1]["restored"], context, phase="final_network_observation")
        network = verify_network(
            base,
            base_url,
            failures,
            {"sources": assets["vector_sources"], "sprite": assets["sprite"]},
            asset_observation=assets,
        )
        network_document = load(base / "network_classification.json")
        classification_keys = {
            "schema_version",
            "policy",
            "raw_record_count",
            "classifications",
            "browser_cancelled_local_map_asset_indexes",
            "blocking_failure_indexes",
        }
        require(
            set(network_document)
            == classification_keys | {"map_configuration_http_file", "rendered_map_asset_sources_file"}
            and assertions["network_classification"] == {key: network_document[key] for key in classification_keys}
            and network_document["map_configuration_http_file"] == assets["map_configuration_http_file"]
            and network_document["rendered_map_asset_sources_file"] == "map_asset_sources.json"
            and network["blocking_network_failures"] == 0,
            "network_classification_unbound",
        )
    return {
        "verified": True,
        "test_count": 2,
        "evaluated_variants": 20,
        "scope": "evaluated_preview_reload_undo",
        "variant_acceptance_claimed": False,
        "provider_calls": 0,
    }


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        StopIteration,
        ET.ParseError,
    ) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_variants_evidence_malformed") from error
