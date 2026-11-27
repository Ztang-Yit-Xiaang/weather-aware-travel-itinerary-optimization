"""Counterexamples for the independent real-control evidence verifier."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fault", [None, "old_session", "future_workspace", "foreign_owner", "newer_workspace", "missing_action"]
)
def test_prior_selection_session_and_workspace_are_separate_real_responses(fault):
    import json

    import verify_e2e1_capability_controls as verifier

    old = {"session_id": "s", "accepted_plan_id": "a", "revision": 1}
    current = {**old, "revision": 2}
    workspace = {
        "geography": {
            "plans": [
                {"plan_id": "a", "content_hash": "hash", "stops": {"features": []}, "validated_legs": {"features": []}}
            ]
        },
        "alternatives": [],
        "draft_capabilities": {"candidate_choices": []},
    }
    row = {
        "session_before": current,
        "workspace_before": verifier.selection_workspace_projection(workspace),
        "phase": "new_selection",
    }
    records = [
        {"phase": "create", "response_body_utf8": json.dumps({"session": old, "workspace": workspace})},
        {"phase": "prior_selection", "response_body_utf8": json.dumps({"session": current})},
        {"phase": "new_selection", "response_body_utf8": "{}"},
    ]
    if fault == "old_session":
        row["session_before"] = old
    elif fault == "future_workspace":
        records.append(records.pop(0))
    elif fault == "foreign_owner":
        records[0]["response_body_utf8"] = json.dumps(
            {"session": {**old, "session_id": "foreign"}, "workspace": workspace}
        )
    elif fault == "newer_workspace":
        records.insert(
            2, {"phase": "later_workspace", "response_body_utf8": json.dumps({"session": current, "workspace": {}})}
        )
    elif fault == "missing_action":
        records.pop()
    if fault:
        with pytest.raises(verifier.core.E2E1VerificationError):
            verifier.selection_prior_sources(row, records)
    else:
        verifier.selection_prior_sources(row, records)


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_capability_controls as controls  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402


def test_control_mode_binds_implementation_inventory_and_independent_verifier():
    inputs = deterministic_source_inputs("tests/product_app/browser/test_capability_controls_live.py")
    assert {
        "scripts/verify_e2e1_capability_controls.py",
        "scripts/verify_e2e1_self_managed.py",
        "scripts/e2e1_interaction_coverage.py",
        "configs/e2e1_interaction_coverage.json",
    } <= set(inputs)
    assert len(inputs) == len(set(inputs))
    assert all((core.ROOT / path).is_file() for path in inputs)


@pytest.mark.parametrize(
    "operation,parameters,expected",
    [
        ("set_stop_duration", {"duration": {"mode": "range"}}, "range"),
        ("set_stop_role", {"role": "lodging"}, "lodging"),
        ("set_stop_commitment", {"strength": "prefer_keep"}, "prefer_keep"),
        ("set_attribute_constraint", {"attribute": "day"}, "attribute:day"),
        ("report_route_issue", {"issue_type": "wrong_geometry"}, "wrong_geometry"),
    ],
)
def test_variant_is_derived_from_actual_submitted_parameters(operation, parameters, expected):
    assert controls.variant({"type": operation, "parameters": parameters}) == expected


def _disabled_fixture():
    caps = {
        "typed_edit_capabilities": {
            "operations": {
                "set_stop_day": {"enabled": False},
                "set_stop_commitment": {"enabled": True, "protected_strengths": []},
                "set_attribute_constraint": {"enabled": True, "protected_strengths": []},
            }
        }
    }
    rows = [
        {
            "operation": "set_stop_day",
            "variant": "disabled",
            "context": "original",
            "visible": True,
            "disabled": True,
            "forced_interaction": False,
            "draft_request_delta": 0,
            "label": "Unavailable",
        }
    ]
    inventory = [
        {"operation": "set_stop_day", "controls": [{"disabled": True, "visible": True, "label": "Unavailable"}]}
    ]
    return rows, caps, inventory


def test_disabled_action_requires_actual_disabled_visible_dom():
    rows, caps, inventory = _disabled_fixture()
    controls._disabled(rows, caps, inventory, "original")
    inventory[0]["controls"][0]["disabled"] = False
    with pytest.raises(core.E2E1VerificationError, match="disabled_dom_unbound"):
        controls._disabled(rows, caps, inventory, "original")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "hidden", "forced", "mutated"])
def test_disabled_claims_cannot_hide_missing_or_active_controls(fault):
    rows, caps, inventory = _disabled_fixture()
    if fault == "missing":
        rows.clear()
    elif fault == "duplicate":
        rows += deepcopy(rows)
    elif fault == "hidden":
        inventory[0]["controls"][0]["visible"] = False
    elif fault == "forced":
        rows[0]["forced_interaction"] = True
    else:
        rows[0]["draft_request_delta"] = 1
    with pytest.raises(core.E2E1VerificationError):
        controls._disabled(rows, caps, inventory, "original")


def test_missing_control_bundle_rejects_stably(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_controls_evidence_malformed"):
        controls.verify(tmp_path, "http://127.0.0.1:8127")


@pytest.mark.parametrize("first", ["duration", "replacement"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "exact",
        "advertised_evaluated",
        "unadvertised",
        "reversed",
        "hidden",
        "enabled",
        "input_mismatch",
        "missing_second",
        "second_post",
    ],
)
def test_cross_family_negative_requires_actual_unsupported_mode_and_disabled_input(first, fault):
    duration = {
        "type": "set_stop_duration",
        "target": "griffith_observatory",
        "parameters": {
            "duration": {"mode": "range", "preferred_minutes": None, "minimum_minutes": 45, "maximum_minutes": 90}
        },
    }
    replacement = {
        "type": "replace_nearby",
        "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
    }
    operations = [duration, replacement] if first == "duration" else [replacement, duration]
    control = {
        "id": "draft-replace" if first == "duration" else None,
        "operation": None if first == "duration" else "set_stop_duration",
        "path": "/app/itinerary",
        "visible": True,
        "disabled": True,
        "fields": None
        if first == "duration"
        else {"duration_mode": "range", "duration-minimum": "45", "duration-maximum": "90"},
    }
    row = {
        "first": first,
        "first_operation": operations[0],
        "second_operation": operations[1],
        "second_control": {"control": control},
        "selected_candidate_id": "pfeiffer_big_sur_state_park",
        "assertions_passed": True,
        "second_operation_disabled": True,
        "second_stage_request_delta": 0,
    }
    typed = {
        "operations": {"set_stop_duration": {"draft_only_modes": ["range"], "supported_evaluated_modes": ["exact"]}}
    }
    if fault == "exact":
        duration["parameters"]["duration"]["mode"] = "exact"
    elif fault == "advertised_evaluated":
        typed["operations"]["set_stop_duration"]["supported_evaluated_modes"].append("range")
    elif fault == "unadvertised":
        typed["operations"]["set_stop_duration"]["draft_only_modes"].clear()
    elif fault == "reversed":
        row["first_operation"], row["second_operation"] = row["second_operation"], row["first_operation"]
    elif fault in {"hidden", "enabled"}:
        control["visible" if fault == "hidden" else "disabled"] = False
    elif fault == "input_mismatch":
        if first == "duration":
            row["selected_candidate_id"] = "point_lobos_state_natural_reserve"
        else:
            control["fields"]["duration-minimum"] = "60"
    elif fault == "missing_second":
        row["second_operation"] = {}
    elif fault == "second_post":
        row["second_stage_request_delta"] = 1
    if fault:
        with pytest.raises((core.E2E1VerificationError, KeyError)):
            controls.cross_family_negative(row, typed)
    else:
        assert controls.cross_family_negative(row, typed) == "range"
