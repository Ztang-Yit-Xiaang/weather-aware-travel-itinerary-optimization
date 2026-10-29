"""Adversarial typed preview artifact and semantic evidence fixtures."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_evaluated_variants as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def _fixture(outcome="eligible"):
    geo = {
        "plan_id": "child",
        "content_hash": "childhash",
        "coverage": {"complete": True, "gap_count": 0},
        "stops": {"type": "FeatureCollection", "features": []},
        "validated_legs": {"type": "FeatureCollection", "features": []},
    }
    certificate = {
        "plan_id": "child",
        "plan_content_hash": "childhash",
        "certificate_id": "certificate",
        "comparison_eligibility": outcome,
        "failures": [{"code": "weather_infeasible"}] if outcome == "ineligible" else [],
        "route_validation": {
            "required_leg_count": 1,
            "road_validated_leg_count": 1,
            "missing_leg_count": 0,
            "fallback_leg_count": 0,
        },
    }
    certificate["content_hash"] = core._artifact_content_hash(certificate)
    diff = {"parent_plan_id": "parent", "child_plan_id": "child", "diff_id": "diff"}
    staged = {
        "operation_id": "operation",
        "type": "set_stop_role",
        "target": "target",
        "parameters": {"role": "meal"},
        "created_at": "stamp",
    }
    proposal = {
        "execution_mode": "typed_direct_edit_independent_evaluation",
        "state": outcome,
        "eligibility": outcome,
        "parent_plan_id": "parent",
        "parent_plan_content_hash": "parenthash",
        "child_plan_id": "child",
        "child_plan_content_hash": "childhash",
        "compiled_request": {
            "source_request_id": "request",
            "operations": [{key: value for key, value in staged.items() if key != "created_at"}],
        },
        "evaluation_certificate": certificate,
        "plan_diff": diff,
        "geography_plan": geo,
        "certificate_id": "certificate",
        "certificate_content_hash": certificate["content_hash"],
        "diff_id": "diff",
        "diff_content_hash": core._artifact_content_hash(diff),
        "acceptance_eligible": outcome == "eligible",
    }
    parent = {
        "active_plan_id": "parent",
        "active_plan_content_hash": "parenthash",
        "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "parentgeo",
        "rendered_marker_ids": [],
        "preview_state": "PARENT_RENDERED",
    }
    rendered = {
        "active_plan_id": "child",
        "active_plan_content_hash": "childhash",
        "rendered_map_plan_id": "child",
        "rendered_geometry_hash": core._completed_race_geometry_hash(geo),
        "rendered_marker_ids": [],
        "preview_state": "PREVIEW_RENDERED",
    }
    if outcome == "ineligible":
        rendered = {**parent, "preview_state": "PREVIEW_INELIGIBLE"}
    trace = {
        "parent": parent,
        "parent_plan_id": "parent",
        "parent_plan_content_hash": "parenthash",
        "child_plan_id": "child",
        "child_plan_content_hash": "childhash",
        "request_id": "request",
        "evaluation_id": "certificate",
        "comparison_eligibility": outcome,
        "failure_codes": [row["code"] for row in certificate["failures"]],
        "child_geometry_hash": core._completed_race_geometry_hash(geo),
        "rendered": rendered,
        "reloaded": deepcopy(rendered),
        "staged": deepcopy(parent),
        "restored": deepcopy(parent),
    }
    return proposal, staged, trace


@pytest.mark.parametrize("outcome", ["eligible", "ineligible"])
def test_eligible_child_or_weather_blocked_parent_are_distinct_valid_states(outcome):
    proposal, staged, trace = _fixture(outcome)
    verifier.artifact_chain(proposal, staged, trace, outcome)


@pytest.mark.parametrize(
    "fault",
    [
        "request",
        "operation",
        "parent",
        "certificate_hash",
        "diff_hash",
        "fallback",
        "geometry_hash",
        "ineligible_child_shown",
        "wrong_ineligibility_reason",
        "eligible_parent_shown",
        "staged_child",
        "reload_changed",
        "undo_not_parent",
    ],
)
def test_schema_valid_but_disconnected_artifacts_are_rejected(fault):
    outcome = "ineligible" if fault in {"ineligible_child_shown", "wrong_ineligibility_reason"} else "eligible"
    proposal, staged, trace = _fixture(outcome)
    if fault == "request":
        trace["request_id"] = "other"
    elif fault == "operation":
        proposal["compiled_request"]["operations"][0]["type"] = "mark_flexible"
    elif fault == "parent":
        trace["parent_plan_content_hash"] = "other"
    elif fault == "certificate_hash":
        proposal["certificate_content_hash"] = "other"
    elif fault == "diff_hash":
        proposal["diff_content_hash"] = "other"
    elif fault == "fallback":
        proposal["evaluation_certificate"]["route_validation"]["fallback_leg_count"] = 1
    elif fault == "geometry_hash":
        trace["child_geometry_hash"] = "other"
    elif fault == "ineligible_child_shown":
        trace["rendered"]["active_plan_id"] = "child"
    elif fault == "wrong_ineligibility_reason":
        certificate = proposal["evaluation_certificate"]
        certificate["failures"] = [{"code": "route_missing"}]
        certificate["content_hash"] = proposal["certificate_content_hash"] = core._artifact_content_hash(certificate)
        trace["failure_codes"] = ["route_missing"]
    elif fault == "eligible_parent_shown":
        trace["rendered"] = deepcopy(trace["parent"])
    elif fault == "staged_child":
        trace["staged"] = deepcopy(trace["rendered"])
    elif fault == "reload_changed":
        trace["reloaded"]["rendered_geometry_hash"] = "other"
    else:
        trace["restored"]["preview_state"] = "PREVIEW_RENDERED"
    with pytest.raises(core.E2E1VerificationError):
        verifier.artifact_chain(proposal, staged, trace, outcome)


def _window_fixture():
    stop = {"stop_id": "stearns_wharf", "day": 6, "stop_order": 1}
    parent = {"stops": {"features": [{"properties": stop}]}}
    source_trace = {"stop_id": "stearns_wharf", "departure_minute": 634}
    earliest = {
        "child_plan_id": "earlier_child",
        "certificate_id": "earlier_evaluation",
        "schedule_impact": {"child": {"stop_traces": [source_trace]}},
    }
    evidence = {
        "parent_stop": stop,
        "latest_bound_source": {
            "case": "time_window_earliest_only",
            "child_plan_id": "earlier_child",
            "evaluation_id": "earlier_evaluation",
            "trace": source_trace,
            "chosen_latest_departure": "11:30",
        },
    }
    operation = {
        "type": "set_stop_time_window",
        "target": "stearns_wharf",
        "source": "map",
        "parameters": {"earliest_arrival": None, "latest_departure": "11:30"},
    }
    proposal = {
        "geography_plan": parent,
        "plan_diff": {
            "time_window_changes": [
                {
                    "stop_id": "stearns_wharf",
                    "from_earliest_arrival": None,
                    "from_latest_departure": None,
                    "to_earliest_arrival": None,
                    "to_latest_departure": "11:30",
                }
            ]
        },
        "schedule_impact": {
            "child": {
                "stop_traces": [
                    {"stop_id": "stearns_wharf", "departure_minute": 634, "latest_departure_status": "satisfied"}
                ]
            }
        },
    }
    return operation, evidence, proposal, parent, earliest


def test_latest_bound_is_derived_from_exact_prior_evaluation_not_an_invented_time():
    verifier.semantics("time_window", "latest_only", "stearns_wharf", *_window_fixture())


@pytest.mark.parametrize("fault", ["wrong_source_child", "guessed_bound", "late_departure", "wrong_target"])
def test_time_window_plausibility_cannot_replace_evaluator_evidence(fault):
    operation, evidence, proposal, parent, earliest = _window_fixture()
    if fault == "wrong_source_child":
        evidence["latest_bound_source"]["child_plan_id"] = "other"
    elif fault == "guessed_bound":
        operation["parameters"]["latest_departure"] = "12:00"
    elif fault == "late_departure":
        proposal["schedule_impact"]["child"]["stop_traces"][0]["departure_minute"] = 700
    else:
        operation["target"] = "other"
    with pytest.raises(core.E2E1VerificationError):
        verifier.semantics(
            "time_window", "latest_only", "stearns_wharf", operation, evidence, proposal, parent, earliest
        )


def test_missing_evaluated_variant_evidence_rejects_stably(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_variants_evidence_malformed"):
        verifier.verify(tmp_path, "http://127.0.0.1:8127")
