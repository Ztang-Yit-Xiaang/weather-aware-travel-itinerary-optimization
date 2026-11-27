"""Portable adversarial receipt tests; these do not certify a browser run."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from verify_e2e1_owned_edits import editor_contract
from verify_e2e1_vertical_slice import E2E1VerificationError


@pytest.mark.parametrize("family", ["duration", "time_window"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "disabled",
        "hidden",
        "untrusted_submit",
        "untrusted_change",
        "wrong_value",
        "wrong_target",
        "wrong_parameters",
        "missing_preview",
        "untrusted_preview",
        "foreign_fields",
        "foreign_event_log",
        "foreign_request",
        "wrong_source",
    ],
)
def test_editor_receipt_rejects_forged_or_disconnected_control_input(tmp_path, family, fault):
    operation = "set_stop_" + family
    name = "duration-preferred" if family == "duration" else "earliest_arrival"
    text = "90" if family == "duration" else "10:00"
    fields = {name: text}
    control = {"form_id": 1, "operation": operation, "fields": fields, "visible": True, "disabled": False}
    request = {
        "type": operation,
        "source": "map",
        "target": "griffith_observatory" if family == "duration" else "golden_gate_bridge",
        "parameters": {
            "duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}
        }
        if family == "duration"
        else {"earliest_arrival": "10:00", "latest_departure": None},
    }
    events = [
        {"type": "change", "trusted": True, "operation": operation, "name": name, "value": text, "form_id": 1},
        {"type": "submit", "trusted": True, "operation": operation, "form_id": 1, "fields": fields},
        {"type": "click", "trusted": True, "id": "inline-preview", "form_id": None, "operation": None},
    ]
    stage = {"input": request, "output": {"synthetic_unit_response": True}}
    value = {
        "control": control,
        "context": "accepted_child",
        "phase": "stage",
        "request": deepcopy(request),
        "response": deepcopy(stage["output"]),
        "event_start": 1,
        "event_end": 2,
        "raw_events": deepcopy(events[:2]),
    }
    if fault in {"disabled", "hidden"}:
        control["disabled" if fault == "disabled" else "visible"] = fault == "disabled"
    elif fault == "untrusted_submit":
        events[1]["trusted"] = False
    elif fault == "untrusted_change":
        events[0]["trusted"] = False
    elif fault == "wrong_value":
        events[0]["value"] = "other"
    elif fault in {"wrong_target", "wrong_parameters"}:
        request["target" if fault == "wrong_target" else "parameters"] = "other"
        value["request"] = deepcopy(request)
    elif fault == "missing_preview":
        events.pop()
    elif fault == "untrusted_preview":
        events[2]["trusted"] = False
    elif fault == "foreign_fields":
        events[1]["fields"] = {name: "other"}
    elif fault == "foreign_request":
        value["request"] = {"type": "unrelated"}
    elif fault == "wrong_source":
        request["source"] = "copilot"
        value["request"] = deepcopy(request)
    if fault != "foreign_event_log":
        value["raw_events"] = deepcopy(events[:2])
    else:
        value["raw_events"] = []
    result = {
        "scope": "synthetic_unit_only_not_browser_proof",
        "family": family,
        "fault": fault,
        "input": {"receipt": value, "events": events, "stage": stage},
    }
    if fault is None:
        editor_contract(value, events, stage, family)
        result["output"] = {"accepted": True}
    else:
        with pytest.raises(E2E1VerificationError) as rejected:
            editor_contract(value, events, stage, family)
        result["output"] = {"accepted": False, "reason": str(rejected.value)}
    (tmp_path / "exact_unit_io.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
