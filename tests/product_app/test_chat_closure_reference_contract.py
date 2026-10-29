"""Adversarial tests of the browser observer's full accepted-reference check."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "browser"))
from test_chat_runtime_closure_live import _assert_accepted_reference, _hash, _write_json  # noqa: E402


def _seal(record):
    body = {key: value for key, value in record.items() if key != "content_hash"}
    record["content_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _inputs(accepted_child):
    original = {"plan_id": "plan_parent", "sequence": ["bixby", "golden_gate"]}
    original["content_hash"] = _hash(original)
    accepted = (
        {"plan_id": "plan_child", "sequence": ["pfeiffer", "golden_gate"]} if accepted_child else deepcopy(original)
    )
    accepted["content_hash"] = _hash(accepted)
    initial = {
        "active_plan_id": original["plan_id"],
        "active_plan_content_hash": original["content_hash"],
        "rendered_map_plan_id": original["plan_id"],
        "rendered_marker_ids": original["sequence"],
    }
    parent = {
        "active_plan_id": accepted["plan_id"],
        "active_plan_content_hash": accepted["content_hash"],
        "rendered_map_plan_id": accepted["plan_id"],
        "rendered_marker_ids": accepted["sequence"],
    }
    pointer = {
        "schema_version": "workspace-pointer-v1",
        "trip_id": "trip",
        "run_id": "run",
        "original_plan_id": original["plan_id"],
        "original_plan_content_hash": original["content_hash"],
        "accepted_plan_id": accepted["plan_id"],
        "accepted_plan_content_hash": accepted["content_hash"],
        "decision_id": "decision_" + "a" * 32 if accepted_child else None,
        "revision": int(accepted_child),
        "updated_at": "2026-09-08T12:00:00+00:00",
    }
    _seal(pointer)
    public = {
        key: pointer[key]
        for key in ("schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision")
    }
    decision = (
        {
            "decision_id": pointer["decision_id"],
            "pointer_revision_after": 1,
            "trip_id": "trip",
            "run_id": "run",
            "selected_plan_id": accepted["plan_id"],
            "selected_plan_content_hash": accepted["content_hash"],
        }
        if accepted_child
        else None
    )
    if decision:
        _seal(decision)
    return {
        "pointer": pointer,
        "public": public,
        "session": {
            "session_id": "session",
            "trip_id": "trip",
            "run_id": "run",
            "accepted_plan_id": accepted["plan_id"],
        },
        "accepted": accepted,
        "original": original,
        "parent": parent,
        "initial": initial,
        "decision": decision,
    }


@pytest.mark.parametrize("accepted_child", [False, True])
def test_full_reference_retains_extra_canonical_fields(tmp_path, accepted_child):
    inputs = _inputs(accepted_child)
    assert len(inputs["pointer"]) == 11 and len(inputs["public"]) == 5
    output = _assert_accepted_reference(**inputs)
    assert output["durable_pointer"] == inputs["pointer"]
    assert output["public_pointer"] == inputs["public"]
    _write_json(tmp_path / "reference_exact_io.json", {"input": inputs, "output": output, "provider_calls": 0})


@pytest.mark.parametrize(
    "fault",
    [
        "public_extra",
        "public_hash",
        "pointer_hash",
        "trip",
        "run",
        "original_id",
        "original_hash",
        "accepted_id",
        "accepted_hash",
        "missing_timestamp",
        "naive_timestamp",
        "boolean_revision",
        "decision_id",
        "missing_decision",
        "decision_hash",
        "decision_selected",
        "plan_tamper",
        "rendered_id",
        "marker_set",
    ],
)
def test_mismatched_or_coherently_rehashed_reference_is_rejected(tmp_path, fault):
    inputs = _inputs(True)
    _assert_accepted_reference(**inputs)
    if fault == "public_extra":
        inputs["public"]["not_canonical"] = True
    elif fault == "public_hash":
        inputs["public"]["accepted_plan_content_hash"] = "wrong"
    elif fault in {"trip", "run", "original_id", "original_hash", "accepted_id", "accepted_hash"}:
        field = {
            "trip": "trip_id",
            "run": "run_id",
            "original_id": "original_plan_id",
            "original_hash": "original_plan_content_hash",
            "accepted_id": "accepted_plan_id",
            "accepted_hash": "accepted_plan_content_hash",
        }[fault]
        inputs["pointer"][field] = "wrong"
        if field in inputs["public"]:
            inputs["public"][field] = "wrong"
    elif fault == "pointer_hash":
        inputs["pointer"]["content_hash"] = "wrong"
    elif fault == "missing_timestamp":
        del inputs["pointer"]["updated_at"]
    elif fault == "naive_timestamp":
        inputs["pointer"]["updated_at"] = "2026-09-08T12:00:00"
    elif fault == "boolean_revision":
        inputs["pointer"]["revision"] = inputs["public"]["revision"] = True
    elif fault == "decision_id":
        inputs["pointer"]["decision_id"] = inputs["public"]["decision_id"] = "decision_" + "b" * 32
    elif fault == "missing_decision":
        inputs["decision"] = None
    elif fault == "decision_hash":
        inputs["decision"]["content_hash"] = "wrong"
    elif fault == "decision_selected":
        inputs["decision"]["selected_plan_id"] = "wrong"
        _seal(inputs["decision"])
    elif fault == "plan_tamper":
        inputs["accepted"]["sequence"] = ["wrong"]
    elif fault == "rendered_id":
        inputs["parent"]["rendered_map_plan_id"] = "wrong"
    elif fault == "marker_set":
        inputs["parent"]["rendered_marker_ids"] = ["wrong"]
    if fault != "pointer_hash":
        _seal(inputs["pointer"])
    with pytest.raises(AssertionError):
        _assert_accepted_reference(**inputs)
    _write_json(
        tmp_path / "reference_exact_io.json",
        {"fault": fault, "input": inputs, "output": {"rejected": True}, "provider_calls": 0},
    )
