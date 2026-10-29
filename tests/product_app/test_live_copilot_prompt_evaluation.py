from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.models import (
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_live_copilot_prompt_evaluation.py"
VERIFIER = ROOT / "scripts" / "verify_live_copilot_prompt_evidence_v2.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner():
    return _load_module(RUNNER, "run_live_copilot_prompt_evaluation")


def _typed_case(case: dict) -> tuple[list[dict], dict[str, list], str | None]:
    references = list(case["expected_referenced_ids"])
    highlights = {"day_ids": [], "stop_ids": [], "segment_ids": [], "candidate_ids": []}
    typed_intents = []
    permission = None
    for intent_type in case["allowed_intents"]:
        values = {
            "type": intent_type,
            "target_stop_id": None,
            "destination_day": None,
            "candidate_id": None,
            "preference": None,
            "clarification_question": None,
        }
        strings = [value for value in references if isinstance(value, str)]
        integers = [value for value in references if type(value) is int]
        if intent_type == "move_day":
            values["target_stop_id"] = strings[0]
            values["destination_day"] = integers[0]
        elif intent_type == "keep_stop":
            values["target_stop_id"] = strings[0]
        elif intent_type == "replace_nearby":
            values["target_stop_id"] = strings[0]
            values["candidate_id"] = strings[1]
        elif intent_type == "route_feedback":
            values["preference"] = "reduce_driving"
        elif intent_type == "request_clarification":
            values["clarification_question"] = "Which itinerary item should I review?"
        elif intent_type == "external_action_not_supported":
            permission = "external_booking_change"
        typed_intents.append(values)
    if case["allowed_intents"] == ["review_registered_repair"]:
        highlights["day_ids"] = integers
        highlights["stop_ids"] = strings
    elif case["allowed_intents"] == ["route_feedback"]:
        highlights["segment_ids"] = strings
    return typed_intents, highlights, permission


def _live_records(runner, corpus, *, effort: str) -> list[dict[str, object]]:
    source_snapshot = runner.build_source_snapshot()
    controls = runner.build_request_controls(
        model="gpt-5.6-terra", effort=effort, timeout_seconds=30
    )
    cases = {case["case_id"]: case for case in corpus["cases"]}
    records = []
    for case_id in runner.LIVE_SUBSET_CASE_IDS:
        case = cases[case_id]
        intent_types = list(case["allowed_intents"])
        typed_intents, highlights, permission = _typed_case(case)
        records.append(
            {
                "case_evidence_schema": "copilot-live-case-evidence-record-v1",
                "case_id": case_id,
                "state": case["expected_state"],
                "intents": intent_types,
                "typed_intents": typed_intents,
                "permission": permission,
                "highlights": highlights,
                "referenced_ids": list(case["expected_referenced_ids"]),
                "structured_valid": True,
                "contract_valid": True,
                "automatic_action_observed": False,
                "latency_ms": 1.0,
                "input_tokens": 10,
                "output_tokens": 1 if effort == "low" else 2,
                "attempt_count": 1,
                "prompt_sha256": runner.prompt_sha256(),
                "provider_response_id_sha256": hashlib.sha256(
                    f"{effort}:{case_id}".encode()
                ).hexdigest(),
                "provider_created_at": time.time(),
                "provider_model": "gpt-5.6-terra",
                "request_controls_sha256": controls["content_hash"],
                "source_snapshot_sha256": source_snapshot["content_hash"],
                "safe_failure_code": None,
            }
        )
    return records


def _write_fake_evidence(tmp_path: Path):
    runner = _load_runner()
    verifier = _load_module(VERIFIER, "verify_live_copilot_prompt_evidence")
    corpus_path = runner.DEFAULT_CORPUS.resolve()
    corpus = runner.load_corpus(corpus_path)
    source_snapshot = runner.build_source_snapshot()
    summaries = {}
    evidence_hashes = {}
    for effort in ("low", "medium"):
        records = _live_records(runner, corpus, effort=effort)
        controls = runner.build_request_controls(
            model="gpt-5.6-terra", effort=effort, timeout_seconds=30
        )
        evidence = runner._case_evidence_document(
            effort=effort,
            model="gpt-5.6-terra",
            corpus_path=corpus_path,
            source_snapshot=source_snapshot,
            request_controls=controls,
            records=records,
        )
        evidence_path = tmp_path / f"{effort}_cases.json"
        runner.write_summary(evidence_path, evidence)
        summary = runner.evaluate_results(corpus, records, model="gpt-5.6-terra")
        summary.update(
            {
                "reasoning_effort": effort,
                "safe_failure_code_counts": {},
                "raw_provider_content_persisted": False,
                "request_controls_sha256": controls["content_hash"],
                "corpus_sha256": runner._file_sha256(corpus_path),
                "case_evidence_path": evidence_path.name,
                "case_evidence_sha256": runner._file_sha256(evidence_path),
            }
        )
        summaries[effort] = summary
        evidence_hashes[effort] = summary["case_evidence_sha256"]
        runner.write_summary(tmp_path / f"{effort}_summary.json", summary)
    recommended = runner.choose_effort(summaries)
    comparison = {
        "schema_version": "copilot-reasoning-comparison-v2",
        "model": "gpt-5.6-terra",
        "prompt_version": summaries["low"]["prompt_version"],
        "prompt_sha256": summaries["low"]["prompt_sha256"],
        "evaluation_scope": "live_fixed_24",
        "reasoning_efforts": ["low", "medium"],
        "recommended_reasoning_effort": recommended,
        "all_thresholds_passed": True,
        "provider_attempts_total": 48,
        "case_evidence_sha256": evidence_hashes,
        "source_snapshot_sha256": source_snapshot["content_hash"],
        "raw_provider_content_persisted": False,
        "summaries": summaries,
    }
    runner.write_summary(tmp_path / "reasoning_comparison.json", comparison)
    return runner, verifier, corpus, summaries


def test_referenced_ids_are_exact_deduplicated_typed_fields() -> None:
    runner = _load_runner()
    interpretation = CopilotInterpretationV1(
        state="proposal_ready",
        assistant_message="Local copy.",
        intents=(
            CopilotIntentV1(
                type="replace_nearby",
                target_stop_id="stop_a",
                candidate_id="candidate_b",
                destination_day=2,
            ),
        ),
        highlights=CopilotHighlightsV1(
            day_ids=(2,),
            stop_ids=("stop_a",),
            segment_ids=("segment_a_b",),
            candidate_ids=("candidate_b",),
        ),
    )

    assert runner.referenced_ids(interpretation) == [
        "stop_a",
        2,
        "candidate_b",
        "segment_a_b",
    ]


def test_reasoning_choice_requires_a_passing_effort_and_prefers_lower_cost() -> None:
    runner = _load_runner()
    low = {
        "thresholds_passed": True,
        "exact_state_allowed_intent_rate": 1.0,
        "input_tokens_total": 100,
        "output_tokens_total": 20,
        "latency_ms_total": 500.0,
    }
    medium = dict(low, input_tokens_total=110, latency_ms_total=450.0)

    assert runner.choose_effort({"low": low, "medium": medium}) == "low"
    assert runner.choose_effort({"low": dict(low, thresholds_passed=False)}) is None


def test_recording_response_counts_attempt_and_hashes_metadata() -> None:
    runner = _load_runner()

    class FakeResponses:
        async def parse(self, **_kwargs):
            return SimpleNamespace(
                id="resp_private_identifier",
                created_at=1_786_570_000,
                model="gpt-5.6-terra-2026-08-01",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
                output=[SimpleNamespace(type="function_call")],
            )

    recorder = runner._RecordingResponses(FakeResponses())
    asyncio.run(recorder.parse())

    assert recorder.attempt_count == 1
    assert recorder.input_tokens == 11
    assert recorder.output_tokens == 7
    assert recorder.provider_response_id_sha256 == hashlib.sha256(
        b"resp_private_identifier"
    ).hexdigest()
    assert recorder.automatic_action_observed is True


def test_live_evaluator_rejects_fabricated_or_inconsistent_provenance() -> None:
    runner = _load_runner()
    corpus = runner.load_corpus(runner.DEFAULT_CORPUS)
    records = _live_records(runner, corpus, effort="low")

    summary = runner.evaluate_results(corpus, records, model="gpt-5.6-terra")
    assert summary["provider_attempts_total"] == 24
    assert summary["unsafe_automatic_action_cases"] == 0

    forged = [dict(record) for record in records]
    forged[0]["attempt_count"] = 0
    with pytest.raises(ValueError, match="invalid_live_attempt_count"):
        runner.evaluate_results(corpus, forged, model="gpt-5.6-terra")

    forged = json.loads(json.dumps(records))
    move_record = next(record for record in forged if record["case_id"] == "move_01")
    move_record["typed_intents"][0]["target_stop_id"] = "stop_park"
    with pytest.raises(ValueError, match="live_reference_projection_mismatch"):
        runner.evaluate_results(corpus, forged, model="gpt-5.6-terra")

    forged = [dict(record) for record in records]
    forged[0]["source_snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="live_source_snapshot_mismatch"):
        runner.evaluate_results(corpus, forged, model="gpt-5.6-terra")

    aggregate_only = [
        {
            "case_id": record["case_id"],
            "state": record["state"],
            "intents": record["intents"],
            "referenced_ids": record["referenced_ids"],
            "structured_valid": True,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        for record in records
    ]
    with pytest.raises(ValueError, match="invalid_live_case_evidence_schema"):
        runner.evaluate_results(corpus, aggregate_only, model="gpt-5.6-terra")


def test_independent_reducer_recomputes_and_rejects_tampering(tmp_path: Path) -> None:
    _runner, verifier, _corpus, _summaries = _write_fake_evidence(tmp_path)

    result = verifier.verify_evidence_directory(tmp_path)
    assert result["verified"] is True
    assert result["provider_attempts_total"] == 48
    assert result["recommended_reasoning_effort"] == "low"

    summary_path = tmp_path / "low_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["structured_schema_validity_rate"] = 0.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="low_summary_mismatch"):
        verifier.verify_evidence_directory(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("record_control", "record_control_binding_mismatch"),
        ("document_timeout", "case_evidence_timeout_mismatch"),
        ("future_created_at", "record_provider_created_at_outside_run"),
        ("duplicate_response", "duplicate_provider_response_hash"),
        ("provider_model", "record_provider_model_mismatch"),
    ],
)
def test_independent_reducer_rejects_coherently_rehashed_provenance(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    runner, verifier, _corpus, _summaries = _write_fake_evidence(tmp_path)
    path = tmp_path / "low_cases.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    if mutation == "record_control":
        document["records"][0]["request_controls_sha256"] = "f" * 64
    elif mutation == "document_timeout":
        controls = dict(document["request_controls"])
        controls["timeout_seconds"] = 31.0
        controls.pop("content_hash")
        controls["content_hash"] = runner._canonical_sha256(controls)
        document["request_controls"] = controls
        for record in document["records"]:
            record["request_controls_sha256"] = controls["content_hash"]
    elif mutation == "future_created_at":
        document["records"][0]["provider_created_at"] = 9_999_999_999
    elif mutation == "duplicate_response":
        document["records"][1]["provider_response_id_sha256"] = document["records"][0][
            "provider_response_id_sha256"
        ]
    elif mutation == "provider_model":
        document["records"][0]["provider_model"] = "gpt-5.6-terra-forged"
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)

    unsigned = dict(document)
    unsigned.pop("content_hash")
    document["content_hash"] = runner._canonical_sha256(unsigned)
    runner.write_summary(path, document)

    with pytest.raises(ValueError, match=expected_error):
        verifier.verify_evidence_directory(tmp_path)
