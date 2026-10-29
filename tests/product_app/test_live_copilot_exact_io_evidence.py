from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from test_live_copilot_prompt_evaluation import _live_records

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify_live_copilot_prompt_evidence_v4.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("exact_io_verifier_v4", VERIFIER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_document(verifier, path: Path, document: dict) -> None:
    unsigned = dict(document)
    unsigned.pop("content_hash")
    document["content_hash"] = verifier.canonical_sha256(unsigned)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def current_fixture(verifier, tmp_path: Path) -> Path:
    """Synthetic verifier input, not rebadged historical paid-call evidence.

    The old fixed-24 contract intentionally uses V1 context/schema. Build its
    complete artifact chain from the current prompt/source, then verify it
    before adversarial edits. No API client or provider transport is created.
    """
    directory = (tmp_path / "evidence" if tmp_path.is_relative_to(ROOT)
        else ROOT / "tmp" / f"exact_io_verifier_fixture_{uuid4().hex}")
    directory.mkdir(parents=True)
    (tmp_path / "fixture_location.json").write_text(json.dumps({
        "path": str(directory), "synthetic_fixture_only": True, "actual_provider_calls": 0,
    }), encoding="utf-8")
    runner = verifier.runner
    source = runner.build_source_snapshot()
    corpus_path = verifier.DEFAULT_CORPUS.resolve()
    corpus = verifier.load_corpus(corpus_path)
    cases = {row["case_id"]: row for row in corpus["cases"]}
    summaries, exact_hashes = {}, {}
    for effort in ("low", "medium"):
        controls = runner.base.build_request_controls(model="gpt-5.6-terra", effort=effort, timeout_seconds=30)
        records = _live_records(runner.base, corpus, effort=effort)
        transport = []
        for sequence, record in enumerate(records, 1):
            record["source_snapshot_sha256"] = source["content_hash"]
            case = cases[record["case_id"]]
            context = runner.base._context(corpus["context_fixtures"][case["context_fixture"]])
            prompt = verifier.build(context, (), case["input"], reasoning_effort=effort)
            schema = verifier.type_to_response_format_param(verifier.OpenAIInterpretationSchemaV1)
            request = {"model": "gpt-5.6-terra", "input": prompt.input, "instructions": prompt.instructions,
                "max_output_tokens": prompt.max_output_tokens,
                "reasoning": {"effort": effort, "context": prompt.reasoning_context},
                "store": False, "tools": [],
                "text": {"verbosity": prompt.text_verbosity, "format": {"type": schema["type"], **schema["json_schema"]}}}
            parsed = {"schema_version": "copilot-interpretation-v1", "state": record["state"],
                "assistant_message": "Synthetic verifier fixture; no provider call occurred.",
                "intents": record["typed_intents"], "highlights": record["highlights"], "permission": record["permission"]}
            canonical = verifier._canonicalize_provider_copy(
                verifier._to_dataclass(verifier.OpenAIInterpretationSchemaV1.model_validate(parsed, strict=True)),
                context=context, message=case["input"])
            assert canonical.state == record["state"] and canonical.permission == record["permission"]
            assert [intent.type for intent in canonical.intents] == record["intents"]
            assert runner.base.referenced_ids(canonical) == record["referenced_ids"]
            # Local clarification copy is authoritative, not provider prose or
            # the older fixture helper's wording. Operation semantics stay exact.
            record["typed_intents"] = [intent.as_dict() for intent in canonical.intents]
            response = {"id": f'{effort}:{record["case_id"]}', "status": "completed",
                "model": record["provider_model"], "created_at": record["provider_created_at"],
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(parsed)}]}]}
            request_text = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
            response_text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            timestamp = datetime.fromtimestamp(record["provider_created_at"], UTC).isoformat()
            transport.append({"schema_version": "copilot-live-exact-transport-attempt-v1", "sequence": sequence,
                "effort": effort, "case_id": record["case_id"], "case_attempt": 1,
                "started_at": timestamp, "completed_at": timestamp, "latency_ms": 0,
                "method": "POST", "url": "https://api.openai.com/v1/responses",
                "request_body_utf8": request_text, "request_body_sha256": verifier.hash_text(request_text),
                "response_status_code": 200, "response_body_utf8": response_text,
                "response_body_sha256": verifier.hash_text(response_text), "transport_error_code": None,
                "request_headers_persisted": False, "response_headers_persisted": False, "authorization_persisted": False})
        cases_path = directory / f"{effort}_cases.json"
        cases_document = runner.base._case_evidence_document(effort=effort, model="gpt-5.6-terra",
            corpus_path=corpus_path, source_snapshot=source, request_controls=controls, records=records)
        runner.base.write_summary(cases_path, cases_document)
        exact_path = directory / f"{effort}_exact_llm_io.json"
        runner.base.write_summary(exact_path, runner.exact_io_document(effort=effort, model="gpt-5.6-terra",
            source_snapshot=source, request_controls=controls, records=transport))
        # Exercise each unmodified boundary, including current prompt identity.
        summary = verifier.verify_case_document(path=cases_path, corpus_path=corpus_path, corpus=corpus,
            effort=effort, source_builder=runner.build_source_snapshot)
        exact = verifier.validate_exact_io(path=exact_path, effort=effort, corpus=corpus, case_document=cases_document)
        summary.update({"exact_llm_io_persisted": True, "transport_attempts_total": exact["attempts"],
            "exact_llm_io_path": exact["path"], "exact_llm_io_sha256": exact["sha256"]})
        runner.base.write_summary(directory / f"{effort}_summary.json", summary)
        summaries[effort], exact_hashes[effort] = summary, exact["sha256"]
    comparison = {"schema_version": "copilot-reasoning-comparison-v3", "model": "gpt-5.6-terra",
        "prompt_version": summaries["low"]["prompt_version"], "prompt_sha256": summaries["low"]["prompt_sha256"],
        "evaluation_scope": "live_fixed_24_exact_io", "reasoning_efforts": ["low", "medium"],
        "recommended_reasoning_effort": verifier.choose_effort(summaries),
        "all_thresholds_passed": all(row["thresholds_passed"] for row in summaries.values()),
        "provider_attempts_total": sum(row["provider_attempts_total"] for row in summaries.values()),
        "transport_attempts_total": sum(row["transport_attempts_total"] for row in summaries.values()),
        "case_evidence_sha256": {effort: row["case_evidence_sha256"] for effort, row in summaries.items()},
        "exact_llm_io_sha256": exact_hashes, "source_snapshot_sha256": source["content_hash"],
        "raw_provider_content_persisted": False, "exact_llm_io_persisted": True, "summaries": summaries}
    runner.base.write_summary(directory / "reasoning_comparison.json", comparison)
    manifest = runner.build_file_io_manifest(output_dir=directory, source_snapshot=source,
        output_paths=list(directory.iterdir()))
    runner.base.write_summary(directory / "file_io_manifest.json", manifest)
    baseline = verifier.verify_evidence_directory(directory)
    assert baseline["verified"] is True and baseline["logical_cases"] == 48
    runner.base.write_summary(tmp_path / "baseline_verification.json", {"synthetic_fixture_only": True,
        "actual_provider_calls": 0, "verifier_result": baseline})
    return directory


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("request_input", "exact_request_semantics_mismatch"),
        ("response_projection", "exact_response_projection_mismatch"),
        ("transport_count", "exact_transport_case_universe_mismatch"),
    ],
)
def test_exact_io_verifier_rejects_coherently_rehashed_body_tampering(
    tmp_path: Path, mutation: str, error: str
) -> None:
    verifier = load_verifier()
    copied = current_fixture(verifier, tmp_path)
    path = copied / "low_exact_llm_io.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    if mutation == "request_input":
        record = document["records"][0]
        request = json.loads(record["request_body_utf8"])
        request["input"] += "\nforged"
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        record["request_body_utf8"] = body
        record["request_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    elif mutation == "response_projection":
        record = document["records"][0]
        response = json.loads(record["response_body_utf8"])
        output = next(
            content
            for item in response["output"]
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        parsed = json.loads(output["text"])
        parsed["state"] = "refused"
        output["text"] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        body = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        record["response_body_utf8"] = body
        record["response_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    elif mutation == "transport_count":
        document["records"].pop()
    else:  # pragma: no cover
        raise AssertionError(mutation)

    write_document(verifier, path, document)
    with pytest.raises(ValueError, match=f"^{error}$") as rejected:
        verifier.verify_evidence_directory(copied)
    (tmp_path / "mutation_assertions.json").write_text(json.dumps({
        "mutation": mutation, "expected_error": error, "actual_error": str(rejected.value),
        "baseline_verified": True, "synthetic_fixture_only": True, "actual_provider_calls": 0,
        "mutated_exact_io_path": str(path),
    }, indent=2), encoding="utf-8")

