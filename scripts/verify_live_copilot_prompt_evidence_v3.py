"""Verify exact synthetic LLM I/O, transport attempts, and file provenance.

This verifier performs no provider calls. It checks the original aggregate and
case contracts, then independently binds each exact HTTP request/response body
to its synthetic corpus case and typed case evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_live_copilot_prompt_evaluation_v2 as runner  # noqa: E402
import verify_live_copilot_prompt_evidence_v2 as v2  # noqa: E402
from run_live_copilot_prompt_evaluation import (  # noqa: E402
    _file_sha256,
    choose_effort,
    load_corpus,
)
from run_live_copilot_prompt_evaluation_v2 import (  # noqa: E402
    DEFAULT_CORPUS,
    build_source_snapshot,
    canonical_sha256,
)

from itinerary_system.product_app.copilot_prompt import build  # noqa: E402
from itinerary_system.product_app.openai_copilot import (  # noqa: E402
    OpenAIInterpretationSchemaV1,
)

_EXACT_IO_FIELDS = {
    "schema_version",
    "generated_at",
    "model",
    "reasoning_effort",
    "prompt_sha256",
    "source_snapshot_sha256",
    "request_controls_sha256",
    "scope",
    "exact_request_and_response_bodies_persisted",
    "request_headers_persisted",
    "response_headers_persisted",
    "authorization_persisted",
    "api_key_persisted",
    "records",
    "content_hash",
}
_TRANSPORT_FIELDS = {
    "schema_version",
    "sequence",
    "effort",
    "case_id",
    "case_attempt",
    "started_at",
    "completed_at",
    "latency_ms",
    "method",
    "url",
    "request_body_utf8",
    "request_body_sha256",
    "response_status_code",
    "response_body_utf8",
    "response_body_sha256",
    "transport_error_code",
    "request_headers_persisted",
    "response_headers_persisted",
    "authorization_persisted",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_json_document")
    return value


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_aware(value: Any, code: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(code)
    return parsed


def output_text(response: dict[str, Any]) -> str:
    values: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if (
                isinstance(content, dict)
                and content.get("type") == "output_text"
                and isinstance(content.get("text"), str)
            ):
                values.append(content["text"])
    if len(values) != 1:
        raise ValueError("exact_response_output_text_invalid")
    return values[0]


def validate_exact_io(
    *,
    path: Path,
    effort: str,
    corpus: dict[str, Any],
    case_document: dict[str, Any],
) -> dict[str, Any]:
    document = read_json(path)
    if set(document) != _EXACT_IO_FIELDS:
        raise ValueError("exact_io_fields_invalid")
    if document["schema_version"] != "copilot-live-exact-llm-io-v1":
        raise ValueError("exact_io_schema_invalid")
    unsigned = dict(document)
    content_hash = unsigned.pop("content_hash")
    if content_hash != canonical_sha256(unsigned):
        raise ValueError("exact_io_hash_mismatch")
    if (
        document["reasoning_effort"] != effort
        or document["model"] != case_document["model"]
        or document["prompt_sha256"] != case_document["prompt_sha256"]
        or document["source_snapshot_sha256"]
        != case_document["source_snapshot"]["content_hash"]
        or document["request_controls_sha256"]
        != case_document["request_controls"]["content_hash"]
    ):
        raise ValueError("exact_io_identity_mismatch")
    if (
        document["scope"] != "synthetic_fixed_24_only"
        or document["exact_request_and_response_bodies_persisted"] is not True
        or document["request_headers_persisted"] is not False
        or document["response_headers_persisted"] is not False
        or document["authorization_persisted"] is not False
        or document["api_key_persisted"] is not False
    ):
        raise ValueError("exact_io_boundary_invalid")
    parse_aware(document["generated_at"], "exact_io_generated_at_invalid")
    records = document["records"]
    if not isinstance(records, list):
        raise ValueError("exact_io_records_invalid")
    cases = {case["case_id"]: case for case in corpus["cases"]}
    fixtures = corpus["context_fixtures"]
    evidence_records = {
        record["case_id"]: record for record in case_document["records"]
    }
    counts: dict[str, int] = {}
    previous_completed: datetime | None = None
    for sequence, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != _TRANSPORT_FIELDS:
            raise ValueError("exact_transport_fields_invalid")
        if (
            record["schema_version"] != "copilot-live-exact-transport-attempt-v1"
            or record["sequence"] != sequence
            or record["effort"] != effort
            or record["method"] != "POST"
            or record["url"] != "https://api.openai.com/v1/responses"
            or record["request_headers_persisted"] is not False
            or record["response_headers_persisted"] is not False
            or record["authorization_persisted"] is not False
        ):
            raise ValueError("exact_transport_identity_invalid")
        case_id = record["case_id"]
        if case_id not in evidence_records or case_id not in cases:
            raise ValueError("exact_transport_case_invalid")
        counts[case_id] = counts.get(case_id, 0) + 1
        if record["case_attempt"] != counts[case_id]:
            raise ValueError("exact_transport_attempt_order_invalid")
        started = parse_aware(record["started_at"], "exact_transport_time_invalid")
        completed = parse_aware(record["completed_at"], "exact_transport_time_invalid")
        if completed < started or (
            previous_completed is not None and started < previous_completed
        ):
            raise ValueError("exact_transport_time_order_invalid")
        previous_completed = completed
        if (
            not isinstance(record["latency_ms"], (int, float))
            or isinstance(record["latency_ms"], bool)
            or record["latency_ms"] < 0
        ):
            raise ValueError("exact_transport_latency_invalid")
        request_text = record["request_body_utf8"]
        response_text = record["response_body_utf8"]
        if not isinstance(request_text, str) or hash_text(request_text) != record[
            "request_body_sha256"
        ]:
            raise ValueError("exact_request_body_hash_mismatch")
        if (
            record["response_status_code"] != 200
            or record["transport_error_code"] is not None
            or not isinstance(response_text, str)
            or hash_text(response_text) != record["response_body_sha256"]
        ):
            raise ValueError("exact_response_body_invalid")
        request = json.loads(request_text)
        response = json.loads(response_text)
        case = cases[case_id]
        bundle = build(
            runner.base._context(fixtures[case["context_fixture"]]),
            (),
            case["input"],
            reasoning_effort=effort,
        )
        if (
            request.get("model") != document["model"]
            or request.get("instructions") != bundle.instructions
            or request.get("input") != bundle.input
            or request.get("store") is not False
            or request.get("tools") != []
            or request.get("max_output_tokens") != bundle.max_output_tokens
            or request.get("reasoning")
            != {"effort": effort, "context": bundle.reasoning_context}
            or not isinstance(request.get("text"), dict)
            or request["text"].get("verbosity") != bundle.text_verbosity
        ):
            raise ValueError("exact_request_semantics_mismatch")
        evidence = evidence_records[case_id]
        response_id = response.get("id")
        if (
            response.get("status") != "completed"
            or not isinstance(response_id, str)
            or not response_id
            or hash_text(response_id) != evidence["provider_response_id_sha256"]
            or response.get("created_at") != evidence["provider_created_at"]
            or response.get("model") != evidence["provider_model"]
        ):
            raise ValueError("exact_response_identity_mismatch")
        try:
            parsed_json = json.loads(output_text(response))
            parsed = OpenAIInterpretationSchemaV1.model_validate(
                parsed_json, strict=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("exact_response_schema_invalid") from exc
        parsed_intents = [intent.model_dump(mode="json") for intent in parsed.intents]
        if (
            parsed.state != evidence["state"]
            or [intent["type"] for intent in parsed_intents] != evidence["intents"]
            or parsed_intents != evidence["typed_intents"]
            or parsed.permission != evidence["permission"]
            or parsed.highlights.model_dump(mode="json") != evidence["highlights"]
        ):
            raise ValueError("exact_response_projection_mismatch")
    if set(counts) != set(evidence_records):
        raise ValueError("exact_transport_case_universe_mismatch")
    for case_id, evidence in evidence_records.items():
        if counts[case_id] != evidence["attempt_count"]:
            raise ValueError("exact_transport_attempt_count_mismatch")
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
        "attempts": len(records),
    }


def verify_file_manifest(output_dir: Path, source_snapshot: dict[str, Any]) -> None:
    path = output_dir / "file_io_manifest.json"
    document = read_json(path)
    if document.get("schema_version") != "copilot-live-file-io-manifest-v1":
        raise ValueError("file_io_manifest_schema_invalid")
    unsigned = dict(document)
    content_hash = unsigned.pop("content_hash", None)
    if content_hash != canonical_sha256(unsigned):
        raise ValueError("file_io_manifest_hash_mismatch")
    if (
        document.get("secret_input_files_omitted") is not True
        or document.get("api_key_value_persisted") is not False
    ):
        raise ValueError("file_io_secret_boundary_invalid")
    expected_inputs = {
        row["path"]: (row["sha256"], row["size_bytes"])
        for row in source_snapshot["files"]
    }
    observed_inputs = {
        row.get("path"): (row.get("sha256"), row.get("size_bytes"))
        for row in document.get("input_files", [])
        if isinstance(row, dict)
        and row.get("exact_content_location") == row.get("path")
    }
    if observed_inputs != expected_inputs:
        raise ValueError("file_io_inputs_mismatch")
    outputs = document.get("output_files")
    if not isinstance(outputs, list):
        raise ValueError("file_io_outputs_invalid")
    observed_paths: set[str] = set()
    for row in outputs:
        if (
            not isinstance(row, dict)
            or row.get("path") != row.get("exact_content_location")
            or row.get("path") in observed_paths
        ):
            raise ValueError("file_io_output_row_invalid")
        observed_paths.add(row["path"])
        output_path = REPOSITORY_ROOT / row["path"]
        if (
            not output_path.is_file()
            or _file_sha256(output_path) != row.get("sha256")
            or output_path.stat().st_size != row.get("size_bytes")
        ):
            raise ValueError("file_io_output_binding_mismatch")
    actual = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "file_io_manifest.json"
    }
    if observed_paths != actual:
        raise ValueError("file_io_output_universe_mismatch")


def verify_evidence_directory(
    output_dir: Path, corpus_path: Path = DEFAULT_CORPUS
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    corpus_path = corpus_path.resolve()
    corpus = load_corpus(corpus_path)
    old_builder = v2.build_source_snapshot
    v2.build_source_snapshot = build_source_snapshot
    try:
        summaries: dict[str, dict[str, Any]] = {}
        exact: dict[str, dict[str, Any]] = {}
        for effort in ("low", "medium"):
            case_path = output_dir / f"{effort}_cases.json"
            recomputed = v2._verify_case_document(
                path=case_path,
                corpus_path=corpus_path,
                corpus=corpus,
                effort=effort,
            )
            case_document = read_json(case_path)
            exact_result = validate_exact_io(
                path=output_dir / f"{effort}_exact_llm_io.json",
                effort=effort,
                corpus=corpus,
                case_document=case_document,
            )
            recomputed.update(
                {
                    "exact_llm_io_persisted": True,
                    "transport_attempts_total": exact_result["attempts"],
                    "exact_llm_io_path": exact_result["path"],
                    "exact_llm_io_sha256": exact_result["sha256"],
                }
            )
            if read_json(output_dir / f"{effort}_summary.json") != recomputed:
                raise ValueError(f"{effort}_summary_mismatch")
            summaries[effort] = recomputed
            exact[effort] = exact_result
    finally:
        v2.build_source_snapshot = old_builder
    recommended = choose_effort(summaries)
    comparison = {
        "schema_version": "copilot-reasoning-comparison-v3",
        "model": summaries["low"]["model"],
        "prompt_version": summaries["low"]["prompt_version"],
        "prompt_sha256": summaries["low"]["prompt_sha256"],
        "evaluation_scope": "live_fixed_24_exact_io",
        "reasoning_efforts": ["low", "medium"],
        "recommended_reasoning_effort": recommended,
        "all_thresholds_passed": all(
            summary["thresholds_passed"] for summary in summaries.values()
        ),
        "provider_attempts_total": sum(
            summary["provider_attempts_total"] for summary in summaries.values()
        ),
        "transport_attempts_total": sum(
            summary["transport_attempts_total"] for summary in summaries.values()
        ),
        "case_evidence_sha256": {
            effort: summary["case_evidence_sha256"]
            for effort, summary in summaries.items()
        },
        "exact_llm_io_sha256": {
            effort: exact[effort]["sha256"] for effort in ("low", "medium")
        },
        "source_snapshot_sha256": summaries["low"]["source_snapshot_sha256"],
        "raw_provider_content_persisted": False,
        "exact_llm_io_persisted": True,
        "summaries": summaries,
    }
    if read_json(output_dir / "reasoning_comparison.json") != comparison:
        raise ValueError("reasoning_comparison_mismatch")
    source_snapshot = read_json(output_dir / "low_cases.json")["source_snapshot"]
    if source_snapshot != build_source_snapshot():
        raise ValueError("source_snapshot_mismatch")
    verify_file_manifest(output_dir, source_snapshot)
    return {
        "schema_version": "copilot-live-exact-io-verification-v1",
        "verified": True,
        "logical_cases": sum(summary["total_cases"] for summary in summaries.values()),
        "provider_attempts_total": comparison["provider_attempts_total"],
        "transport_attempts_total": comparison["transport_attempts_total"],
        "recommended_reasoning_effort": recommended,
        "exact_llm_io_sha256": comparison["exact_llm_io_sha256"],
        "source_snapshot_sha256": comparison["source_snapshot_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            verify_evidence_directory(args.output_dir, args.corpus),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

