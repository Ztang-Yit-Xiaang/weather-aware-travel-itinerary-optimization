"""Run the fixed-24 Copilot gate with exact synthetic LLM I/O evidence.

This version preserves every v1 aggregate and per-case field, and adds an
independent HTTP-transport witness containing exact UTF-8 request/response
bodies. Authorization headers, API keys, cookies, and raw exception messages
are never persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_live_copilot_prompt_evaluation as base  # noqa: E402
from run_product_app import load_local_environment  # noqa: E402

from itinerary_system.product_app.copilot import CopilotContractError  # noqa: E402
from itinerary_system.product_app.openai_copilot import (  # noqa: E402
    OpenAICopilotAdapter,
    OpenAICopilotError,
)

ReasoningEffort = Literal["low", "medium"]
DEFAULT_CORPUS = base.DEFAULT_CORPUS
SOURCE_INPUTS = tuple(
    dict.fromkeys(
        (
            *base.SOURCE_INPUTS,
            "scripts/verify_live_copilot_prompt_evidence_v2.py",
            "scripts/run_live_copilot_prompt_evaluation_v2.py",
            "scripts/verify_live_copilot_prompt_evidence_v3.py",
        )
    )
)


class ExactIOTransport(httpx.AsyncBaseTransport):
    """Capture exact non-header HTTP bodies and independent attempt counts."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        request_validator: Callable[[bytes], None] | None = None,
    ) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self._request_validator = request_validator
        self._effort: str | None = None
        self._case_id: str | None = None
        self._case_attempt = 0
        self.records: list[dict[str, Any]] = []

    def begin_case(self, *, effort: str, case_id: str) -> None:
        self._effort = effort
        self._case_id = case_id
        self._case_attempt = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._effort is None or self._case_id is None:
            raise RuntimeError("transport_case_not_bound")
        self._case_attempt += 1
        sequence = len(self.records) + 1
        request_body = await request.aread()
        request_text = request_body.decode("utf-8", errors="strict")
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        try:
            if self._request_validator is not None:
                self._request_validator(request_body)
            response = await self._inner.handle_async_request(request)
            response_body = await response.aread()
            response_text = response_body.decode("utf-8", errors="strict")
            completed_at = datetime.now(UTC).isoformat()
            self.records.append(
                {
                    "schema_version": "copilot-live-exact-transport-attempt-v1",
                    "sequence": sequence,
                    "effort": self._effort,
                    "case_id": self._case_id,
                    "case_attempt": self._case_attempt,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "method": request.method,
                    "url": str(request.url),
                    "request_body_utf8": request_text,
                    "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "response_status_code": response.status_code,
                    "response_body_utf8": response_text,
                    "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
                    "transport_error_code": None,
                    "request_headers_persisted": False,
                    "response_headers_persisted": False,
                    "authorization_persisted": False,
                }
            )
            return response
        except asyncio.CancelledError as exc:
            completed_at = datetime.now(UTC).isoformat()
            self.records.append(
                {
                    "schema_version": "copilot-live-exact-transport-attempt-v1",
                    "sequence": sequence,
                    "effort": self._effort,
                    "case_id": self._case_id,
                    "case_attempt": self._case_attempt,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "method": request.method,
                    "url": str(request.url),
                    "request_body_utf8": request_text,
                    "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "response_status_code": None,
                    "response_body_utf8": None,
                    "response_body_sha256": None,
                    "transport_error_code": type(exc).__name__,
                    "request_headers_persisted": False,
                    "response_headers_persisted": False,
                    "authorization_persisted": False,
                }
            )
            raise
        except Exception as exc:
            completed_at = datetime.now(UTC).isoformat()
            self.records.append(
                {
                    "schema_version": "copilot-live-exact-transport-attempt-v1",
                    "sequence": sequence,
                    "effort": self._effort,
                    "case_id": self._case_id,
                    "case_attempt": self._case_attempt,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "method": request.method,
                    "url": str(request.url),
                    "request_body_utf8": request_text,
                    "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "response_status_code": None,
                    "response_body_utf8": None,
                    "response_body_sha256": None,
                    "transport_error_code": type(exc).__name__,
                    "request_headers_persisted": False,
                    "response_headers_persisted": False,
                    "authorization_persisted": False,
                }
            )
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()


def canonical_sha256(value: Any) -> str:
    return base._canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return base._file_sha256(path)


def build_source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": file_sha256(REPOSITORY_ROOT / relative),
            "size_bytes": (REPOSITORY_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_INPUTS
    ]
    snapshot = {"schema_version": "copilot-live-source-snapshot-v2", "files": files}
    snapshot["content_hash"] = canonical_sha256(snapshot)
    return snapshot


def assert_source_snapshot_current(snapshot: dict[str, Any]) -> None:
    if snapshot != build_source_snapshot():
        raise RuntimeError("evaluation_source_changed_during_run")


def exact_io_document(
    *,
    effort: ReasoningEffort,
    model: str,
    source_snapshot: dict[str, Any],
    request_controls: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    document = {
        "schema_version": "copilot-live-exact-llm-io-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "reasoning_effort": effort,
        "prompt_sha256": base.prompt_sha256(),
        "source_snapshot_sha256": source_snapshot["content_hash"],
        "request_controls_sha256": request_controls["content_hash"],
        "scope": "synthetic_fixed_24_only",
        "exact_request_and_response_bodies_persisted": True,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "api_key_persisted": False,
        "records": records,
    }
    document["content_hash"] = canonical_sha256(document)
    return document


async def evaluate_effort(
    *,
    corpus: dict[str, Any],
    corpus_path: Path,
    model: str,
    api_key: str,
    timeout_seconds: float,
    effort: ReasoningEffort,
    source_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    transport = ExactIOTransport()
    http_client = httpx.AsyncClient(transport=transport, timeout=timeout_seconds)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=timeout_seconds,
        http_client=http_client,
    )
    recording = base._RecordingClient(client)
    adapter = OpenAICopilotAdapter(
        model=model,
        api_key="injected-client",
        timeout_seconds=timeout_seconds,
        reasoning_effort=effort,
        client=recording,
    )
    request_controls = base.build_request_controls(model=model, effort=effort, timeout_seconds=timeout_seconds)
    cases = {case["case_id"]: case for case in corpus["cases"]}
    fixtures = corpus["context_fixtures"]
    results: list[dict[str, Any]] = []
    failure_code_counts: dict[str, int] = {}
    try:
        for index, case_id in enumerate(base.LIVE_SUBSET_CASE_IDS, start=1):
            case = cases[case_id]
            transport.begin_case(effort=effort, case_id=case_id)
            recording.responses.begin_case()
            started = time.perf_counter()
            failure_code: str | None = None
            try:
                interpretation = await adapter.interpret(
                    context=base._context(fixtures[case["context_fixture"]]),
                    history=(),
                    message=case["input"],
                )
                state = interpretation.state
                intents = [intent.type for intent in interpretation.intents]
                typed_intents = [intent.as_dict() for intent in interpretation.intents]
                highlights = interpretation.highlights.as_dict()
                permission = interpretation.permission
                references = base.referenced_ids(interpretation)
                structured_valid = True
                input_tokens = recording.responses.input_tokens
                output_tokens = recording.responses.output_tokens
            except Exception as exc:  # noqa: BLE001 - raw diagnostics must not persist
                failure_code = (
                    exc.code
                    if isinstance(exc, (CopilotContractError, OpenAICopilotError))
                    else "evaluation_unexpected_failure"
                )
                failure_code_counts[failure_code] = failure_code_counts.get(failure_code, 0) + 1
                state = "failed"
                intents = []
                typed_intents = []
                highlights = base._empty_highlights()
                permission = None
                references = []
                structured_valid = False
                input_tokens = 0
                output_tokens = 0
            results.append(
                {
                    "case_evidence_schema": "copilot-live-case-evidence-record-v1",
                    "case_id": case_id,
                    "state": state,
                    "intents": intents,
                    "typed_intents": typed_intents,
                    "permission": permission,
                    "highlights": highlights,
                    "referenced_ids": references,
                    "structured_valid": structured_valid,
                    "contract_valid": structured_valid,
                    "automatic_action_observed": recording.responses.automatic_action_observed,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "attempt_count": recording.responses.attempt_count,
                    "prompt_sha256": base.prompt_sha256(),
                    "provider_response_id_sha256": (
                        recording.responses.provider_response_id_sha256 if structured_valid else None
                    ),
                    "provider_created_at": (recording.responses.provider_created_at if structured_valid else None),
                    "provider_model": recording.responses.provider_model or model,
                    "request_controls_sha256": request_controls["content_hash"],
                    "source_snapshot_sha256": source_snapshot["content_hash"],
                    "safe_failure_code": failure_code,
                }
            )
            print(json.dumps({"case": index, "effort": effort, "status": "complete"}))
    finally:
        await client.close()
    assert_source_snapshot_current(source_snapshot)
    summary = base.evaluate_results(corpus, results, model=model)
    summary.update(
        {
            "reasoning_effort": effort,
            "safe_failure_code_counts": dict(sorted(failure_code_counts.items())),
            "raw_provider_content_persisted": False,
            "exact_llm_io_persisted": True,
            "request_controls_sha256": request_controls["content_hash"],
            "corpus_sha256": file_sha256(corpus_path),
            "source_snapshot_sha256": source_snapshot["content_hash"],
            "transport_attempts_total": len(transport.records),
        }
    )
    evidence = base._case_evidence_document(
        effort=effort,
        model=model,
        corpus_path=corpus_path,
        source_snapshot=source_snapshot,
        request_controls=request_controls,
        records=results,
    )
    exact_io = exact_io_document(
        effort=effort,
        model=model,
        source_snapshot=source_snapshot,
        request_controls=request_controls,
        records=transport.records,
    )
    return summary, evidence, exact_io


def build_file_io_manifest(
    *, output_dir: Path, source_snapshot: dict[str, Any], output_paths: list[Path]
) -> dict[str, Any]:
    inputs = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "exact_content_location": row["path"],
        }
        for row in source_snapshot["files"]
    ]
    outputs = [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "exact_content_location": path.relative_to(REPOSITORY_ROOT).as_posix(),
        }
        for path in sorted(output_paths)
    ]
    document = {
        "schema_version": "copilot-live-file-io-manifest-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_files": inputs,
        "output_files": outputs,
        "secret_input_files_omitted": True,
        "api_key_value_persisted": False,
        "content_hash": None,
    }
    document["content_hash"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "content_hash"}
    )
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument(
        "--reasoning-efforts",
        nargs="+",
        choices=("low", "medium"),
        default=("low", "medium"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    load_local_environment(REPOSITORY_ROOT)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("openai_not_configured")
    corpus_path = args.corpus.resolve()
    corpus = base.load_corpus(corpus_path)
    source_snapshot = build_source_snapshot()
    summaries: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    exact_io_hashes: dict[str, str] = {}
    output_paths: list[Path] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for effort in args.reasoning_efforts:
        summary, evidence, exact_io = await evaluate_effort(
            corpus=corpus,
            corpus_path=corpus_path,
            model=args.model,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            effort=effort,
            source_snapshot=source_snapshot,
        )
        evidence_path = args.output_dir / f"{effort}_cases.json"
        exact_io_path = args.output_dir / f"{effort}_exact_llm_io.json"
        base.write_summary(evidence_path, evidence)
        base.write_summary(exact_io_path, exact_io)
        evidence_hash = file_sha256(evidence_path)
        exact_io_hash = file_sha256(exact_io_path)
        summary["case_evidence_path"] = evidence_path.name
        summary["case_evidence_sha256"] = evidence_hash
        summary["exact_llm_io_path"] = exact_io_path.name
        summary["exact_llm_io_sha256"] = exact_io_hash
        summaries[effort] = summary
        evidence_hashes[effort] = evidence_hash
        exact_io_hashes[effort] = exact_io_hash
        summary_path = args.output_dir / f"{effort}_summary.json"
        base.write_summary(summary_path, summary)
        output_paths.extend((evidence_path, exact_io_path, summary_path))
    assert_source_snapshot_current(source_snapshot)
    recommended = base.choose_effort(summaries)
    comparison = {
        "schema_version": "copilot-reasoning-comparison-v3",
        "model": args.model,
        "prompt_version": next(iter(summaries.values()))["prompt_version"],
        "prompt_sha256": next(iter(summaries.values()))["prompt_sha256"],
        "evaluation_scope": "live_fixed_24_exact_io",
        "reasoning_efforts": list(summaries),
        "recommended_reasoning_effort": recommended,
        "all_thresholds_passed": all(summary["thresholds_passed"] for summary in summaries.values()),
        "provider_attempts_total": sum(summary["provider_attempts_total"] for summary in summaries.values()),
        "transport_attempts_total": sum(summary["transport_attempts_total"] for summary in summaries.values()),
        "case_evidence_sha256": evidence_hashes,
        "exact_llm_io_sha256": exact_io_hashes,
        "source_snapshot_sha256": source_snapshot["content_hash"],
        "raw_provider_content_persisted": False,
        "exact_llm_io_persisted": True,
        "summaries": summaries,
    }
    comparison_path = args.output_dir / "reasoning_comparison.json"
    base.write_summary(comparison_path, comparison)
    output_paths.append(comparison_path)
    file_manifest = build_file_io_manifest(
        output_dir=args.output_dir,
        source_snapshot=source_snapshot,
        output_paths=output_paths,
    )
    file_manifest_path = args.output_dir / "file_io_manifest.json"
    base.write_summary(file_manifest_path, file_manifest)
    print(
        json.dumps(
            {
                "provider_attempts_total": comparison["provider_attempts_total"],
                "transport_attempts_total": comparison["transport_attempts_total"],
                "recommended_reasoning_effort": recommended,
                "exact_llm_io_persisted": True,
                "status": "written",
            }
        )
    )
    return 0 if recommended is not None else 2


def main() -> int:
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
