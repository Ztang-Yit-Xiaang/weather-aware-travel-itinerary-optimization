"""Run a provenance-bound fixed-24 Copilot prompt gate.

Each effort writes sanitized per-case typed evidence plus an aggregate summary.
Raw prompts, provider prose, transcripts, errors, response identifiers, and
secrets are never persisted. This script performs provider calls only when the
operator invokes it explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_copilot_prompt import (  # noqa: E402
    DEFAULT_CORPUS,
    LIVE_SUBSET_CASE_IDS,
    evaluate_results,
    load_corpus,
    write_summary,
)
from run_product_app import load_local_environment  # noqa: E402

from itinerary_system.product_app.copilot import CopilotContractError  # noqa: E402
from itinerary_system.product_app.copilot_prompt import prompt_sha256  # noqa: E402
from itinerary_system.product_app.models import (  # noqa: E402
    CopilotContextV1,
    CopilotInterpretationV1,
)
from itinerary_system.product_app.openai_copilot import (  # noqa: E402
    OpenAICopilotAdapter,
    OpenAICopilotError,
    OpenAIInterpretationSchemaV1,
)

ReasoningEffort = Literal["low", "medium"]
SOURCE_INPUTS = (
    "scripts/evaluate_copilot_prompt.py",
    "scripts/run_live_copilot_prompt_evaluation.py",
    "scripts/verify_live_copilot_prompt_evidence.py",
    "scripts/run_product_app.py",
    "scripts/run_live_copilot_smoke.py",
    "src/itinerary_system/product_app/copilot.py",
    "src/itinerary_system/product_app/copilot_prompt.py",
    "src/itinerary_system/product_app/models.py",
    "src/itinerary_system/product_app/openai_copilot.py",
    "src/itinerary_system/product_app/service.py",
    "tests/product_app/fixtures/copilot_prompt_cases_v1.json",
)


class _RecordingResponses:
    def __init__(self, responses: Any) -> None:
        self._responses = responses
        self.begin_case()

    def begin_case(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.attempt_count = 0
        self.provider_created_at: int | float | None = None
        self.provider_model: str | None = None
        self.provider_response_id_sha256: str | None = None
        self.automatic_action_observed = False

    async def parse(self, **kwargs: Any) -> Any:
        self.attempt_count += 1
        response = await self._responses.parse(**kwargs)
        usage = getattr(response, "usage", None)
        self.input_tokens = _usage_value(usage, "input_tokens")
        self.output_tokens = _usage_value(usage, "output_tokens")
        response_id = getattr(response, "id", None)
        created_at = getattr(response, "created_at", None)
        provider_model = getattr(response, "model", None)
        if not isinstance(response_id, str) or not response_id:
            raise OpenAICopilotError("openai_invalid_structured_output", 502, diagnostic_code="response_id_missing")
        if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
            raise OpenAICopilotError(
                "openai_invalid_structured_output", 502, diagnostic_code="response_created_at_missing"
            )
        if not isinstance(provider_model, str) or not provider_model:
            raise OpenAICopilotError("openai_invalid_structured_output", 502, diagnostic_code="response_model_missing")
        self.provider_response_id_sha256 = hashlib.sha256(response_id.encode()).hexdigest()
        self.provider_created_at = created_at
        self.provider_model = provider_model
        self.automatic_action_observed = _provider_action_observed(response)
        return response


class _RecordingClient:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.responses = _RecordingResponses(client.responses)


def _usage_value(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _provider_action_observed(response: Any) -> bool:
    action_types = {"computer_call", "file_search_call", "function_call", "tool_call", "web_search_call"}
    for item in getattr(response, "output", ()) or ():
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type in action_types:
            return True
    return False


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_snapshot() -> dict[str, Any]:
    files = [{"path": relative, "sha256": _file_sha256(REPOSITORY_ROOT / relative)} for relative in SOURCE_INPUTS]
    snapshot = {"schema_version": "copilot-live-source-snapshot-v1", "files": files}
    snapshot["content_hash"] = _canonical_sha256(snapshot)
    return snapshot


def assert_source_snapshot_current(snapshot: dict[str, Any]) -> None:
    if snapshot != build_source_snapshot():
        raise RuntimeError("evaluation_source_changed_during_run")


def build_request_controls(*, model: str, effort: ReasoningEffort, timeout_seconds: float) -> dict[str, Any]:
    controls = {
        "schema_version": "copilot-live-request-controls-v1",
        "provider_endpoint": "https://api.openai.com/v1",
        "requested_model": model,
        "reasoning_effort": effort,
        "timeout_seconds": float(timeout_seconds),
        "adapter_attempt_limit": 2,
        "sdk_max_retries": 0,
        "store": False,
        "tools": [],
        "max_output_tokens": 1_200,
        "prompt_sha256": prompt_sha256(),
        "response_schema_sha256": _canonical_sha256(OpenAIInterpretationSchemaV1.model_json_schema()),
    }
    controls["content_hash"] = _canonical_sha256(controls)
    return controls


def _context(fixture: dict[str, Any]) -> CopilotContextV1:
    values = dict(fixture)
    for field in (
        "draft_operations",
        "allowed_stop_ids",
        "allowed_candidate_ids",
        "allowed_days",
        "allowed_segment_ids",
        "allowed_alternative_ids",
    ):
        values[field] = tuple(values.get(field, ()))
    return CopilotContextV1(**values)


def referenced_ids(interpretation: CopilotInterpretationV1) -> list[str | int]:
    values: list[str | int] = []
    for intent in interpretation.intents:
        for value in (intent.target_stop_id, intent.destination_day, intent.candidate_id):
            if value is not None and value not in values:
                values.append(value)
    highlights = interpretation.highlights
    for value in (*highlights.day_ids, *highlights.stop_ids, *highlights.segment_ids, *highlights.candidate_ids):
        if value not in values:
            values.append(value)
    return values


def _empty_highlights() -> dict[str, list[Any]]:
    return {"day_ids": [], "stop_ids": [], "segment_ids": [], "candidate_ids": []}


def _case_evidence_document(
    *,
    effort: ReasoningEffort,
    model: str,
    corpus_path: Path,
    source_snapshot: dict[str, Any],
    request_controls: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    document = {
        "schema_version": "copilot-live-case-evidence-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "reasoning_effort": effort,
        "prompt_version": "copilot-prompt-v1",
        "prompt_sha256": prompt_sha256(),
        "corpus_path": corpus_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "corpus_sha256": _file_sha256(corpus_path),
        "source_snapshot": source_snapshot,
        "request_controls": request_controls,
        "raw_provider_content_persisted": False,
        "records": records,
    }
    document["content_hash"] = _canonical_sha256(document)
    return document


async def _evaluate_effort(
    *,
    corpus: dict[str, Any],
    corpus_path: Path,
    model: str,
    api_key: str,
    timeout_seconds: float,
    effort: ReasoningEffort,
    source_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout_seconds)
    recording = _RecordingClient(client)
    adapter = OpenAICopilotAdapter(
        model=model,
        api_key="injected-client",
        timeout_seconds=timeout_seconds,
        reasoning_effort=effort,
        client=recording,
    )
    request_controls = build_request_controls(model=model, effort=effort, timeout_seconds=timeout_seconds)
    cases = {case["case_id"]: case for case in corpus["cases"]}
    fixtures = corpus["context_fixtures"]
    results: list[dict[str, Any]] = []
    failure_code_counts: dict[str, int] = {}
    try:
        for index, case_id in enumerate(LIVE_SUBSET_CASE_IDS, start=1):
            case = cases[case_id]
            recording.responses.begin_case()
            started = time.perf_counter()
            failure_code: str | None = None
            try:
                interpretation = await adapter.interpret(
                    context=_context(fixtures[case["context_fixture"]]),
                    history=(),
                    message=case["input"],
                )
                state = interpretation.state
                intents = [intent.type for intent in interpretation.intents]
                typed_intents = [intent.as_dict() for intent in interpretation.intents]
                highlights = interpretation.highlights.as_dict()
                permission = interpretation.permission
                references = referenced_ids(interpretation)
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
                highlights = _empty_highlights()
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
                    "prompt_sha256": prompt_sha256(),
                    "provider_response_id_sha256": recording.responses.provider_response_id_sha256
                    if structured_valid
                    else None,
                    "provider_created_at": recording.responses.provider_created_at if structured_valid else None,
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
    summary = evaluate_results(corpus, results, model=model)
    summary.update(
        {
            "reasoning_effort": effort,
            "safe_failure_code_counts": dict(sorted(failure_code_counts.items())),
            "raw_provider_content_persisted": False,
            "request_controls_sha256": request_controls["content_hash"],
            "corpus_sha256": _file_sha256(corpus_path),
        }
    )
    evidence = _case_evidence_document(
        effort=effort,
        model=model,
        corpus_path=corpus_path,
        source_snapshot=source_snapshot,
        request_controls=request_controls,
        records=results,
    )
    return summary, evidence


def choose_effort(summaries: dict[str, dict[str, Any]]) -> str | None:
    passing = [name for name, summary in summaries.items() if summary["thresholds_passed"]]
    if not passing:
        return None
    if len(passing) == 1:
        return passing[0]
    return min(
        passing,
        key=lambda name: (
            -summaries[name]["exact_state_allowed_intent_rate"],
            summaries[name]["input_tokens_total"] + summaries[name]["output_tokens_total"],
            summaries[name]["latency_ms_total"],
            0 if name == "low" else 1,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--reasoning-efforts", nargs="+", choices=("low", "medium"), default=("low", "medium"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    load_local_environment(REPOSITORY_ROOT)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("openai_not_configured")
    corpus_path = args.corpus.resolve()
    corpus = load_corpus(corpus_path)
    source_snapshot = build_source_snapshot()
    summaries: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for effort in args.reasoning_efforts:
        summary, evidence = await _evaluate_effort(
            corpus=corpus,
            corpus_path=corpus_path,
            model=args.model,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            effort=effort,
            source_snapshot=source_snapshot,
        )
        evidence_path = args.output_dir / f"{effort}_cases.json"
        write_summary(evidence_path, evidence)
        evidence_hash = _file_sha256(evidence_path)
        summary["case_evidence_path"] = evidence_path.name
        summary["case_evidence_sha256"] = evidence_hash
        summaries[effort] = summary
        evidence_hashes[effort] = evidence_hash
        write_summary(args.output_dir / f"{effort}_summary.json", summary)
    assert_source_snapshot_current(source_snapshot)
    recommended = choose_effort(summaries)
    comparison = {
        "schema_version": "copilot-reasoning-comparison-v2",
        "model": args.model,
        "prompt_version": next(iter(summaries.values()))["prompt_version"],
        "prompt_sha256": next(iter(summaries.values()))["prompt_sha256"],
        "evaluation_scope": "live_fixed_24",
        "reasoning_efforts": list(summaries),
        "recommended_reasoning_effort": recommended,
        "all_thresholds_passed": all(summary["thresholds_passed"] for summary in summaries.values()),
        "provider_attempts_total": sum(summary["provider_attempts_total"] for summary in summaries.values()),
        "case_evidence_sha256": evidence_hashes,
        "source_snapshot_sha256": source_snapshot["content_hash"],
        "raw_provider_content_persisted": False,
        "summaries": summaries,
    }
    write_summary(args.output_dir / "reasoning_comparison.json", comparison)
    print(
        json.dumps(
            {
                "provider_attempts_total": comparison["provider_attempts_total"],
                "recommended_reasoning_effort": recommended,
                "status": "written",
            }
        )
    )
    return 0 if recommended is not None else 2


def main() -> int:
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
