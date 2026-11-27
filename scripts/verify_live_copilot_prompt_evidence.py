"""Independently recompute sanitized live Copilot evaluation evidence.

This verifier performs no provider calls. It rejects source drift, altered
request controls, malformed case evidence, and aggregate/comparison mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_copilot_prompt import evaluate_results, load_corpus  # noqa: E402
from run_live_copilot_prompt_evaluation import (  # noqa: E402
    DEFAULT_CORPUS,
    _canonical_sha256,
    _file_sha256,
    build_request_controls,
    build_source_snapshot,
    choose_effort,
)

_CASE_FIELDS = {
    "schema_version",
    "generated_at",
    "model",
    "reasoning_effort",
    "prompt_version",
    "prompt_sha256",
    "corpus_path",
    "corpus_sha256",
    "source_snapshot",
    "request_controls",
    "raw_provider_content_persisted",
    "records",
    "content_hash",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_json_document")
    return value


def _verify_case_document(*, path: Path, corpus_path: Path, corpus: dict[str, Any], effort: str) -> dict[str, Any]:
    document = _read_json(path)
    if set(document) != _CASE_FIELDS:
        raise ValueError("invalid_case_evidence_fields")
    if document["schema_version"] != "copilot-live-case-evidence-v1":
        raise ValueError("invalid_case_evidence_schema")
    unsigned = dict(document)
    content_hash = unsigned.pop("content_hash")
    if content_hash != _canonical_sha256(unsigned):
        raise ValueError("case_evidence_hash_mismatch")
    if document["raw_provider_content_persisted"] is not False:
        raise ValueError("raw_provider_content_not_allowed")
    if document["reasoning_effort"] != effort or document["model"] == "offline-fake":
        raise ValueError("case_evidence_identity_mismatch")
    if document["corpus_path"] != corpus_path.relative_to(REPOSITORY_ROOT).as_posix():
        raise ValueError("case_evidence_corpus_path_mismatch")
    if document["corpus_sha256"] != _file_sha256(corpus_path):
        raise ValueError("case_evidence_corpus_hash_mismatch")
    if document["source_snapshot"] != build_source_snapshot():
        raise ValueError("case_evidence_source_snapshot_mismatch")
    controls = build_request_controls(
        model=document["model"],
        effort=effort,
        timeout_seconds=document["request_controls"].get("timeout_seconds"),
    )
    if document["request_controls"] != controls:
        raise ValueError("case_evidence_request_controls_mismatch")
    records = document["records"]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("invalid_case_evidence_records")
    summary = evaluate_results(corpus, records, model=document["model"])
    failure_counts: dict[str, int] = {}
    for record in records:
        code = record.get("safe_failure_code")
        if code is not None:
            failure_counts[code] = failure_counts.get(code, 0) + 1
    summary.update(
        {
            "reasoning_effort": effort,
            "safe_failure_code_counts": dict(sorted(failure_counts.items())),
            "raw_provider_content_persisted": False,
            "request_controls_sha256": controls["content_hash"],
            "corpus_sha256": document["corpus_sha256"],
            "case_evidence_path": path.name,
            "case_evidence_sha256": _file_sha256(path),
        }
    )
    return summary


def verify_evidence_directory(output_dir: Path, corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    corpus_path = corpus_path.resolve()
    corpus = load_corpus(corpus_path)
    summaries = {
        effort: _verify_case_document(
            path=output_dir / f"{effort}_cases.json",
            corpus_path=corpus_path,
            corpus=corpus,
            effort=effort,
        )
        for effort in ("low", "medium")
    }
    for effort, recomputed in summaries.items():
        if _read_json(output_dir / f"{effort}_summary.json") != recomputed:
            raise ValueError(f"{effort}_summary_mismatch")
    source_hashes = {summary["source_snapshot_sha256"] for summary in summaries.values()}
    if len(source_hashes) != 1:
        raise ValueError("comparison_source_snapshot_mismatch")
    recommended = choose_effort(summaries)
    comparison = {
        "schema_version": "copilot-reasoning-comparison-v2",
        "model": next(iter(summaries.values()))["model"],
        "prompt_version": next(iter(summaries.values()))["prompt_version"],
        "prompt_sha256": next(iter(summaries.values()))["prompt_sha256"],
        "evaluation_scope": "live_fixed_24",
        "reasoning_efforts": list(summaries),
        "recommended_reasoning_effort": recommended,
        "all_thresholds_passed": all(summary["thresholds_passed"] for summary in summaries.values()),
        "provider_attempts_total": sum(summary["provider_attempts_total"] for summary in summaries.values()),
        "case_evidence_sha256": {effort: summary["case_evidence_sha256"] for effort, summary in summaries.items()},
        "source_snapshot_sha256": next(iter(source_hashes)),
        "raw_provider_content_persisted": False,
        "summaries": summaries,
    }
    if _read_json(output_dir / "reasoning_comparison.json") != comparison:
        raise ValueError("reasoning_comparison_mismatch")
    return {
        "schema_version": "copilot-live-evidence-verification-v1",
        "verified": True,
        "recommended_reasoning_effort": recommended,
        "provider_attempts_total": comparison["provider_attempts_total"],
        "source_snapshot_sha256": comparison["source_snapshot_sha256"],
        "case_evidence_sha256": comparison["case_evidence_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(verify_evidence_directory(args.output_dir, args.corpus), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
