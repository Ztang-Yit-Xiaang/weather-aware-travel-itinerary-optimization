"""Run one exact-I/O billed smoke for the statewide-aware product Copilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from run_live_copilot_prompt_evaluation import (  # noqa: E402
    _canonical_sha256,
    _RecordingClient,
)
from run_live_copilot_prompt_evaluation_v2 import ExactIOTransport  # noqa: E402
from run_product_app import load_local_environment  # noqa: E402

from itinerary_system.product_app.api import create_product_app  # noqa: E402
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter  # noqa: E402

ReasoningEffort = Literal["low", "medium"]
MESSAGE = "what if I want to visit red wood"
SOURCE_INPUTS = (
    "configs/product_app_registry.json",
    "configs/product_poi_catalogs/california_statewide/catalog.json",
    "configs/product_poi_catalogs/california_statewide/manifest.json",
    "scripts/build_california_statewide_poi_catalog.py",
    "scripts/run_live_copilot_statewide_smoke.py",
    "src/itinerary_system/product_app/api.py",
    "src/itinerary_system/product_app/copilot.py",
    "src/itinerary_system/product_app/copilot_prompt.py",
    "src/itinerary_system/product_app/models.py",
    "src/itinerary_system/product_app/openai_copilot.py",
    "src/itinerary_system/product_app/poi_catalog.py",
    "src/itinerary_system/product_app/service.py",
)
PROTECTED_STATE_FIELDS = (
    "trip_id",
    "run_id",
    "revision",
    "accepted_plan_id",
    "selected_day",
    "selected_stop_id",
    "selected_segment_id",
    "selected_candidate_id",
    "selected_alternative_id",
    "draft",
    "proposal",
    "permission_decisions",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": _file_sha256(REPOSITORY_ROOT / relative),
            "size_bytes": (REPOSITORY_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_INPUTS
    ]
    value = {"schema_version": "copilot-statewide-smoke-source-v1", "files": files}
    value["content_hash"] = _canonical_sha256(value)
    return value


def _protected_state(session: dict[str, Any]) -> dict[str, Any]:
    if any(field not in session for field in PROTECTED_STATE_FIELDS):
        raise RuntimeError("statewide_smoke_session_contract_invalid")
    return {field: session[field] for field in PROTECTED_STATE_FIELDS}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report(evidence: dict[str, Any], evidence_path: Path) -> str:
    transport = evidence["exact_llm_transport"]
    return "\n".join(
        (
            "# Current-source statewide Copilot exact-I/O smoke",
            "",
            f"- Completed: `{evidence['completed_at']}`",
            f"- Requested model: `{evidence['requested_model']}`",
            f"- Provider model: `{evidence['provider_model']}`",
            f"- Reasoning effort: `{evidence['reasoning_effort']}`",
            f"- Provider attempts: `{evidence['provider_attempts']}`",
            f"- Protected itinerary state unchanged: `{str(evidence['protected_state_unchanged']).lower()}`",
            f"- Evidence JSON: `{evidence_path.as_posix()}`",
            "- Exact content scope: synthetic California demo only.",
            "- Excluded: API key, Authorization/cookie headers, session token, and raw exception traces.",
            "",
            "## Exact product input",
            "",
            "```json",
            json.dumps(evidence["exact_product_input"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Exact product output",
            "",
            "```json",
            json.dumps(evidence["exact_product_output"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Exact LLM HTTP request body",
            "",
            "```json",
            transport["request_body_utf8"],
            "```",
            "",
            "## Exact LLM HTTP response body",
            "",
            "```json",
            transport["response_body_utf8"],
            "```",
            "",
            "## File inputs",
            "",
            "```json",
            json.dumps(evidence["source_snapshot"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
        )
    )


def run(*, model: str, effort: ReasoningEffort, timeout_seconds: float) -> dict[str, Any]:
    load_local_environment(REPOSITORY_ROOT)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("openai_api_key_missing")
    started_at = datetime.now(UTC).isoformat()
    source_snapshot = _source_snapshot()
    transport = ExactIOTransport()
    http_client = httpx.AsyncClient(transport=transport, timeout=timeout_seconds)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=timeout_seconds,
        http_client=http_client,
    )
    recording = _RecordingClient(client)
    adapter = OpenAICopilotAdapter(
        model=model,
        api_key="injected-client",
        timeout_seconds=timeout_seconds,
        reasoning_effort=effort,
        client=recording,
    )
    with tempfile.TemporaryDirectory(prefix="copilot-statewide-smoke-") as state_root:
        app = create_product_app(
            repository_root=REPOSITORY_ROOT,
            registry_path=REPOSITORY_ROOT / "configs" / "product_app_registry.json",
            state_root=Path(state_root),
            additional_allowed_authorities=("testserver",),
        )
        with TestClient(app) as product:
            created = product.post("/api/sessions", json={}).json()
            session = created["session"]
            token = created["mutation_token"]
            session_id = session["session_id"]
            headers = {"X-Session-Token": token}
            before = product.get(f"/api/sessions/{session_id}", headers=headers).json()["session"]
            app.state.product_service.copilot = adapter
            transport.begin_case(effort=effort, case_id="statewide_redwood_smoke")
            recording.responses.begin_case()
            product_input = {
                "expected_revision": before["revision"],
                "client_message_id": f"client_message_{uuid4().hex}",
                "message": MESSAGE,
            }
            response = product.post(
                f"/api/sessions/{session_id}/copilot/messages",
                headers=headers,
                json=product_input,
            )
            if response.status_code != 200:
                raise RuntimeError(f"statewide_smoke_http_{response.status_code}")
            product_output = response.json()
            after = product.get(f"/api/sessions/{session_id}", headers=headers).json()["session"]
    if len(transport.records) != 1 or recording.responses.attempt_count != 1:
        raise RuntimeError("statewide_smoke_attempt_count_invalid")
    turn = product_output.get("turn")
    if not isinstance(turn, dict):
        raise RuntimeError("statewide_smoke_turn_missing")
    interpretation = turn.get("interpretation")
    if not isinstance(interpretation, dict):
        raise RuntimeError("statewide_smoke_interpretation_missing")
    assistant_message = turn.get("assistant_message")
    if (
        turn.get("provider") != "openai"
        or turn.get("state") != "completed"
        or interpretation.get("state") != "clarification_required"
        or not isinstance(assistant_message, str)
        or "Redwood National and State Parks" not in assistant_message
        or "Nothing was changed" not in assistant_message
    ):
        raise RuntimeError("statewide_smoke_semantics_invalid")
    before_state = _protected_state(before)
    after_state = _protected_state(after)
    evidence = {
        "schema_version": "copilot-statewide-exact-io-smoke-v1",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "requested_model": model,
        "provider_model": recording.responses.provider_model,
        "reasoning_effort": effort,
        "provider_attempts": recording.responses.attempt_count,
        "exact_product_input": product_input,
        "exact_product_output": product_output,
        "exact_llm_transport": transport.records[0],
        "protected_state_before": before_state,
        "protected_state_after": after_state,
        "protected_state_unchanged": before_state == after_state,
        "source_snapshot": source_snapshot,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "api_key_persisted": False,
        "session_token_persisted": False,
    }
    evidence["content_hash"] = _canonical_sha256(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--effort", choices=("low", "medium"), default="low")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = run(
        model=args.model,
        effort=args.effort,
        timeout_seconds=args.timeout_seconds,
    )
    evidence_path = output_dir / "statewide_copilot_exact_io.json"
    report_path = output_dir / "statewide_copilot_exact_io_report.md"
    _write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence, evidence_path), encoding="utf-8")
    manifest = {
        "schema_version": "copilot-statewide-smoke-file-io-v1",
        "inputs": evidence["source_snapshot"]["files"],
        "outputs": [
            {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (evidence_path, report_path)
        ],
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    _write_json(output_dir / "file_io_manifest.json", manifest)
    print(
        json.dumps(
            {
                "model": evidence["provider_model"],
                "attempts": evidence["provider_attempts"],
                "protected_state_unchanged": evidence["protected_state_unchanged"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["protected_state_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
