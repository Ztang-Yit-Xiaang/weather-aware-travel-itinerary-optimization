"""Independently verify one frozen E2E-1L website-Copilot evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from openai.lib._pydantic import to_strict_json_schema

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from e2e1_source_universe import live_source_inputs  # noqa: E402 - standalone script path bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from itinerary_system.product_app.copilot_prompt import DEVELOPER_INSTRUCTIONS  # noqa: E402
from itinerary_system.product_app.openai_copilot import (  # noqa: E402
    OpenAIInterpretationSchemaV1,
    OpenAIInterpretationSchemaV2,
)

MANIFEST_NAME = "file_io_manifest.json"
REPORT_NAME = "exact_io_report.md"
SCENARIO = "live_copilot_chat"
EXPECTED_PROMPT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, "
    "preserving the rest of the seven-day trip."
)
COMBINED_PROMPT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; "
    "set Griffith Observatory duration to 90 minutes."
)
TARGET = "bixby_creek_bridge_viewpoint"
CANDIDATE = "pfeiffer_big_sur_state_park"
SOURCE_INPUTS = (*live_source_inputs(
    "tests/product_app/browser/test_e2e1_live_copilot_chat.py"
), "scripts/e2e1_provider_isolation.py")
SOURCE_KEYS = {"path", "size_bytes", "sha256"}
SESSION_ID = re.compile(r"session_[0-9a-f]{32}")
CLIENT_MESSAGE_ID = re.compile(r"client_message_[0-9a-f]{32}")
STREAM_REQUEST_ID = re.compile(r"copilot_request_[0-9a-f]{32}")
PROVIDER_REQUEST_KEYS = {
    "input", "instructions", "max_output_tokens", "model", "reasoning",
    "store", "text", "tools",
}
PROVIDER_ATTEMPT_KEYS = {
    "schema_version", "sequence", "effort", "case_id", "case_attempt",
    "started_at", "completed_at", "latency_ms", "method", "url",
    "request_body_utf8", "request_body_sha256", "response_status_code",
    "response_body_utf8", "response_body_sha256", "transport_error_code",
    "request_headers_persisted", "response_headers_persisted",
    "authorization_persisted",
}
PROVIDER_DOCUMENT_KEYS = {
    "schema_version", "generated_at", "scope", "model", "reasoning_effort",
    "request_headers_persisted", "response_headers_persisted",
    "authorization_persisted", "records", "content_hash",
}
HISTORICAL_CONTEXT_KEYS = {
    "schema_version", "session_id", "session_revision", "run_id", "trip_id",
    "accepted_plan_id", "selected_day", "selected_stop_id",
    "selected_segment_id", "selected_candidate_id", "selected_alternative_id",
    "allowed_days", "allowed_stop_ids", "allowed_segment_ids",
    "allowed_candidate_ids", "allowed_alternative_ids", "known_places",
    "draft_operations", "evaluated_proposal",
}
CONTEXT_KEYS = HISTORICAL_CONTEXT_KEYS | {"accepted_plan_content_hash", "capabilities", "conversation_memory"}
EXPECTED_NORMALIZED_CONTEXT_SHA256 = (
    "c57e8c42023a8897323b22d0e51c9c97223b6063b497afb46f3885cc0a782995"
)
AUTHORIZED_HISTORICAL_NORMALIZED_CONTEXT_SHA256 = {
    "9e11b56b6d6e61481f9f4cc55ef6210d7303fa25dd6ea124ae2579ad79f3eb62",
}
AUTHORIZED_HISTORICAL_INSTRUCTIONS_SHA256 = {
    "6685aabbddd7a94d2d594331a5d9c780953c838bdb2f80a92e14f0ba9b11d711",
}
INPUT_RATE = Decimal("0.75")
CACHED_INPUT_RATE = Decimal("0.075")
OUTPUT_RATE = Decimal("4.50")
AUTHORIZED_HISTORICAL_OUTPUT_SCHEMA_SHA256 = {
    "0e3115581815d32954bea65cfee3534a1ea7887e448f4c385e5de8da7ac41027",
}
ZERO_USAGE = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "uncached_input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}
TRACE_KEYS = {
    "runtime_execution_id", "runtime_execution_sha256", "runtime_request_content_hash",
    "executed_method", "evaluation_executed",
    "schema_version", "execution_mode", "authorization_scope",
    "browser_prompt_sha256", "session_id", "client_message_id",
    "copilot_api_request_sha256", "copilot_api_response_sha256",
    "draft_api_request_sha256", "draft_api_response_sha256",
    "preview_api_request_sha256", "preview_api_response_sha256",
    "provider", "model", "reasoning_effort", "provider_transport_attempts",
    "provider_calls_current_run", "provider_request_body_sha256",
    "provider_response_body_sha256", "provider_request_authorization",
    "interpretation_sha256", "conversation_turn_id", "draft_operation_id",
    "draft_content_hash", "preview_request_id", "parent_plan_id",
    "child_plan_id", "child_plan_content_hash", "diff_id", "certificate_id",
    "geography_hash", "rendered_map_plan_id", "rendered_geometry_hash",
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "output_tokens", "total_tokens", "billed_cost_usd",
    "accept_or_keep_invoked", "assertions_passed",
}
HISTORICAL_PROVIDER_KEYS = {
    "schema_version", "provider_transport_attempts", "provider_io_source_path",
    "provider_io_file_sha256", "replayed_provider_io_file_sha256",
    "provider_io_content_hash", "provider_request_body_sha256",
    "provider_response_body_sha256", "request_authorization", "usage",
    "billed_cost_usd",
}


class LiveE2E1VerificationError(RuntimeError):
    """Stable live evidence-verification failure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LiveE2E1VerificationError(code)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "e2e1_live_json_object_required")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _canonical_prompt_json(value: Any) -> str:
    return (
        _canonical_json(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _parse_timestamp(value: object, code: str) -> datetime:
    _require(isinstance(value, str), code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LiveE2E1VerificationError(code) from error
    _require(parsed.tzinfo is not None, code)
    return parsed


def _safe(root: Path, relative: object) -> Path:
    _require(isinstance(relative, str) and relative, "e2e1_live_path_invalid")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and ".." not in pure.parts, "e2e1_live_path_invalid")
    path = (root / Path(*pure.parts)).resolve()
    _require(path == root or root in path.parents, "e2e1_live_path_escape")
    return path


def _provider_projection(response: dict[str, Any]) -> dict[str, Any]:
    texts = [
        content.get("text")
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "message"
        for content in item.get("content", [])
        if isinstance(content, dict) and content.get("type") == "output_text"
    ]
    _require(len(texts) == 1 and isinstance(texts[0], str), "e2e1_live_provider_output_invalid")
    value = json.loads(texts[0])
    _require(isinstance(value, dict), "e2e1_live_provider_projection_invalid")
    return value


def _provider_request_context(
    request: dict[str, Any],
    *,
    request_text: str,
    allow_historical_schema: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(set(request) == PROVIDER_REQUEST_KEYS, "e2e1_live_provider_request_fields_invalid")
    expected_format = {
        "type": "json_schema",
        "strict": True,
        "name": "OpenAIInterpretationSchemaV2",
        "schema": to_strict_json_schema(OpenAIInterpretationSchemaV2),
    }
    actual_text = request.get("text")
    actual_format = actual_text.get("format") if isinstance(actual_text, dict) else None
    actual_schema = actual_format.get("schema") if isinstance(actual_format, dict) else None
    actual_schema_sha256 = (
        _canonical_sha256(actual_schema) if isinstance(actual_schema, dict) else None
    )
    historical_text_allowed = (
        allow_historical_schema
        and isinstance(actual_text, dict)
        and set(actual_text) == {"format", "verbosity"}
        and actual_text.get("verbosity") == "low"
        and isinstance(actual_format, dict)
        and set(actual_format) == {"type", "name", "schema", "strict"}
        and actual_format.get("type") == "json_schema"
        and actual_format.get("name") == "OpenAIInterpretationSchemaV1"
        and actual_format.get("strict") is True
        and actual_schema_sha256 in AUTHORIZED_HISTORICAL_OUTPUT_SCHEMA_SHA256
    )
    _require(
        request.get("model") == "gpt-5.4-mini"
        and request.get("store") is False
        and request.get("tools") == []
        and (request.get("instructions") == DEVELOPER_INSTRUCTIONS or (
            allow_historical_schema and isinstance(request.get("instructions"), str)
            and _hash_text(request["instructions"]) in AUTHORIZED_HISTORICAL_INSTRUCTIONS_SHA256
        ))
        and request.get("max_output_tokens") == 1_200
        and request.get("reasoning") == {"effort": "low", "context": "current_turn"}
        and (
            request.get("text") == {"verbosity": "low", "format": expected_format}
            or historical_text_allowed
        ),
        "e2e1_live_provider_controls_invalid",
    )
    prompt_input = request.get("input")
    _require(isinstance(prompt_input, str), "e2e1_live_provider_input_invalid")
    match = re.fullmatch(
        r"<workspace_context>\n(.*)\n</workspace_context>\n"
        r"<recent_conversation>\n(.*)\n</recent_conversation>\n"
        r"<user_request>\n(.*)\n</user_request>",
        prompt_input,
        flags=re.DOTALL,
    )
    _require(match is not None, "e2e1_live_provider_input_invalid")
    assert match is not None
    try:
        context = json.loads(match.group(1))
        history = json.loads(match.group(2))
        user_request = json.loads(match.group(3))
    except json.JSONDecodeError as error:
        raise LiveE2E1VerificationError("e2e1_live_provider_input_invalid") from error
    _require(
        isinstance(context, dict)
        and ((context.get("schema_version") == "copilot-context-v2" and set(context) == CONTEXT_KEYS
              and request.get("instructions") == DEVELOPER_INSTRUCTIONS
              and request.get("text") == {"verbosity": "low", "format": expected_format})
             or (allow_historical_schema and context.get("schema_version") == "copilot-context-v1"
                 and set(context) == HISTORICAL_CONTEXT_KEYS
                 and _hash_text(request["instructions"]) in AUTHORIZED_HISTORICAL_INSTRUCTIONS_SHA256))
        and SESSION_ID.fullmatch(str(context.get("session_id"))) is not None
        and type(context.get("session_revision")) is int and context.get("session_revision") == 1
        and context.get("run_id") == "california_coast_coherent_demo_v1"
        and context.get("trip_id") == "california_coast_approved_seven_day"
        and context.get("selected_day") == 5
        and context.get("selected_stop_id") == TARGET
        and context.get("draft_operations") == []
        and context.get("evaluated_proposal") is None
        and history == []
        and user_request in (EXPECTED_PROMPT, COMBINED_PROMPT),
        "e2e1_live_provider_context_invalid",
    )
    if context["schema_version"] == "copilot-context-v2":
        _verify_current_context_extensions(context)
    _require(
        prompt_input
        == "\n".join(
            (
                "<workspace_context>",
                _canonical_prompt_json(context),
                "</workspace_context>",
                "<recent_conversation>",
                _canonical_prompt_json(history),
                "</recent_conversation>",
                "<user_request>",
                _canonical_prompt_json(user_request),
                "</user_request>",
            )
        ),
        "e2e1_live_provider_input_not_canonical",
    )
    normalized_context = dict(context)
    normalized_context["session_id"] = "<SESSION_ID>"
    normalized_context_sha256 = hashlib.sha256(
        _canonical_prompt_json(normalized_context).encode()
    ).hexdigest()
    _require(
        (context["schema_version"] == "copilot-context-v2" and normalized_context_sha256 == EXPECTED_NORMALIZED_CONTEXT_SHA256)
        or (
            allow_historical_schema
            and context["schema_version"] == "copilot-context-v1"
            and normalized_context_sha256
            in AUTHORIZED_HISTORICAL_NORMALIZED_CONTEXT_SHA256
        ),
        "e2e1_live_provider_context_hash_invalid",
    )
    authorized_text = request["text"]
    authorization = {
        "schema_version": "e2e1-live-provider-request-authorization-v1",
        "request_body_sha256": _hash_text(request_text),
        "request_body_size_bytes": len(request_text.encode("utf-8")),
        "request_body_exact_location": (
            "live_copilot_chat/provider_exact_llm_io.json"
            "#records[0].request_body_utf8"
        ),
        "request_fields": sorted(PROVIDER_REQUEST_KEYS),
        "instructions": request["instructions"],
        "instructions_sha256": _hash_text(request["instructions"]),
        "input_sha256": _hash_text(prompt_input),
        "workspace_context": context,
        "workspace_context_sha256": _canonical_sha256(context),
        "recent_conversation": history,
        "recent_conversation_sha256": _canonical_sha256(history),
        "user_prompt": user_request,
        "user_prompt_sha256": _hash_text(user_request),
        "model": "gpt-5.4-mini",
        "reasoning": {"effort": "low", "context": "current_turn"},
        "store": False,
        "tools": [],
        "max_output_tokens": 1_200,
        "text": authorized_text,
        "text_sha256": _canonical_sha256(authorized_text),
        "output_schema_sha256": actual_schema_sha256,
        "output_schema_current": actual_schema
        == to_strict_json_schema(OpenAIInterpretationSchemaV2),
        "session_id": context["session_id"],
    }
    return context, authorization


def _verify_current_context_extensions(context: dict[str, Any]) -> None:
    """Independently constrain the first-message v2 fixture before its whole hash."""
    code = "e2e1_live_provider_context_v2_invalid"
    plan_hash = context.get("accepted_plan_content_hash")
    _require(type(plan_hash) is str and re.fullmatch(r"[a-f0-9]{16,64}", plan_hash) is not None, code)
    _require(context.get("conversation_memory") == {
        "schema_version": "copilot-memory-v1", "plan_id": context["accepted_plan_id"], "plan_content_hash": plan_hash,
        "pending_clarification": None, "resolved_ids": {"stop_ids": [], "candidate_ids": [], "day_ids": [], "segment_ids": []},
        "proposed_preferences": [], "last_proposal": None,
    }, code)
    caps = context.get("capabilities")
    _require(type(caps) is dict and set(caps) == {"schema_version", "operations", "registered_repair_current", "candidate_replacements", "candidate_day_readiness", "stop_positions", "executable_combinations"}
             and caps["schema_version"] == "copilot-capabilities-v2" and caps["registered_repair_current"] is True
             and type(caps["operations"]) is dict and set(caps["operations"]) == {
                 "keep_stop", "lock_stop", "mark_flexible", "move_day", "replace_nearby", "add_candidate", "route_feedback",
                 "set_stop_duration", "set_stop_order", "set_stop_time_window", "set_stop_role", "replan_trip"}, code)
    _require(caps["executable_combinations"] == [{"operations":["keep_stop", "replace_nearby", "set_stop_duration",
        "set_stop_order", "set_stop_role", "set_stop_time_window"], "cardinality":32},
        {"operations": ["replan_trip"], "cardinality": 1}], code)
    _require(type(caps["executable_combinations"][-1]["cardinality"]) is int
             and caps["operations"]["replan_trip"] == {"enabled": True, "preview_executable": True, "blocking_code": None}, code)
    for row in caps["operations"].values():
        _require(type(row) is dict and set(row) == {"enabled", "preview_executable", "blocking_code"}
                 and type(row["enabled"]) is bool and type(row["preview_executable"]) is bool
                 and (not row["preview_executable"] or row["enabled"])
                 and (row["blocking_code"] is None or type(row["blocking_code"]) is str and re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", row["blocking_code"]) is not None), code)
    positions = caps["stop_positions"]
    _require(type(positions) is list and 0 < len(positions) <= 64, code)
    days: dict[int, list[int]] = {}
    for row in positions:
        _require(type(row) is dict and set(row) == {"stop_id", "day", "sequence_index"}
                 and row["stop_id"] in context["allowed_stop_ids"]
                 and type(row["day"]) is int and row["day"] in context["allowed_days"]
                 and type(row["sequence_index"]) is int and row["sequence_index"] >= 0, code)
        days.setdefault(row["day"], []).append(row["sequence_index"])
    _require([row["stop_id"] for row in positions] == context["allowed_stop_ids"]
             and len({row["stop_id"] for row in positions}) == len(positions)
             and all(indices == list(range(len(indices))) for indices in days.values()), code)
    for name, maximum, fields in (
        ("candidate_replacements", 64, {"candidate_id", "replaces_stop_id"}),
        ("candidate_day_readiness", 128, {"candidate_id", "day", "preview_executable", "blocking_code"}),
    ):
        rows = caps[name]
        _require(type(rows) is list and len(rows) <= maximum, code)
        identities = []
        for row in rows:
            _require(type(row) is dict and set(row) == fields and row["candidate_id"] in context["allowed_candidate_ids"], code)
            if name == "candidate_replacements":
                _require(row["replaces_stop_id"] in context["allowed_stop_ids"], code)
                identities.append((row["candidate_id"], row["replaces_stop_id"]))
            else:
                _require(type(row["day"]) is int and row["day"] in context["allowed_days"] and type(row["preview_executable"]) is bool
                         and (row["blocking_code"] is None or type(row["blocking_code"]) is str and re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", row["blocking_code"]) is not None), code)
                identities.append((row["candidate_id"], row["day"]))
        _require(len(identities) == len(set(identities)), code)


def _verify_approved_interpretation(provider_projection: dict[str, Any], interpretation: dict[str, Any],
                                    *, prompt: str = EXPECTED_PROMPT) -> None:
    combined = prompt == COMBINED_PROMPT
    _require(prompt in (EXPECTED_PROMPT, COMBINED_PROMPT), "e2e1_live_intent_invalid")
    for field in ("schema_version", "state", "intents", "permission"):
        _require(provider_projection.get(field) == interpretation.get(field), "e2e1_live_provider_projection_mismatch")
    _require(interpretation.get("schema_version") in {"copilot-interpretation-v1", "copilot-interpretation-v2"}, "e2e1_live_intent_invalid")
    expected = {"type": "replace_nearby", "target_stop_id": TARGET, "destination_day": None,
                "candidate_id": CANDIDATE, "preference": None, "clarification_question": None}
    if interpretation["schema_version"] == "copilot-interpretation-v2":
        expected["parameters"] = None
    expected_intents = [expected]
    expected_highlights = {"day_ids": [5], "stop_ids": [TARGET], "segment_ids": [], "candidate_ids": [CANDIDATE]}
    if combined:
        _require(interpretation["schema_version"] == "copilot-interpretation-v2", "e2e1_live_intent_invalid")
        expected_intents.append({"type": "set_stop_duration", "target_stop_id": "griffith_observatory",
            "destination_day": None, "candidate_id": None, "preference": None, "clarification_question": None,
            "parameters": {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}})
        expected_highlights = {"day_ids": [5, 7], "stop_ids": [TARGET, "griffith_observatory"],
            "segment_ids": [], "candidate_ids": [CANDIDATE]}
    # Current highlights are derived locally after exact intent comparison.
    _require(interpretation.get("highlights") == expected_highlights,
             "e2e1_live_combined_highlights_invalid" if combined else "e2e1_live_highlights_invalid")
    _require(interpretation.get("state") == "proposal_ready" and interpretation.get("intents") == expected_intents, "e2e1_live_intent_invalid")


def _verify_provider_records(
    provider: dict[str, Any],
    *,
    allow_historical_schema: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _require(set(provider) == PROVIDER_DOCUMENT_KEYS, "e2e1_live_provider_fields_invalid")
    _parse_timestamp(provider.get("generated_at"), "e2e1_live_provider_generated_at_invalid")
    _require(
        provider.get("schema_version") == "e2e1-live-provider-exact-io-v1"
        and provider.get("scope") == "synthetic_california_coast_single_chat"
        and provider.get("model") == "gpt-5.4-mini"
        and provider.get("reasoning_effort") == "low"
        and provider.get("request_headers_persisted") is False
        and provider.get("response_headers_persisted") is False
        and provider.get("authorization_persisted") is False,
        "e2e1_live_provider_document_invalid",
    )
    provider_seed = dict(provider)
    provider_hash = provider_seed.pop("content_hash")
    _require(provider_hash == _canonical_sha256(provider_seed), "e2e1_live_provider_hash_invalid")
    records = provider.get("records")
    _require(isinstance(records, list) and len(records) == 1, "e2e1_live_attempts_invalid")
    contexts: list[dict[str, Any]] = []
    authorizations: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        _require(isinstance(row, dict) and set(row) == PROVIDER_ATTEMPT_KEYS, "e2e1_live_attempt_fields_invalid")
        started = _parse_timestamp(row.get("started_at"), "e2e1_live_attempt_timestamp_invalid")
        completed = _parse_timestamp(row.get("completed_at"), "e2e1_live_attempt_timestamp_invalid")
        _require(
            completed >= started
            and isinstance(row.get("latency_ms"), (int, float))
            and not isinstance(row.get("latency_ms"), bool)
            and row["latency_ms"] >= 0
            and row.get("schema_version") == "copilot-live-exact-transport-attempt-v1"
            and row.get("sequence") == index
            and row.get("case_attempt") == index
            and row.get("case_id") == "e2e1_live_website_chat"
            and row.get("effort") == "low"
            and row.get("method") == "POST"
            and row.get("url") == "https://api.openai.com/v1/responses"
            and row.get("request_headers_persisted") is False
            and row.get("response_headers_persisted") is False
            and row.get("authorization_persisted") is False,
            "e2e1_live_transport_identity_invalid",
        )
        request_text = row.get("request_body_utf8")
        _require(
            isinstance(request_text, str)
            and row.get("request_body_sha256") == _hash_text(request_text),
            "e2e1_live_provider_request_hash_invalid",
        )
        try:
            request = json.loads(request_text)
        except json.JSONDecodeError as error:
            raise LiveE2E1VerificationError("e2e1_live_provider_request_invalid") from error
        _require(isinstance(request, dict), "e2e1_live_provider_request_invalid")
        context, authorization = _provider_request_context(
            request,
            request_text=request_text,
            allow_historical_schema=allow_historical_schema,
        )
        contexts.append(context)
        authorizations.append(authorization)
        status = row.get("response_status_code")
        response_text = row.get("response_body_utf8")
        response_hash = row.get("response_body_sha256")
        transport_error = row.get("transport_error_code")
        _require(
            status == 200
            and isinstance(response_text, str)
            and response_hash == _hash_text(response_text)
            and transport_error is None,
            "e2e1_live_provider_response_hash_invalid",
        )
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise LiveE2E1VerificationError(
                "e2e1_live_provider_response_invalid"
            ) from error
        _require(
            isinstance(response, dict) and response.get("status") == "completed",
            "e2e1_live_provider_not_completed",
        )
    return records, contexts[0], authorizations[0]


def _sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.splitlines()
        _require(
            len(lines) == 1 and lines[0].startswith("data: "),
            "e2e1_live_copilot_stream_frame_invalid",
        )
        value = json.loads(lines[0].removeprefix("data: "))
        _require(isinstance(value, dict), "e2e1_live_copilot_stream_event_invalid")
        events.append(value)
    _require(len(events) == 3, "e2e1_live_copilot_stream_invalid")
    common = {
        "schema_version", "type", "sequence", "request_id", "client_message_id",
    }
    expected_keys = (
        common,
        common | {"state", "message"},
        common | {"payload"},
    )
    _require(
        [event.get("type") for event in events]
        == ["message.accepted", "assistant.status", "turn.completed"]
        and all(type(event.get("sequence")) is int for event in events)
        and [event.get("sequence") for event in events] == [1, 2, 3]
        and all(
            set(event) == keys
            for event, keys in zip(events, expected_keys, strict=True)
        )
        and all(event.get("schema_version") == "copilot-stream-v1" for event in events)
        and all(type(event.get("type")) is str for event in events)
        and len({event.get("request_id") for event in events}) == 1
        and STREAM_REQUEST_ID.fullmatch(str(events[0].get("request_id"))) is not None
        and len({event.get("client_message_id") for event in events}) == 1
        and CLIENT_MESSAGE_ID.fullmatch(str(events[0].get("client_message_id"))) is not None
        and events[1].get("state") == "interpreting"
        and events[1].get("message")
        == "Reading the current itinerary and validating a typed suggestion."
        and isinstance(events[2].get("payload"), dict),
        "e2e1_live_copilot_stream_invalid",
    )
    return events


def _verify_http_record(
    record: dict[str, Any],
    suffix: str,
    *,
    base_url: str,
    endpoint_suffix: str,
    response_format: str = "json",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    expected_keys = {
        "schema_version", "method", "url", "request_body_utf8", "request_body_sha256",
        "response_status_code", "response_body_utf8", "response_body_sha256",
        "request_headers_persisted", "response_headers_persisted",
        "authorization_persisted", "session_token_persisted",
        "request_original_body_sha256", "response_original_body_sha256", "credential_redactions", "body_capture",
    }
    _require(set(record) == expected_keys, f"{suffix}_fields_invalid")
    _require(record.get("schema_version") == "e2e1-product-http-io-v1", f"{suffix}_schema_invalid")
    url = record.get("url")
    _require(isinstance(url, str), f"{suffix}_request_target_invalid")
    parsed = urlparse(url)
    base = urlparse(base_url)
    path_match = re.fullmatch(
        rf"/api/sessions/(session_[0-9a-f]{{32}}){re.escape(endpoint_suffix)}",
        parsed.path,
    )
    _require(
        record.get("method") == "POST"
        and parsed.scheme == base.scheme
        and parsed.netloc == base.netloc
        and path_match is not None
        and not parsed.query
        and not parsed.fragment,
        f"{suffix}_request_target_invalid",
    )
    request_text = record.get("request_body_utf8")
    response_text = record.get("response_body_utf8")
    _require(isinstance(request_text, str) and isinstance(response_text, str), f"{suffix}_body_invalid")
    _require(record.get("request_body_sha256") == _hash_text(request_text), f"{suffix}_request_hash")
    _require(record.get("response_body_sha256") == _hash_text(response_text), f"{suffix}_response_hash")
    _require(record.get("body_capture") == "exact_utf8"
             and record.get("credential_redactions") == {"request":[], "response":[]}
             and record.get("request_original_body_sha256") == record["request_body_sha256"]
             and record.get("response_original_body_sha256") == record["response_body_sha256"], f"{suffix}_exact_capture_invalid")
    _require(record.get("response_status_code") == 200, f"{suffix}_status_invalid")
    _require(
        record.get("request_headers_persisted") is False
        and record.get("response_headers_persisted") is False
        and record.get("authorization_persisted") is False
        and record.get("session_token_persisted") is False,
        f"{suffix}_secret_boundary_invalid",
    )
    stream_events = None
    if response_format == "json":
        response_value = json.loads(response_text)
    elif response_format == "sse":
        stream_events = _sse_events(response_text)
        response_value = stream_events[-1]["payload"]
    else:
        raise LiveE2E1VerificationError(f"{suffix}_response_format_invalid")
    _require(isinstance(response_value, dict), f"{suffix}_response_object_invalid")
    request_value = json.loads(request_text)
    _require(isinstance(request_value, dict), f"{suffix}_request_object_invalid")
    if stream_events is not None:
        _require(
            request_value.get("client_message_id")
            == stream_events[0]["client_message_id"]
            == response_value.get("turn", {}).get("client_message_id"),
            f"{suffix}_stream_identity_invalid",
        )
    assert path_match is not None
    return request_value, response_value, path_match.group(1)


def _verify_replay_current_run_accounting(
    *,
    trace: dict[str, Any],
    manifest: dict[str, Any],
    usage_file: dict[str, Any],
    historical_provider_evidence: dict[str, Any],
) -> None:
    authorization = manifest.get("authorization")
    _require(
        trace.get("execution_mode") == "recorded_exact_output_replay"
        and trace.get("provider_transport_attempts") == 0
        and trace.get("provider_calls_current_run") == 0
        and trace.get("provider_request_body_sha256") is None
        and trace.get("provider_response_body_sha256") is None
        and trace.get("provider_request_authorization") is None
        and all(trace.get(key) == 0 for key in ZERO_USAGE)
        and trace.get("billed_cost_usd") == "0"
        and trace.get("historical_provider_evidence")
        == historical_provider_evidence
        and manifest.get("provider_transport_attempts") == 0
        and manifest.get("usage") == ZERO_USAGE
        and isinstance(manifest.get("pricing"), dict)
        and manifest["pricing"].get("billed_cost_usd") == "0"
        and isinstance(authorization, dict)
        and authorization.get("confirmed") is False
        and authorization.get("execution_mode")
        == "recorded_exact_output_replay"
        and authorization.get("current_run_provider_calls") == 0
        and authorization.get("transport_attempt_cap") == 0
        and authorization.get("exact_provider_request") is None
        and authorization.get("historical_provider_evidence")
        == historical_provider_evidence
        and usage_file.get("schema_version")
        == "e2e1-live-replay-usage-and-cost-v1"
        and usage_file.get("execution_mode")
        == "recorded_exact_output_replay"
        and usage_file.get("current_run_provider_calls") == 0
        and usage_file.get("current_run_usage") == ZERO_USAGE
        and usage_file.get("current_run_billed_cost_usd") == "0"
        and usage_file.get("historical_provider_evidence")
        == historical_provider_evidence,
        "e2e1_live_replay_current_run_accounting_invalid",
    )


def _artifact_hash(value: dict[str, Any]) -> str:
    return _hash_text(json.dumps(value, sort_keys=True, separators=(",", ":")))[:16]


def _independent_geometry_hash(geography: dict[str, Any]) -> str:
    # Node supplies ECMAScript number encoding; no production JS is imported.
    code = """
      let text = ''; process.stdin.setEncoding('utf8');
      process.stdin.on('data', chunk => { text += chunk; });
      process.stdin.on('end', () => {
        const legs = JSON.parse(text).validated_legs;
        for (const feature of legs.features) delete feature.properties.role;
        const encode = value => Array.isArray(value) ? '[' + value.map(encode).join(',') + ']'
          : value !== null && typeof value === 'object'
            ? '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + encode(value[key])).join(',') + '}'
            : JSON.stringify(value);
        process.stdout.write(require('node:crypto').createHash('sha256').update(encode(legs)).digest('hex'));
      });
    """
    completed = subprocess.run(
        ["node", "-e", code], input=json.dumps(geography),
        text=True, capture_output=True, check=True, timeout=30,
    )
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", completed.stdout)), "e2e1_live_geometry_digest_invalid")
    return completed.stdout


def _verify_visible_metrics(snapshot: dict[str, Any], proposal: dict[str, Any]) -> None:
    """Bind the visible metric values to the independently evaluated map plan."""
    identity = {"planId": proposal["child_plan_id"], "planContentHash": proposal["child_plan_content_hash"]}
    _require(snapshot.get("scope") == {**identity, "evaluationId": proposal["certificate_id"]},
             "e2e1_live_visible_metrics_identity_mismatch")
    rows = snapshot.get("metrics", [])
    _require(len(rows) == 5 and all(row.get("scope") == identity for row in rows),
             "e2e1_live_visible_metrics_identity_mismatch")
    legs = proposal["geography_plan"]["validated_legs"]["features"]
    route_minutes = sum(leg["properties"]["duration_s"] / 60 for leg in legs)
    tradeoffs = proposal["repair"]["tradeoffs"]

    def display(value: float | None) -> str:
        return "Unavailable" if value is None else f"{value:.2f}".rstrip("0").rstrip(".")

    preservation = tradeoffs["preservation_rate"]
    expected = {
        "Route time": display(route_minutes) + " min",
        "Contextual risk": display(tradeoffs["weather_risk_delta"]),
        "Preservation": "Unavailable" if preservation is None else f"{int(preservation * 100 + .5)}%",
        "Edit cost": display(tradeoffs["weighted_edit_cost"]),
        "Certificate": proposal["eligibility"],
    }
    _require({row.get("label"): row.get("value") for row in rows} == expected,
             "e2e1_live_visible_metrics_values_mismatch")


def _verify_runtime_execution(proposal: dict[str, Any]) -> dict[str, Any]:
    receipt = proposal.get("runtime_execution")
    combined = proposal.get("execution_mode") == "atomic_combined_independent_evaluation"
    _require(
        proposal.get("execution_mode") == ("atomic_combined_independent_evaluation" if combined else "deterministic_repair_pipeline")
        and isinstance(receipt, dict)
        and receipt.get("schema_version") == ("atomic-combined-repair-execution-v1" if combined else "runtime-repair-execution-v1")
        and re.fullmatch(r"execution_[0-9a-f]{32}", str(receipt.get("execution_id"))) is not None
        and receipt.get("evaluation_executed") is True
        and receipt.get("evaluator") == "PlanEvaluator.evaluate_final_plan",
        "e2e1_live_runtime_execution_missing",
    )
    request = receipt["repair_request"]
    planner = receipt["planner_run"]
    child = receipt["child_plan"]
    certificate = receipt["evaluation_certificate"]
    diff = receipt["plan_diff"]
    constraints = request["confirmed_constraints"]
    _require(
        _parse_timestamp(receipt["started_at"], "e2e1_live_runtime_timestamp_invalid")
        <= _parse_timestamp(certificate["evaluated_at"], "e2e1_live_runtime_timestamp_invalid")
        <= _parse_timestamp(receipt["completed_at"], "e2e1_live_runtime_timestamp_invalid")
        and certificate["source_run_id"] == planner["run_id"]
        and certificate["route_validation"]["context_snapshot_id"]
        == planner["context_snapshot_id"] == child["context_snapshot_id"],
        "e2e1_live_runtime_evaluator_freshness_invalid",
    )
    _require(
        (combined or (TARGET in constraints["must_delete"] and CANDIDATE in constraints["must_include"]))
        and receipt["request_id"] == request["request_id"]
        == proposal["compiled_request"]["source_request_id"]
        == planner["planning_request_id"] == child["planning_request_id"]
        and receipt["request_content_hash"] == _artifact_hash(request),
        "e2e1_live_runtime_request_binding_invalid",
    )
    _require(
        planner["method_executed"] == ("atomic_combined_repair_v1" if combined else "deterministic_context_aware_heuristic")
        and planner["execution_status"] == "COMPLETED"
        and planner["result_plan_id"] == child["plan_id"] == receipt["child_plan_id"]
        == proposal["child_plan_id"] == certificate["plan_id"] == diff["child_plan_id"]
        and planner["run_id"] == child["source_run_id"]
        and receipt["parent_plan_id"] == child["parent_plan_id"] == proposal["parent_plan_id"]
        == diff["parent_plan_id"],
        "e2e1_live_runtime_planner_binding_invalid",
    )
    for record in (planner, child, certificate):
        _require(record["content_hash"] == _artifact_hash({key: value for key, value in record.items() if key != "content_hash"}), "e2e1_live_runtime_artifact_hash_invalid")
    _require(
        child["content_hash"] == receipt["child_plan_content_hash"]
        == proposal["child_plan_content_hash"] == certificate["plan_content_hash"]
        and certificate["certificate_id"] == receipt["evaluation_id"] == proposal["certificate_id"]
        and certificate["content_hash"] == receipt["evaluation_content_hash"] == proposal["certificate_content_hash"]
        and certificate["comparison_eligibility"] == "eligible"
        and certificate["evaluation_status"] in {"PASSED", "PASSED_WITH_WARNINGS"}
        and diff["diff_id"] == proposal["diff_id"]
        and _artifact_hash(diff) == proposal["diff_content_hash"],
        "e2e1_live_runtime_evaluation_diff_binding_invalid",
    )
    _require(
        CANDIDATE in child["sequence"] and TARGET not in child["sequence"]
        and child["day_assignments"][CANDIDATE] == 5,
        "e2e1_live_runtime_semantic_result_invalid",
    )
    if combined:
        _require(request["schema_version"] == "atomic-combined-repair-request-v1"
            and request["request_family"] == "combined_typed_edit"
            and request["operations"] == receipt["operations"] == proposal["compiled_request"]["operations"],
            "e2e1_live_combined_request_binding_invalid")
        griffith = next((stop for stop in child["selected_stops"] if stop["stop_id"] == "griffith_observatory"), None)
        _require(griffith is not None and griffith["visit_duration_minutes"] == 90
            and griffith["duration_constraint"] == {"mode":"exact", "preferred_minutes":90, "minimum_minutes":90, "maximum_minutes":90},
            "e2e1_live_combined_duration_invalid")
        optimizer = receipt["optimizer_execution"]
        _verify_runtime_execution({
            "execution_mode": "deterministic_repair_pipeline", "runtime_execution": optimizer,
            "compiled_request": {"source_request_id": optimizer["request_id"]},
            "child_plan_id": optimizer["child_plan_id"], "child_plan_content_hash": optimizer["child_plan_content_hash"],
            "parent_plan_id": optimizer["parent_plan_id"], "certificate_id": optimizer["evaluation_id"],
            "certificate_content_hash": optimizer["evaluation_content_hash"], "diff_id": optimizer["plan_diff"]["diff_id"],
            "diff_content_hash": _artifact_hash(optimizer["plan_diff"]),
        })
        # Attribute edits intentionally form the optimizer's temporary baseline;
        # the final child is independently evaluated against the accepted parent.
        _require(optimizer["parent_plan_id"] == optimizer["repair_request"]["parent_plan_id"]
            and all(optimizer["child_plan"][key] == child[key]
                for key in ("selected_stops", "sequence", "day_assignments", "ordered_days"))
            and receipt["started_at"] <= optimizer["started_at"] <= optimizer["completed_at"] <= receipt["completed_at"],
            "e2e1_live_combined_optimizer_lineage_invalid")
        return receipt
    decisions = receipt["planner_trace"]
    _require(
        decisions["request_id"] == request["request_id"]
        and decisions["planner_run"] == planner
        and decisions["child_plan_id"] == child["plan_id"]
        and decisions["diff_id"] == diff["diff_id"]
        and decisions["failure_reasons"] == []
        and any(row.get("selected") is True and row.get("feasible") is True
                and row.get("day") == 5 and CANDIDATE in row.get("stop_sequence", [])
                and TARGET not in row.get("stop_sequence", []) and row.get("violations") == []
                for row in decisions["decision_records"]),
        "e2e1_live_runtime_decision_trace_invalid",
    )
    return receipt


def _verify_bundle(root: Path, manifest: dict[str, Any]) -> None:
    base = root / SCENARIO
    context = _load(base / "browser_context.json")
    prompt = context.get("approved_prompt")
    combined = prompt == COMBINED_PROMPT
    trace = _load(base / "trace.json")
    _require(
        set(context)
        == {
            "approved_prompt", "context_chips", "selected_day",
            "selected_stop_id", "session_id",
        },
        "e2e1_live_browser_context_schema_invalid",
    )
    copilot_http = _load(base / "product_copilot_http_io.json")
    copilot_request, copilot_response, copilot_session_id = _verify_http_record(
        copilot_http,
        "e2e1_live_copilot",
        base_url=manifest["base_url"],
        endpoint_suffix="/copilot/messages",
        response_format="sse",
    )
    stream_events = json.loads(
        (base / "copilot_stream_events.json").read_text(encoding="utf-8")
    )
    _require(
        stream_events
        == _sse_events(
            _load(base / "product_copilot_http_io.json")["response_body_utf8"]
        ),
        "e2e1_live_copilot_stream_projection_mismatch",
    )
    draft_http = _load(base / "product_draft_http_io.json")
    draft_request, draft_response, draft_session_id = _verify_http_record(
        draft_http,
        "e2e1_live_draft",
        base_url=manifest["base_url"],
        endpoint_suffix="/draft/operations",
    )
    preview_http = _load(base / "product_preview_http_io.json")
    preview_request, preview_response, preview_session_id = _verify_http_record(
        preview_http,
        "e2e1_live_preview",
        base_url=manifest["base_url"],
        endpoint_suffix="/preview",
    )
    provider = _load(base / "provider_exact_llm_io.json")
    records, provider_context, provider_authorization = _verify_provider_records(
        provider,
        allow_historical_schema=(
            manifest["authorization"]["execution_mode"]
            == "recorded_exact_output_replay"
        ),
    )
    provider_response = json.loads(records[-1]["response_body_utf8"])
    provider_projection = _provider_projection(provider_response)
    try:
        schema = OpenAIInterpretationSchemaV2 if provider_context["schema_version"] == "copilot-context-v2" else OpenAIInterpretationSchemaV1
        schema.model_validate(provider_projection, strict=True)
    except Exception as error:
        raise LiveE2E1VerificationError("e2e1_live_provider_projection_invalid") from error

    turn = copilot_response["turn"]
    stored_turn = _load(base / "copilot_turn.json")
    interpretation = _load(base / "typed_interpretation.json")
    _require(
        prompt in (EXPECTED_PROMPT, COMBINED_PROMPT)
        and context["selected_day"] == provider_context["selected_day"] == 5
        and context["selected_stop_id"] == provider_context["selected_stop_id"] == TARGET,
        "e2e1_live_browser_prompt_invalid",
    )
    _require(
        copilot_request["message"] == turn["user_message"] == provider_authorization["user_prompt"] == prompt,
        "e2e1_live_product_prompt_mismatch",
    )
    _require(
        set(copilot_request) == {"expected_revision", "client_message_id", "message"}
        and copilot_request["expected_revision"] == provider_context["session_revision"]
        and copilot_request["client_message_id"] == turn["client_message_id"]
        and turn == stored_turn,
        "e2e1_live_product_message_identity_invalid",
    )
    _require(turn["provider"] == "openai" and turn["model"] == "gpt-5.4-mini", "e2e1_live_turn_provider_invalid")
    _require(turn["interpretation"] == interpretation, "e2e1_live_turn_projection_mismatch")
    _verify_approved_interpretation(provider_projection, interpretation, prompt=prompt)

    operations = draft_response["operations"] if combined else [draft_response["operation"]]
    operation = operations[0]
    draft_operations = draft_request["operations"] if combined else [draft_request]
    first_request = draft_operations[0]
    _require(
        first_request["type"] == operation["type"] == "replace_nearby"
        and first_request["target"] == operation["target"] == TARGET
        and first_request["parameters"] == operation["parameters"] == {"candidate_id": CANDIDATE}
        and first_request["source"] == operation["source"] == "copilot",
        "e2e1_live_draft_binding_invalid",
    )
    if combined:
        _require(set(draft_request) == {"expected_revision", "operations"}
            and len(draft_operations) == len(operations) == 2
            and draft_response["session"]["draft"] == operations
            and draft_response["session"]["revision"] == draft_request["expected_revision"] + 1,
            "e2e1_live_combined_staging_invalid")
        for requested, staged in zip(draft_operations, operations, strict=True):
            _require(all(staged.get(key) == value for key, value in requested.items())
                and requested["source"] == "copilot", "e2e1_live_combined_staging_invalid")
        _require(draft_operations[1]["type"] == "set_stop_duration"
            and draft_operations[1]["target"] == "griffith_observatory"
            and draft_operations[1]["parameters"] == interpretation["intents"][1]["parameters"],
            "e2e1_live_combined_staging_invalid")
        staging = _load(base / "combined_staging_observation.json")
        _require(staging["initial_map"] == staging["staged_map"]
            and staging["initial_map"]["active_plan_id"] == provider_context["accepted_plan_id"]
            and staging["control_clicks"] == [{"text":"Add all to draft", "isTrusted":True},
                {"text":"Preview combined repair", "isTrusted":True}], "e2e1_live_combined_controls_invalid")
    _require(preview_request == {"expected_revision": draft_response["session"]["revision"]}, "e2e1_live_preview_request_invalid")
    proposal = preview_response["session"]["proposal"]
    _require(proposal["compiled_request"]["operations"] == [
        {key: value for key, value in row.items() if key != "created_at"} for row in operations],
        "e2e1_live_compiled_operations_mismatch")
    receipt = _verify_runtime_execution(proposal)
    _require(receipt == _load(base / "runtime_execution.json"), "e2e1_live_runtime_receipt_projection_mismatch")
    _verify_visible_metrics(_load(base / "visible_metrics.json"), proposal)
    browser = _load(base / "browser_assertions.json")
    _require(
        set(browser)
        == {
            "schema_version", "session_id", "client_message_id",
            "copilot_stream_request_id", "context_revision",
            "conversation_turn_id", "provider", "model", "intent",
            "draft_operation", "parent_plan_id", "child_plan_id",
            "child_plan_content_hash", "diff_id", "certificate_id",
            "rendered_map_plan_id", "rendered_geometry_hash",
            "rendered_marker_ids", "console_errors", "request_failures",
            "http_errors", "accept_or_keep_invoked", "assertions_passed",
        } | ({"intents", "draft_operations"} if combined else set())
        and browser.get("schema_version")
        == "e2e1-live-copilot-browser-assertions-v1"
        and browser.get("assertions_passed") is True,
        "e2e1_live_browser_assertions_schema_invalid",
    )
    _require(browser["intent"] == interpretation["intents"][0] and browser["draft_operation"] == operation,
        "e2e1_live_browser_semantics_mismatch")
    if combined:
        _require(browser["intents"] == interpretation["intents"] and browser["draft_operations"] == operations,
            "e2e1_live_browser_semantics_mismatch")
    map_ack = _load(base / "map_render_ack.json")
    evaluation = _load(base / "evaluation_summary.json")
    plan_diff = _load(base / "plan_diff.json")
    runtime_browser = _load(base / "runtime_browser_binding.json")
    started = _parse_timestamp(runtime_browser["preview_started_at"], "e2e1_live_runtime_timestamp_invalid")
    received = _parse_timestamp(runtime_browser["preview_received_at"], "e2e1_live_runtime_timestamp_invalid")
    _require(
        started <= _parse_timestamp(receipt["started_at"], "e2e1_live_runtime_timestamp_invalid")
        <= _parse_timestamp(receipt["completed_at"], "e2e1_live_runtime_timestamp_invalid") <= received,
        "e2e1_live_runtime_not_current_request",
    )
    geography = proposal["geography_plan"]
    digest = _independent_geometry_hash(geography)
    _require(
        geography["plan_id"] == receipt["child_plan_id"]
        and geography["content_hash"] == receipt["child_plan_content_hash"]
        and digest == runtime_browser["independent_geometry_hash"] == map_ack["rendered_geometry_hash"],
        "e2e1_live_raw_geography_binding_invalid",
    )
    _require(
        plan_diff == receipt["plan_diff"]
        and evaluation["certificate_id"] == receipt["evaluation_id"]
        and evaluation["diff_id"] == receipt["plan_diff"]["diff_id"]
        and sorted(map_ack["rendered_marker_ids"]) == sorted(
            feature["properties"]["stop_id"] for feature in geography["stops"]["features"]
        ),
        "e2e1_live_visible_artifact_projection_invalid",
    )
    reload_http = _load(base / "product_reload_http_io.json")
    _require(
        reload_http["method"] == "GET"
        and reload_http["url"] == f"{manifest['base_url']}/api/sessions/{copilot_session_id}"
        and reload_http["request_body_utf8"] == ""
        and reload_http["request_body_sha256"] == _hash_text("")
        and reload_http["response_status_code"] == 200
        and reload_http["response_body_sha256"] == _hash_text(reload_http["response_body_utf8"]),
        "e2e1_live_reload_http_binding_invalid",
    )
    restored = json.loads(reload_http["response_body_utf8"])["session"]
    _require(
        restored["session_id"] == copilot_session_id
        and restored["selected_stop_id"] == preview_response["session"]["selected_stop_id"] == CANDIDATE
        and restored["proposal"]["runtime_execution"] == receipt
        and _independent_geometry_hash(restored["proposal"]["geography_plan"]) == digest,
        "e2e1_live_restore_binding_invalid",
    )
    _verify_acceptance_observation(root, manifest, copilot_session_id, proposal, {
        "after_chat":copilot_response["session"], "after_stage":draft_response["session"],
        "after_preview":preview_response["session"], "after_reload":restored,
    })
    navigation = _load(base / "navigation_and_viewport_assertions.json")
    for dimensions in ((1440, 900), (1280, 800), (1024, 768), (390, 844)):
        width, height = dimensions
        readiness = _load(base / f"screenshot_preview_{width}x{height}.readiness.json")
        _require(
            readiness.get("stable_readiness_before_and_after") is True
            and readiness["viewport"] == {"width": width, "height": height}
            and 0.95 <= readiness["visible_tile_area_coverage_ratio"] <= 1
            and readiness["coverage_method"] == "loaded_tile_projected_union_area_v1"
            and readiness["container"]["width"] > 0 and readiness["container"]["height"] > 0
            and abs(readiness["canvas"]["width"] - readiness["container"]["width"]) <= 1
            and abs(readiness["canvas"]["height"] - readiness["container"]["height"]) <= 1
            and readiness["rendered_plan_id"] == receipt["child_plan_id"]
            and readiness["rendered_geometry_hash"] == digest
            and _parse_timestamp(readiness["map_ready_at"], "e2e1_live_map_readiness_timestamp_invalid")
            <= _parse_timestamp(readiness["screenshot_started_at"], "e2e1_live_map_readiness_timestamp_invalid")
            <= _parse_timestamp(readiness["screenshot_completed_at"], "e2e1_live_map_readiness_timestamp_invalid"),
            "e2e1_live_stable_screenshot_readiness_invalid",
        )
    compare_ack = _load(base / "compare_map_render_ack.json")
    _require(
        compare_ack["rendered_map_plan_id"] == receipt["child_plan_id"]
        and compare_ack["rendered_plan_content_hash"] == receipt["child_plan_content_hash"]
        and compare_ack["rendered_geometry_hash"] == digest
        and sorted(compare_ack["rendered_marker_ids"]) == sorted(map_ack["rendered_marker_ids"]),
        "e2e1_live_compare_map_identity_mismatch",
    )
    _require(
        navigation["assertions_passed"] is True and navigation["failures"] == {}
        and navigation["expected"] == map_ack
        and navigation["independent_geometry_hash"] == digest
        and {"after_copilot_restore", "after_module_switching", "after_reload"} <= set(navigation["checkpoints"])
        and all(value == map_ack for value in navigation["checkpoints"].values()),
        "e2e1_live_navigation_identity_mismatch",
    )
    _require(
        trace["runtime_execution_id"] == receipt["execution_id"]
        and trace["runtime_execution_sha256"] == _sha256(base / "runtime_execution.json")
        and trace["runtime_request_content_hash"] == receipt["request_content_hash"]
        and trace["executed_method"] == receipt["planner_run"]["method_executed"]
        and trace["evaluation_executed"] is True,
        "e2e1_live_runtime_trace_binding_invalid",
    )
    _require(
        copilot_session_id == draft_session_id == preview_session_id
        and context.get("session_id") == copilot_session_id
        and browser.get("session_id") == copilot_session_id
        and trace.get("session_id") == copilot_session_id
        and copilot_response["session"]["session_id"] == copilot_session_id
        and draft_response["session"]["session_id"] == copilot_session_id
        and preview_response["session"]["session_id"] == copilot_session_id,
        "e2e1_live_product_session_identity_mismatch",
    )
    execution_mode = trace.get("execution_mode")
    _require(
        execution_mode in {"live_openai", "recorded_exact_output_replay"},
        "e2e1_live_execution_mode_invalid",
    )
    _require(
        execution_mode == manifest["authorization"]["execution_mode"],
        "e2e1_live_execution_mode_mismatch",
    )
    if execution_mode == "live_openai":
        before_transport = _load(base / "provider_request_authorized.json")
        _require(set(before_transport) == {"schema_version", "observed_at", "state", "request_body_utf8", "request_body_sha256", "provider_dispatch_or_completion_confirmed"}
                 and before_transport["schema_version"] == "e2e1-live-before-transport-v1"
                 and before_transport["state"] == "authorized_before_transport"
                 and before_transport["provider_dispatch_or_completion_confirmed"] is False
                 and before_transport["request_body_utf8"] == records[0]["request_body_utf8"]
                 and before_transport["request_body_sha256"] == records[0]["request_body_sha256"], "e2e1_live_pre_dispatch_capture_invalid")
        _require(_parse_timestamp(records[0]["started_at"], "e2e1_live_pre_dispatch_capture_invalid")
                 <= _parse_timestamp(before_transport["observed_at"], "e2e1_live_pre_dispatch_capture_invalid")
                 <= _parse_timestamp(records[0]["completed_at"], "e2e1_live_pre_dispatch_capture_invalid"), "e2e1_live_pre_dispatch_capture_invalid")
        _require(
            provider_context["session_id"] == copilot_session_id,
            "e2e1_live_provider_session_identity_mismatch",
        )
    else:
        server_log = (root / "server_log_excerpt.txt").read_text(encoding="utf-8")
        _require(
            "recorded_exact_provider_output_replayed=true" in server_log
            and "provider_calls_during_browser_replay=0" in server_log,
            "e2e1_live_replay_metadata_invalid",
        )
    _require(
        copilot_request["client_message_id"]
        == browser["client_message_id"]
        == trace["client_message_id"]
        == stream_events[0]["client_message_id"]
        and browser["copilot_stream_request_id"]
        == stream_events[0]["request_id"],
        "e2e1_live_stream_trace_identity_mismatch",
    )
    _require(
        trace["copilot_api_request_sha256"] == copilot_http["request_body_sha256"]
        and trace["copilot_api_response_sha256"] == copilot_http["response_body_sha256"]
        and trace["draft_api_request_sha256"] == draft_http["request_body_sha256"]
        and trace["draft_api_response_sha256"] == draft_http["response_body_sha256"]
        and trace["preview_api_request_sha256"] == preview_http["request_body_sha256"]
        and trace["preview_api_response_sha256"] == preview_http["response_body_sha256"]
        and trace["interpretation_sha256"] == _sha256(base / "typed_interpretation.json"),
        "e2e1_live_trace_hash_binding_invalid",
    )
    if execution_mode == "live_openai":
        _require(
            trace.get("provider_request_body_sha256")
            == records[0]["request_body_sha256"]
            and trace.get("provider_response_body_sha256")
            == records[0]["response_body_sha256"]
            and trace.get("provider_request_authorization")
            == provider_authorization,
            "e2e1_live_provider_authorization_binding_invalid",
        )
    else:
        _require(
            trace.get("provider_request_body_sha256") is None
            and trace.get("provider_response_body_sha256") is None
            and trace.get("provider_request_authorization") is None,
            "e2e1_live_replay_current_provider_binding_invalid",
        )
    _require(
        proposal["state"] == "eligible"
        and proposal["route_validation"]["publication_ready"] is True
        and proposal["route_validation"]["required_leg_count"]
        == proposal["route_validation"]["road_validated_leg_count"]
        and proposal["route_validation"]["fallback_leg_count"] == 0,
        "e2e1_live_proposal_invalid",
    )
    child = proposal["child_plan_id"]
    _require(
        child
        == browser["child_plan_id"]
        == map_ack["active_plan_id"]
        == map_ack["rendered_map_plan_id"]
        == evaluation["plan_id"]
        == trace["child_plan_id"]
        == trace["rendered_map_plan_id"]
        == plan_diff["child_plan_id"],
        "e2e1_live_child_identity_mismatch",
    )
    _require(
        proposal["child_plan_content_hash"]
        == browser["child_plan_content_hash"]
        == map_ack["active_plan_content_hash"]
        == evaluation["plan_content_hash"]
        == trace["child_plan_content_hash"],
        "e2e1_live_content_identity_mismatch",
    )
    _require(
        CANDIDATE in map_ack["rendered_marker_ids"]
        and TARGET not in map_ack["rendered_marker_ids"]
        and trace["geography_hash"]
        == trace["rendered_geometry_hash"]
        == map_ack["rendered_geometry_hash"],
        "e2e1_live_map_projection_invalid",
    )
    _require(
        browser["console_errors"] == []
        and browser["request_failures"] == []
        and browser["http_errors"] == []
        and browser["accept_or_keep_invoked"] is False
        and trace["accept_or_keep_invoked"] is False,
        "e2e1_live_browser_errors",
    )

    historical_raw_usage = provider_response.get("usage")
    _require(isinstance(historical_raw_usage, dict), "e2e1_live_usage_missing")
    details = historical_raw_usage.get("input_tokens_details")
    _require(isinstance(details, dict), "e2e1_live_usage_invalid")
    cached_tokens = details.get("cached_tokens", 0)
    _require(
        all(
            type(historical_raw_usage.get(key)) is int
            and historical_raw_usage[key] > 0
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        and type(cached_tokens) is int
        and 0 <= cached_tokens <= historical_raw_usage["input_tokens"]
        and historical_raw_usage["total_tokens"]
        == historical_raw_usage["input_tokens"] + historical_raw_usage["output_tokens"],
        "e2e1_live_usage_invalid",
    )
    historical_usage = {
        "input_tokens": historical_raw_usage["input_tokens"],
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": historical_raw_usage["input_tokens"] - cached_tokens,
        "output_tokens": historical_raw_usage["output_tokens"],
        "total_tokens": historical_raw_usage["total_tokens"],
    }
    historical_cost = (
        Decimal(historical_usage["uncached_input_tokens"]) * INPUT_RATE
        + Decimal(historical_usage["cached_input_tokens"]) * CACHED_INPUT_RATE
        + Decimal(historical_usage["output_tokens"]) * OUTPUT_RATE
    ) / Decimal(1_000_000)
    historical_cost_text = format(historical_cost, "f")
    expected_pricing_rates = {
        "currency": "USD",
        "input_usd_per_million_tokens": str(INPUT_RATE),
        "cached_input_usd_per_million_tokens": str(CACHED_INPUT_RATE),
        "output_usd_per_million_tokens": str(OUTPUT_RATE),
    }
    _require(
        set(trace)
        == TRACE_KEYS
        | ({"historical_provider_evidence"} if execution_mode == "recorded_exact_output_replay" else set())
        and trace.get("schema_version") == "e2e1-live-copilot-browser-trace-v1"
        and trace.get("authorization_scope") == "synthetic_california_coast_single_chat"
        and trace.get("browser_prompt_sha256") == _hash_text(prompt)
        and trace.get("provider") == "openai"
        and trace.get("model") == "gpt-5.4-mini"
        and trace.get("reasoning_effort") == "low"
        and trace.get("assertions_passed") is True,
        "e2e1_live_trace_schema_invalid",
    )
    historical_provider_evidence: dict[str, Any] | None = None
    if execution_mode == "live_openai":
        expected_usage = historical_usage
        expected_cost_text = historical_cost_text
        expected_attempts = 1
        _require(
            trace.get("provider_transport_attempts") == 1
            and trace.get("provider_calls_current_run") == 1,
            "e2e1_live_live_call_count_invalid",
        )
        expected_usage_file = {
            **expected_usage,
            **{key: value for key, value in expected_pricing_rates.items() if key != "currency"},
            "billed_cost_usd": expected_cost_text,
        }
    else:
        expected_usage = dict(ZERO_USAGE)
        expected_cost_text = "0"
        expected_attempts = 0
        historical_provider_evidence = trace.get("historical_provider_evidence")
        _require(
            isinstance(historical_provider_evidence, dict)
            and set(historical_provider_evidence) == HISTORICAL_PROVIDER_KEYS
            and historical_provider_evidence.get("schema_version")
            == "e2e1-live-historical-provider-evidence-v1"
            and historical_provider_evidence.get("provider_transport_attempts") == 1
            and isinstance(historical_provider_evidence.get("provider_io_source_path"), str)
            and bool(historical_provider_evidence["provider_io_source_path"])
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(historical_provider_evidence.get("provider_io_file_sha256")),
            )
            is not None
            and historical_provider_evidence.get("replayed_provider_io_file_sha256")
            == _sha256(base / "provider_exact_llm_io.json")
            and historical_provider_evidence.get("provider_io_content_hash")
            == provider["content_hash"]
            and historical_provider_evidence.get("provider_request_body_sha256")
            == records[0]["request_body_sha256"]
            and historical_provider_evidence.get("provider_response_body_sha256")
            == records[0]["response_body_sha256"]
            and historical_provider_evidence.get("request_authorization")
            == provider_authorization
            and historical_provider_evidence.get("usage") == historical_usage
            and historical_provider_evidence.get("billed_cost_usd")
            == historical_cost_text
            and trace.get("provider_transport_attempts") == 0
            and trace.get("provider_calls_current_run") == 0,
            "e2e1_live_replay_metadata_invalid",
        )
        expected_usage_file = {
            "schema_version": "e2e1-live-replay-usage-and-cost-v1",
            "execution_mode": "recorded_exact_output_replay",
            "current_run_provider_calls": 0,
            "current_run_usage": dict(ZERO_USAGE),
            "current_run_billed_cost_usd": "0",
            **{key: value for key, value in expected_pricing_rates.items() if key != "currency"},
            "historical_provider_evidence": historical_provider_evidence,
        }
        _verify_replay_current_run_accounting(
            trace=trace,
            manifest=manifest,
            usage_file=_load(base / "usage_and_cost.json"),
            historical_provider_evidence=historical_provider_evidence,
        )
    _require(
        all(trace.get(key) == value for key, value in expected_usage.items())
        and trace.get("billed_cost_usd") == expected_cost_text,
        "e2e1_live_trace_usage_mismatch",
    )
    expected_pricing = {
        **expected_pricing_rates,
        "billed_cost_usd": expected_cost_text,
    }
    expected_authorization = {
        "confirmed": execution_mode == "live_openai",
        "scope": "synthetic_california_coast_single_chat",
        "execution_mode": execution_mode,
        "current_run_provider_calls": expected_attempts,
        "prompt_sha256": _hash_text(prompt),
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
        "logical_message_cap": 1,
        "transport_attempt_cap": expected_attempts,
        "accept_or_keep_allowed": False,
        "exact_provider_request": (
            provider_authorization if execution_mode == "live_openai" else None
        ),
        "historical_provider_evidence": historical_provider_evidence,
    }
    _require(
        manifest.get("authorization") == expected_authorization,
        "e2e1_live_manifest_authorization_invalid",
    )
    _require(
        manifest["provider_transport_attempts"] == expected_attempts,
        "e2e1_live_manifest_attempts_mismatch",
    )
    _require(manifest["usage"] == expected_usage, "e2e1_live_manifest_usage_mismatch")
    _require(manifest["pricing"] == expected_pricing, "e2e1_live_manifest_pricing_mismatch")
    _require(_load(base / "usage_and_cost.json") == expected_usage_file, "e2e1_live_usage_file_mismatch")
    _require(manifest["trace"] == trace, "e2e1_live_manifest_trace_mismatch")


def _verify_command(command: object) -> None:
    _require(
        isinstance(command, list)
        and len(command) == 11
        and Path(command[0]).resolve() == Path(sys.executable).resolve()
        and command[1:8] == [
            "-m", "pytest", "tests/product_app/browser/test_e2e1_live_copilot_chat.py",
            "-q", "-p", "no:cacheprovider", "--basetemp",
        ]
        and isinstance(command[8], str)
        and bool(command[8])
        and command[9] == "--junitxml"
        and isinstance(command[10], str)
        and Path(command[10]).name == "pytest_junit.xml",
        "e2e1_live_command_binding_invalid",
    )


def _verify_runtime_environments(root: Path, manifest: dict[str, Any]) -> None:
    """Bind actual journey identities to this finalized server and pytest run."""
    from e2e1_runtime_environment import RuntimeEnvironmentError, validate_runtime_environment

    code = "e2e1_live_lifecycle_invalid"
    binding = manifest.get("run_lifecycle")
    _require(type(binding) is dict and set(binding) == {"artifact", "record"}
             and binding["artifact"] == "run_lifecycle.json", code)
    _require((root / "run_lifecycle.json").is_file(), code)
    receipt = _load(root / "run_lifecycle.json")
    fields = {"schema_version", "run_id", "execution_mode", "base_url", "process_id", "observed_at",
              "server_ready_at", "pytest_started_at", "pytest_completed_at", "server_stopped_at",
              "finalized_at", "finalized", "pytest_exit_code", "command"}
    _require(set(receipt) == fields and receipt == binding["record"]
             and receipt["schema_version"] == "e2e1-live-run-lifecycle-v1"
             and receipt["run_id"] == root.name and receipt["execution_mode"] == manifest["authorization"]["execution_mode"]
             and receipt["base_url"] == manifest["base_url"]
             and type(receipt["process_id"]) is int and receipt["process_id"] > 0
             and receipt["finalized"] is True and type(receipt["pytest_exit_code"]) is int
             and receipt["pytest_exit_code"] == manifest["pytest_exit_code"] == 0
             and receipt["command"] == manifest["command"]
             and Path(receipt["command"][-1]).resolve() == (root / "pytest_junit.xml").resolve(), code)
    times = [_parse_timestamp(receipt[key], code) for key in (
        "observed_at", "server_ready_at", "pytest_started_at", "pytest_completed_at", "server_stopped_at", "finalized_at")]
    _require(times == sorted(times) and times[-1] <= _parse_timestamp(manifest["generated_at"], code), code)
    try:
        junit = ET.parse(root / "pytest_junit.xml").getroot()
    except (ET.ParseError, OSError) as error:
        raise LiveE2E1VerificationError("e2e1_live_junit_invalid") from error
    cases = list(junit.iter("testcase"))
    suites = list(junit.iter("testsuite"))
    expected_node = "tests/product_app/browser/test_e2e1_live_copilot_chat.py::test_live_website_copilot_stages_and_renders_exact_evaluated_child"
    nodes = {case.get("classname", "").replace(".", "/") + ".py::" + case.get("name", "") for case in cases}
    _require(len(cases) == 1 and nodes == {expected_node} and len(suites) == 1
             and suites[0].get("tests") == "1" and all(suites[0].get(key) == "0" for key in ("errors", "failures", "skipped"))
             and not any(list(junit.iter(key)) for key in ("failure", "error", "skipped")), "e2e1_live_junit_invalid")
    observed, referenced = set(), set()
    screenshots = list(root.rglob("*.readiness.json"))
    _require(bool(screenshots), "e2e1_live_runtime_environment_missing")
    for path in screenshots:
        ready = _load(path)
        bound = ready.get("runtime_environment")
        _require(type(bound) is dict and set(bound) == {"artifact", "sha256", "test_nodeid", "browser"}, "e2e1_live_runtime_journey_binding_invalid")
        node = bound["test_nodeid"]
        _require(type(node) is str and node in nodes, "e2e1_live_runtime_journey_binding_invalid")
        expected_name = "runtime_environment_" + _hash_text(node)[:16] + ".json"
        _require(bound["artifact"] == expected_name, "e2e1_live_runtime_journey_binding_invalid")
        artifact = _safe(path.parent, expected_name)
        _require(artifact.is_file(), "e2e1_live_runtime_environment_missing")
        record = _load(artifact)
        try:
            validate_runtime_environment(record, run_id=receipt["run_id"], root=ROOT)
        except RuntimeEnvironmentError as error:
            raise LiveE2E1VerificationError("e2e1_live_runtime_environment_invalid") from error
        png = path.with_name(path.name.removesuffix(".readiness.json") + ".png")
        _require(bound["sha256"] == _sha256(artifact) and bound["browser"] == record["browser"]
                 and record["application_origin"] == manifest["base_url"]
                 and times[2] <= _parse_timestamp(record["observed_at"], code)
                 <= _parse_timestamp(record["completed_at"], code)
                 <= _parse_timestamp(ready.get("screenshot_started_at"), code)
                 <= _parse_timestamp(ready.get("screenshot_completed_at"), code) <= times[3]
                 and png.is_file() and png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "e2e1_live_runtime_receipt_binding_invalid")
        observed.add(node)
        referenced.add(artifact)
    _require(observed == nodes and referenced == set(root.rglob("runtime_environment_*.json")), "e2e1_live_runtime_journey_universe_invalid")


def _verify_acceptance_observation(root: Path, manifest: dict[str, Any], session_id: str,
                                   proposal: dict[str, Any], sessions: dict[str, Any]) -> None:
    base = root / SCENARIO
    code = "e2e1_live_acceptance_observation_invalid"
    observation = _load(base / "acceptance_observation.json")
    ledger = json.loads((base / "product_mutation_ledger.json").read_text(encoding="utf-8"))
    initial_http = _load(base / "product_initial_selection_http_io.json")
    _, initial_payload, initial_session_id = _verify_http_record(initial_http, "e2e1_live_initial_selection",
        base_url=manifest["base_url"], endpoint_suffix="/selection")
    expected = {"before_chat":initial_payload["session"]["accepted_plan_id"],
                **{name:session["accepted_plan_id"] for name, session in sessions.items()}}
    _require(set(observation) == {"schema_version", "session_id", "initial_plan_id", "initial_content_hash",
             "preview_parent_id", "preview_parent_content_hash", "accepted_plan_checkpoints", "acceptance_mutations", "accept_or_keep_invoked"}
             and observation["schema_version"] == "e2e1-live-acceptance-observation-v1"
             and observation["session_id"] == initial_session_id == session_id
             and observation["accepted_plan_checkpoints"] == expected
             and set(expected.values()) == {proposal["parent_plan_id"]}
             and observation["initial_plan_id"] == observation["preview_parent_id"] == proposal["parent_plan_id"]
             and observation["initial_content_hash"] == observation["preview_parent_content_hash"] == proposal["parent_plan_content_hash"], code)
    _require(type(ledger) is list and bool(ledger), code)
    for index, row in enumerate(ledger, 1):
        _require(type(row) is dict and set(row) == {"sequence", "method", "url", "observed_at", "request_body_utf8",
                 "request_body_sha256", "request_original_body_sha256", "credential_redactions", "response_status_code", "response_observed_at"}
                 and type(row["sequence"]) is int and row["sequence"] == index
                 and row["method"] in {"POST", "PUT", "PATCH", "DELETE"}
                 and type(row["url"]) is str and row["url"].startswith(manifest["base_url"] + "/api/")
                 and row["request_body_sha256"] == _hash_text(row["request_body_utf8"])
                 and row["credential_redactions"] == []
                 and row["request_original_body_sha256"] == row["request_body_sha256"]
                 and type(row["response_status_code"]) is int and 200 <= row["response_status_code"] < 300, code)
        _require(_parse_timestamp(row["observed_at"], code) <= _parse_timestamp(row["response_observed_at"], code)
                 <= _parse_timestamp(manifest["generated_at"], code), code)
    actual = [row for row in ledger if re.fullmatch(r"/api/sessions/[^/]+/(accept|keep-original|acceptance-decisions/[^/]+/rollback)", urlparse(row["url"]).path)]
    _require(observation["acceptance_mutations"] == actual == [] and observation["accept_or_keep_invoked"] is False, code)
    for filename in ("product_initial_selection_http_io.json", "product_copilot_http_io.json", "product_draft_http_io.json", "product_preview_http_io.json"):
        http = _load(base / filename)
        matches = [row for row in ledger if row["method"] == http["method"] and row["url"] == http["url"]
                   and row["request_body_sha256"] == http["request_body_sha256"]]
        _require(len(matches) == 1 and matches[0]["response_status_code"] == http["response_status_code"], code)


def _verify_replay_isolation(root: Path, manifest: dict[str, Any]) -> None:
    if manifest["authorization"]["execution_mode"] != "recorded_exact_output_replay":
        return
    from e2e1_provider_isolation import BOUNDARIES

    code = "e2e1_live_replay_isolation_invalid"
    path = root / "provider_isolation.json"
    _require(path.is_file(), code)
    record = _load(path)
    fields = {"schema_version", "run_id", "process_id", "scope", "guard_boundaries", "installed_at",
              "finalized_at", "finalized", "effective_adapter", "actual_adapter", "api_key_disabled",
              "app_initialized", "server_started", "server_stopped", "blocked_attempts", "sent_provider_calls",
              "allowed_local_connections", "failure_code"}
    lifecycle = manifest["run_lifecycle"]["record"]
    _require(set(record) == fields and record["schema_version"] == "e2e1-provider-isolation-v1"
             and record["run_id"] == root.name and record["process_id"] == lifecycle["process_id"]
             and type(record["process_id"]) is int and record["process_id"] > 0
             and record["scope"] == "application_server" and record["guard_boundaries"] == BOUNDARIES
             and record["effective_adapter"] == "recorded_exact_output_replay"
             and record["actual_adapter"] == "RecordedCopilotAdapter"
             and all(record[key] is True for key in ("finalized", "api_key_disabled", "app_initialized", "server_started", "server_stopped"))
             and record["blocked_attempts"] == [] and type(record["sent_provider_calls"]) is int
             and record["sent_provider_calls"] == 0 and type(record["allowed_local_connections"]) is int
             and record["allowed_local_connections"] > 0 and record["failure_code"] is None, code)
    _require(_parse_timestamp(lifecycle["observed_at"], code) <= _parse_timestamp(record["installed_at"], code)
             <= _parse_timestamp(lifecycle["server_ready_at"], code)
             <= _parse_timestamp(lifecycle["server_stopped_at"], code)
             <= _parse_timestamp(record["finalized_at"], code)
             <= _parse_timestamp(manifest["generated_at"], code), code)


def _verify_text_security(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".log"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        _require(
            not re.search(r"(?i)\bsk-(?:proj-)?[a-z0-9_-]{12,}", text),
            "e2e1_live_api_key_persisted",
        )
        _require(
            re.search(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}", text) is None,
            "e2e1_live_authorization_persisted",
        )
        for match in re.finditer(
            r"(?i)x-session-token[\"']?\s*[:=]\s*[\"']?([^\s\"',}\]]+)",
            text,
        ):
            _require(
                match.group(1).casefold() == "omitted_secret",
                "e2e1_live_session_token_persisted",
            )
        if path.suffix.lower() == ".json":
            try:
                structured = json.loads(text)
            except json.JSONDecodeError as error:
                raise LiveE2E1VerificationError("e2e1_live_json_invalid") from error
            _verify_structured_secrets(structured)


def _verify_structured_secrets(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            compact = re.sub(r"[^a-z0-9]", "", normalized)
            location = (*path, normalized)
            if compact in {
                "apikey", "openaiapikey", "cookie", "setcookie",
                "xsessiontoken", "sessiontoken", "mutationtoken", "authorizationheader",
            }:
                _require(item == "OMITTED_SECRET", "e2e1_live_structured_secret_persisted")
            elif compact == "authorization":
                _require(
                    item == "OMITTED_SECRET"
                    or (
                        isinstance(item, dict)
                        and set(item)
                        == {
                            "confirmed", "scope", "execution_mode",
                            "current_run_provider_calls", "prompt_sha256", "model",
                            "reasoning_effort", "logical_message_cap",
                            "transport_attempt_cap", "accept_or_keep_allowed",
                            "exact_provider_request", "historical_provider_evidence",
                        }
                    ),
                    "e2e1_live_structured_secret_persisted",
                )
            _verify_structured_secrets(item, path=location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_structured_secrets(item, path=(*path, str(index)))
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        _verify_structured_secrets(decoded, path=(*path, "decoded_json"))


def verify(output_dir: Path) -> dict[str, Any]:
    root = output_dir.resolve()
    manifest = _load(root / MANIFEST_NAME)
    expected_keys = {
        "schema_version", "generated_at", "authorization", "base_url", "command",
        "pytest_exit_code", "assertions_passed", "health", "logical_copilot_messages",
        "provider_transport_attempts", "usage", "pricing", "files", "source_inputs",
        "trace", "content_hash", "run_lifecycle",
    }
    _require(set(manifest) == expected_keys, "e2e1_live_manifest_fields_invalid")
    seed = dict(manifest)
    content_hash = seed.pop("content_hash")
    _require(content_hash == _canonical_sha256(seed), "e2e1_live_manifest_hash_invalid")
    generated_at = _parse_timestamp(
        manifest.get("generated_at"), "e2e1_live_generated_at_invalid"
    )
    base_url = re.fullmatch(
        r"http://127\.0\.0\.1:([1-9]\d{0,4})",
        str(manifest["base_url"]),
    )
    authorization = manifest.get("authorization")
    authorization_keys = {
        "confirmed", "scope", "execution_mode", "current_run_provider_calls",
        "prompt_sha256", "model", "reasoning_effort", "logical_message_cap",
        "transport_attempt_cap", "accept_or_keep_allowed",
        "exact_provider_request", "historical_provider_evidence",
    }
    _require(
        isinstance(authorization, dict)
        and set(authorization) == authorization_keys
        and authorization.get("execution_mode")
        in {"live_openai", "recorded_exact_output_replay"},
        "e2e1_live_manifest_authorization_schema_invalid",
    )
    live_provider = authorization["execution_mode"] == "live_openai"
    expected_attempts = 1 if live_provider else 0
    _require(
        manifest["schema_version"] == "e2e1-live-copilot-file-io-manifest-v2"
        and base_url is not None
        and int(base_url.group(1)) <= 65_535
        and manifest["pytest_exit_code"] == 0
        and manifest["assertions_passed"] is True
        and manifest["logical_copilot_messages"] == 1
        and manifest["provider_transport_attempts"] == expected_attempts
        and authorization.get("confirmed") is live_provider
        and authorization.get("current_run_provider_calls") == expected_attempts
        and authorization.get("transport_attempt_cap") == expected_attempts,
        "e2e1_live_manifest_state_invalid",
    )
    _require(
        isinstance(manifest["health"].get("build_id"), str)
        and bool(manifest["health"]["build_id"])
        and manifest["health"].get("map") == "maplibre_ready"
        and manifest["health"].get("routing") == "runtime_osrm_ready"
        and manifest["health"].get("openai") == "openai_configured",
        "e2e1_live_health_invalid",
    )
    _verify_command(manifest["command"])
    declared = set()
    for row in manifest["files"]:
        _require(set(row) == {"path", "size_bytes", "sha256", "exact_content_location"}, "e2e1_live_file_row_invalid")
        path = _safe(root, row["path"])
        _require(
            path.is_file()
            and row["path"] == row["exact_content_location"]
            and path.stat().st_size == row["size_bytes"]
            and _sha256(path) == row["sha256"],
            "e2e1_live_file_binding_invalid",
        )
        declared.add(row["path"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {MANIFEST_NAME, REPORT_NAME}
    }
    _require(actual == declared, "e2e1_live_file_inventory_invalid")
    source_rows = manifest["source_inputs"]
    _require(
        isinstance(source_rows, list)
        and [row.get("path") for row in source_rows] == list(SOURCE_INPUTS),
        "e2e1_live_source_universe_invalid",
    )
    for row in source_rows:
        _require(
            isinstance(row, dict) and set(row) == SOURCE_KEYS,
            "e2e1_live_source_row_invalid",
        )
        path = _safe(ROOT, row["path"])
        _require(
            path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and _sha256(path) == row["sha256"]
            and datetime.fromtimestamp(path.stat().st_mtime, generated_at.tzinfo) <= generated_at,
            "e2e1_live_source_binding_invalid",
        )
    pytest_output = (root / "pytest_output.txt").read_text(encoding="utf-8").strip()
    pytest_lines = pytest_output.splitlines()
    _require(
        sum(line.startswith("1 passed in ") for line in pytest_lines) == 1
        and not any(" failed" in line or " error" in line for line in pytest_lines),
        "e2e1_live_pytest_output_invalid",
    )
    _verify_text_security(root)
    _verify_runtime_environments(root, manifest)
    _verify_replay_isolation(root, manifest)
    _verify_bundle(root, manifest)

    runner_path = ROOT / "scripts" / "run_e2e1_live_copilot_chat.py"
    spec = importlib.util.spec_from_file_location("run_e2e1_live_copilot_chat", runner_path)
    _require(spec is not None and spec.loader is not None, "e2e1_live_report_renderer_missing")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    _require(
        (root / REPORT_NAME).read_text(encoding="utf-8") == runner.render_report(root, manifest),
        "e2e1_live_report_mismatch",
    )
    return {
        "verified": True,
        "manifest_content_hash": content_hash,
        "manifest_sha256": _sha256(root / MANIFEST_NAME),
        "report_sha256": _sha256(root / REPORT_NAME),
        "artifact_files": len(manifest["files"]),
        "source_inputs": len(manifest["source_inputs"]),
        "logical_copilot_messages": manifest["logical_copilot_messages"],
        "provider_transport_attempts": manifest["provider_transport_attempts"],
        "usage": manifest["usage"],
        "billed_cost_usd": manifest["pricing"]["billed_cost_usd"],
        "child_plan_id": manifest["trace"]["child_plan_id"],
        "rendered_map_plan_id": manifest["trace"]["rendered_map_plan_id"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(verify(parse_args().output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
