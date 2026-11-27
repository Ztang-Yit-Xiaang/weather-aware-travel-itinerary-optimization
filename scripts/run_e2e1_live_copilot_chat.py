"""Run the one authorized billed E2E-1 website-Copilot vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from e2e1_source_universe import live_source_inputs  # noqa: E402
from run_live_copilot_prompt_evaluation_v2 import ExactIOTransport  # noqa: E402
from run_product_app import load_local_environment  # noqa: E402

from itinerary_system.product_app.api import create_product_app  # noqa: E402
from itinerary_system.product_app.copilot import validate_copilot_interpretation  # noqa: E402
from itinerary_system.product_app.copilot_prompt import DEVELOPER_INSTRUCTIONS  # noqa: E402
from itinerary_system.product_app.models import (  # noqa: E402
    CopilotContextV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
)
from itinerary_system.product_app.openai_copilot import (  # noqa: E402
    OpenAICopilotAdapter,
    OpenAIInterpretationSchemaV1,
    OpenAIInterpretationSchemaV2,
    _canonicalize_provider_copy,
    _to_dataclass,
)

HOST = "127.0.0.1"
MODEL = "gpt-5.4-mini"
EFFORT = "low"
APPROVED_PROMPT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, preserving the rest of the seven-day trip."
)
COMBINED_PROMPT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes."
)
BROWSER_TEST = "tests/product_app/browser/test_e2e1_live_copilot_chat.py"
SCENARIO = "live_copilot_chat"
MANIFEST_NAME = "file_io_manifest.json"
REPORT_NAME = "exact_io_report.md"
INPUT_USD_PER_MILLION = Decimal("0.75")
CACHED_INPUT_USD_PER_MILLION = Decimal("0.075")
OUTPUT_USD_PER_MILLION = Decimal("4.50")
SOURCE_INPUTS = (*live_source_inputs(BROWSER_TEST), "scripts/e2e1_provider_isolation.py")
AUTHORIZED_NORMALIZED_CONTEXT_SHA256 = "a9a159e36b062e81d288caf249f3b9f2a3963d50e50a2b9c37cb04650e8a12f7"
PROVIDER_ENDPOINT = "https://api.openai.com/v1/responses"
PROVIDER_REQUEST_FIELDS = {
    "input",
    "instructions",
    "max_output_tokens",
    "model",
    "reasoning",
    "store",
    "text",
    "tools",
}
TRANSPORT_ATTEMPT_FIELDS = {
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
CONTEXT_FIELDS = {
    "accepted_plan_content_hash",
    "capabilities",
    "conversation_memory",
    "accepted_plan_id",
    "allowed_alternative_ids",
    "allowed_candidate_ids",
    "allowed_days",
    "allowed_segment_ids",
    "allowed_stop_ids",
    "draft_operations",
    "evaluated_proposal",
    "known_places",
    "run_id",
    "schema_version",
    "selected_alternative_id",
    "selected_candidate_id",
    "selected_day",
    "selected_segment_id",
    "selected_stop_id",
    "session_id",
    "session_revision",
    "trip_id",
}
ZERO_USAGE = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "uncached_input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}
AUTHORIZED_HISTORICAL_OUTPUT_SCHEMA_SHA256 = {
    "0e3115581815d32954bea65cfee3534a1ea7887e448f4c385e5de8da7ac41027",
}


class LiveE2E1Error(RuntimeError):
    """Stable live E2E-1 runner failure."""


class RecordedCopilotAdapter:
    """Replay one already-captured strict provider result without network access."""

    provider_name = "openai"
    model = MODEL

    def __init__(self, provider_io: dict[str, Any]) -> None:
        records = provider_io.get("records")
        if not isinstance(records, list) or len(records) != 1:
            raise LiveE2E1Error("e2e1_live_replay_record_invalid")
        response = json.loads(records[0]["response_body_utf8"])
        projection = _provider_projection(response)
        schema = (
            OpenAIInterpretationSchemaV2
            if projection.get("schema_version") == "copilot-interpretation-v2"
            else OpenAIInterpretationSchemaV1
        )
        parsed = schema.model_validate(projection, strict=True)
        self._interpretation = _to_dataclass(parsed)
        request = json.loads(records[0]["request_body_utf8"])
        self._prompt = _provider_input_sections(request["input"])[2]

    async def interpret(
        self,
        *,
        context: CopilotContextV1,
        history: tuple[CopilotTurnV1, ...],
        message: str,
    ) -> CopilotInterpretationV1:
        del history
        if message != self._prompt:
            raise LiveE2E1Error("e2e1_live_replay_prompt_mismatch")
        interpretation = self._interpretation
        if message == COMBINED_PROMPT:
            from dataclasses import replace

            from itinerary_system.product_app.copilot import request_semantic_guard

            combined = request_semantic_guard(context, message)
            if combined is None or combined.state != "proposal_ready" or combined.intents != interpretation.intents:
                raise LiveE2E1Error("e2e1_live_replay_combined_semantics_invalid")
            interpretation = replace(interpretation, highlights=combined.highlights)
        interpretation = _canonicalize_provider_copy(interpretation, context=context, message=message)
        return validate_copilot_interpretation(
            context=context,
            interpretation=interpretation,
        )


def _provider_projection(response: dict[str, Any]) -> dict[str, Any]:
    texts = [
        content.get("text")
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "message"
        for content in item.get("content", [])
        if isinstance(content, dict) and content.get("type") == "output_text"
    ]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise LiveE2E1Error("e2e1_live_replay_output_invalid")
    projection = json.loads(texts[0])
    if not isinstance(projection, dict):
        raise LiveE2E1Error("e2e1_live_replay_output_invalid")
    return projection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_prompt_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _provider_input_sections(value: object) -> tuple[dict[str, Any], list[Any], str]:
    if not isinstance(value, str):
        raise LiveE2E1Error("e2e1_live_provider_input_invalid")
    values: list[Any] = []
    cursor = 0
    for index, name in enumerate(("workspace_context", "recent_conversation", "user_request")):
        opening = f"<{name}>\n"
        closing = f"\n</{name}>"
        if not value.startswith(opening, cursor):
            raise LiveE2E1Error("e2e1_live_provider_input_invalid")
        content_start = cursor + len(opening)
        content_end = value.find(closing, content_start)
        if content_end < 0:
            raise LiveE2E1Error("e2e1_live_provider_input_invalid")
        raw = value[content_start:content_end]
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LiveE2E1Error("e2e1_live_provider_input_invalid") from exc
        if raw != _canonical_prompt_json(decoded):
            raise LiveE2E1Error("e2e1_live_provider_input_not_canonical")
        values.append(decoded)
        cursor = content_end + len(closing)
        if index < 2:
            if not value.startswith("\n", cursor):
                raise LiveE2E1Error("e2e1_live_provider_input_invalid")
            cursor += 1
    if cursor != len(value):
        raise LiveE2E1Error("e2e1_live_provider_input_invalid")
    context, history, message = values
    if not isinstance(context, dict) or not isinstance(history, list) or not isinstance(message, str):
        raise LiveE2E1Error("e2e1_live_provider_input_invalid")
    return context, history, message


def _validate_authorized_context(
    context: dict[str, Any],
    *,
    expected_session_id: str | None,
    allow_historical_schema: bool = False,
) -> None:
    historical = allow_historical_schema and context.get("schema_version") == "copilot-context-v1"
    expected_fields = (
        CONTEXT_FIELDS - {"accepted_plan_content_hash", "capabilities", "conversation_memory"}
        if historical
        else CONTEXT_FIELDS
    )
    if set(context) != expected_fields:
        raise LiveE2E1Error("e2e1_live_provider_context_fields_invalid")
    session_id = context.get("session_id")
    expected_stop_ids = [
        "bixby_creek_bridge_viewpoint",
        "golden_gate_bridge",
        "griffith_observatory",
        "kings_canyon_national_park",
        "pfeiffer_big_sur_state_park",
        "point_lobos_state_natural_reserve",
        "sequoia_national_park",
        "stanford_university_main_quad",
        "stearns_wharf",
        "yosemite_national_park",
    ]
    if not historical:
        expected_stop_ids = [
            "golden_gate_bridge",
            "stanford_university_main_quad",
            "yosemite_national_park",
            "sequoia_national_park",
            "kings_canyon_national_park",
            "bixby_creek_bridge_viewpoint",
            "stearns_wharf",
            "griffith_observatory",
        ]
    if (
        context.get("schema_version") != ("copilot-context-v1" if historical else "copilot-context-v2")
        or context.get("run_id") != "california_coast_coherent_demo_v1"
        or context.get("trip_id") != "california_coast_approved_seven_day"
        or not isinstance(session_id, str)
        or re.fullmatch(r"session_[0-9a-f]{32}", session_id) is None
        or (expected_session_id is not None and session_id != expected_session_id)
        or type(context.get("session_revision")) is not int
        or context.get("session_revision") != 1
        or not isinstance(context.get("accepted_plan_id"), str)
        or re.fullmatch(r"plan_[0-9a-f]{16}", context["accepted_plan_id"]) is None
        or context.get("selected_day") != 5
        or context.get("selected_stop_id") != "bixby_creek_bridge_viewpoint"
        or context.get("selected_segment_id") is not None
        or context.get("selected_alternative_id") is not None
        or context.get("selected_candidate_id") is not None
        or context.get("draft_operations") != []
        or context.get("evaluated_proposal") is not None
        or context.get("allowed_days") != list(range(1, 8))
        or context.get("allowed_candidate_ids") != ["pfeiffer_big_sur_state_park", "point_lobos_state_natural_reserve"]
        or context.get("allowed_stop_ids") != expected_stop_ids
    ):
        raise LiveE2E1Error("e2e1_live_provider_context_invalid")
    if not historical:
        from copy import deepcopy

        from itinerary_system.product_app.copilot import _validate_context
        from itinerary_system.product_app.models import CopilotContextV2

        typed = deepcopy(context)
        for key in (
            "allowed_days",
            "allowed_stop_ids",
            "allowed_segment_ids",
            "allowed_candidate_ids",
            "allowed_alternative_ids",
            "known_places",
            "draft_operations",
        ):
            typed[key] = tuple(typed[key])
        for place in typed["known_places"]:
            for key in ("aliases", "categories"):
                place[key] = tuple(place[key])
        memory = typed["conversation_memory"]
        if memory != {
            "schema_version": "copilot-memory-v1",
            "plan_id": context["accepted_plan_id"],
            "plan_content_hash": context["accepted_plan_content_hash"],
            "pending_clarification": None,
            "resolved_ids": {"stop_ids": [], "candidate_ids": [], "day_ids": [], "segment_ids": []},
            "proposed_preferences": [],
            "last_proposal": None,
        }:
            raise LiveE2E1Error("e2e1_live_provider_context_memory_invalid")
        if typed["capabilities"].get("schema_version") != "copilot-capabilities-v2":
            raise LiveE2E1Error("e2e1_live_provider_context_invalid")
        for key in ("candidate_replacements", "candidate_day_readiness", "stop_positions"):
            typed["capabilities"][key] = tuple(typed["capabilities"][key])
        typed["capabilities"]["executable_combinations"] = tuple(
            {**row, "operations": tuple(row["operations"])} for row in typed["capabilities"]["executable_combinations"]
        )
        memory["resolved_ids"] = {key: tuple(value) for key, value in memory["resolved_ids"].items()}
        memory["proposed_preferences"] = ()
        try:
            _validate_context(CopilotContextV2(**typed))
        except (ValueError, TypeError, KeyError):
            raise LiveE2E1Error("e2e1_live_provider_context_invalid") from None
    alternatives = context.get("allowed_alternative_ids")
    segments = context.get("allowed_segment_ids")
    places = context.get("known_places")
    if (
        not isinstance(alternatives, list)
        or len(alternatives) != 2
        or any(not isinstance(item, str) or re.fullmatch(r"plan_[0-9a-f]{16}", item) is None for item in alternatives)
        or not isinstance(segments, list)
        or not segments
        or any(not isinstance(item, str) or not item for item in segments)
        or not isinstance(places, list)
        or not places
    ):
        raise LiveE2E1Error("e2e1_live_provider_context_invalid")
    place_ids: list[str] = []
    for place in places:
        if (
            not isinstance(place, dict)
            or set(place) != {"aliases", "categories", "name", "place_id", "region_id", "route_status"}
            or not isinstance(place.get("aliases"), list)
            or any(not isinstance(item, str) for item in place["aliases"])
            or not isinstance(place.get("categories"), list)
            or not place["categories"]
            or any(not isinstance(item, str) for item in place["categories"])
            or not isinstance(place.get("name"), str)
            or not place["name"]
            or not isinstance(place.get("place_id"), str)
            or not place["place_id"]
            or (place.get("region_id") is not None and not isinstance(place.get("region_id"), str))
            or place.get("route_status")
            not in {
                "accepted_stop",
                "registered_route_candidate",
                "discovery_only_route_validation_required",
            }
        ):
            raise LiveE2E1Error("e2e1_live_provider_context_place_invalid")
        place_ids.append(place["place_id"])
    if (
        len(place_ids) != len(set(place_ids))
        or not set(expected_stop_ids).issubset(place_ids)
        or "ca_region_redwood" not in place_ids
    ):
        raise LiveE2E1Error("e2e1_live_provider_context_place_invalid")
    if not historical:
        normalized = {**context, "session_id": "<SESSION_ID>"}
        if _text_sha256(_canonical_prompt_json(normalized)) != AUTHORIZED_NORMALIZED_CONTEXT_SHA256:
            raise LiveE2E1Error("e2e1_live_provider_context_hash_invalid")


def _provider_request_authorization(
    request_body: bytes | str,
    *,
    expected_session_id: str | None = None,
    allow_historical_schema: bool = False,
) -> dict[str, Any]:
    if isinstance(request_body, bytes):
        try:
            request_text = request_body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise LiveE2E1Error("e2e1_live_provider_request_invalid") from exc
    elif isinstance(request_body, str):
        request_text = request_body
    else:
        raise LiveE2E1Error("e2e1_live_provider_request_invalid")
    try:
        request = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise LiveE2E1Error("e2e1_live_provider_request_invalid") from exc
    current_schema = to_strict_json_schema(OpenAIInterpretationSchemaV2)
    expected_text = {
        "format": {
            "type": "json_schema",
            "name": "OpenAIInterpretationSchemaV2",
            "schema": current_schema,
            "strict": True,
        },
        "verbosity": "low",
    }
    actual_text = request.get("text") if isinstance(request, dict) else None
    actual_format = actual_text.get("format") if isinstance(actual_text, dict) else None
    actual_schema = actual_format.get("schema") if isinstance(actual_format, dict) else None
    actual_schema_sha256 = _canonical_sha256(actual_schema) if isinstance(actual_schema, dict) else None
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
    if (
        not isinstance(request, dict)
        or set(request) != PROVIDER_REQUEST_FIELDS
        or request.get("model") != MODEL
        or not (
            request.get("instructions") == DEVELOPER_INSTRUCTIONS
            or (
                allow_historical_schema
                and isinstance(request.get("instructions"), str)
                and _text_sha256(request["instructions"])
                == "6685aabbddd7a94d2d594331a5d9c780953c838bdb2f80a92e14f0ba9b11d711"
            )
        )
        or request.get("max_output_tokens") != 1_200
        or request.get("reasoning") != {"effort": EFFORT, "context": "current_turn"}
        or request.get("store") is not False
        or request.get("tools") != []
        or (request.get("text") != expected_text and not historical_text_allowed)
    ):
        raise LiveE2E1Error("e2e1_live_provider_request_invalid")
    context, history, message = _provider_input_sections(request.get("input"))
    if history != [] or message not in (APPROVED_PROMPT, COMBINED_PROMPT):
        raise LiveE2E1Error("e2e1_live_provider_request_invalid")
    _validate_authorized_context(
        context, expected_session_id=expected_session_id, allow_historical_schema=allow_historical_schema
    )
    if context["schema_version"] == "copilot-context-v2" and (
        request["instructions"] != DEVELOPER_INSTRUCTIONS or request["text"] != expected_text
    ):
        raise LiveE2E1Error("e2e1_live_provider_request_invalid")
    input_text = request["input"]
    return {
        "schema_version": "e2e1-live-provider-request-authorization-v1",
        "request_body_sha256": _text_sha256(request_text),
        "request_body_size_bytes": len(request_text.encode("utf-8")),
        "request_body_exact_location": ("live_copilot_chat/provider_exact_llm_io.json#records[0].request_body_utf8"),
        "request_fields": sorted(PROVIDER_REQUEST_FIELDS),
        "instructions": request["instructions"],
        "instructions_sha256": _text_sha256(request["instructions"]),
        "input_sha256": _text_sha256(input_text),
        "workspace_context": context,
        "workspace_context_sha256": _canonical_sha256(context),
        "recent_conversation": history,
        "recent_conversation_sha256": _canonical_sha256(history),
        "user_prompt": message,
        "user_prompt_sha256": _text_sha256(message),
        "model": MODEL,
        "reasoning": {"effort": EFFORT, "context": "current_turn"},
        "store": False,
        "tools": [],
        "max_output_tokens": 1_200,
        "text": request["text"],
        "text_sha256": _canonical_sha256(request["text"]),
        "output_schema_sha256": actual_schema_sha256,
        "output_schema_current": actual_schema == current_schema,
        "session_id": context["session_id"],
    }


def _one_provider_attempt_validator(
    authorization_path: Path | None = None,
    *,
    expected_prompt: str = APPROVED_PROMPT,
):
    authorized = False

    def validate(request_body: bytes) -> None:
        nonlocal authorized
        if authorized:
            raise LiveE2E1Error("e2e1_live_attempt_cap_exceeded")
        authorization = _provider_request_authorization(request_body)
        if authorization["user_prompt"] != expected_prompt:
            raise LiveE2E1Error("e2e1_live_selected_prompt_mismatch")
        authorized = True
        if authorization_path is not None:
            _write_json(
                authorization_path,
                {
                    "schema_version": "e2e1-live-before-transport-v1",
                    "observed_at": datetime.now(UTC).isoformat(),
                    "state": "authorized_before_transport",
                    "request_body_utf8": request_body.decode("utf-8"),
                    "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "provider_dispatch_or_completion_confirmed": False,
                },
            )

    return validate


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _port_available(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((HOST, port)) != 0


def _health(base_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as response:
            value = json.loads(response.read(65_537).decode())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _wait_ready(thread: threading.Thread, base_url: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise LiveE2E1Error("e2e1_live_server_exited_before_ready")
        health = _health(base_url)
        components = health.get("components", {}) if health else {}
        if (
            health
            and health.get("ready") is True
            and components.get("map", {}).get("status") == "ready"
            and components.get("routing", {}).get("status") == "ready"
            and components.get("openai", {}).get("code") == "openai_configured"
        ):
            return health
        time.sleep(0.25)
    raise LiveE2E1Error("e2e1_live_server_not_ready")


def _usage(records: list[dict[str, Any]]) -> dict[str, int]:
    if len(records) != 1:
        raise LiveE2E1Error("e2e1_live_attempt_cap_invalid")
    successful = [row for row in records if row.get("response_status_code") == 200]
    if len(successful) != 1:
        raise LiveE2E1Error("e2e1_live_completed_response_universe_invalid")
    usage = json.loads(successful[0]["response_body_utf8"]).get("usage")
    if not isinstance(usage, dict):
        raise LiveE2E1Error("e2e1_live_usage_missing")
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    result = {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": cached,
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    input_tokens = result["input_tokens"]
    result["uncached_input_tokens"] = (
        input_tokens - cached if type(input_tokens) is int and type(cached) is int else None
    )
    if (
        not all(
            type(result[key]) is int and result[key] > 0 for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        or type(cached) is not int
        or cached < 0
        or cached > result["input_tokens"]
        or type(result["uncached_input_tokens"]) is not int
        or result["uncached_input_tokens"] < 0
        or result["total_tokens"] != result["input_tokens"] + result["output_tokens"]
    ):
        raise LiveE2E1Error("e2e1_live_usage_invalid")
    return result


def _cost(usage: dict[str, int]) -> str:
    if (
        set(usage) != set(ZERO_USAGE)
        or any(type(usage[key]) is not int or usage[key] < 0 for key in ZERO_USAGE)
        or usage["cached_input_tokens"] > usage["input_tokens"]
        or usage["uncached_input_tokens"] != usage["input_tokens"] - usage["cached_input_tokens"]
        or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise LiveE2E1Error("e2e1_live_usage_invalid")
    value = (
        Decimal(usage["uncached_input_tokens"]) * INPUT_USD_PER_MILLION
        + Decimal(usage["cached_input_tokens"]) * CACHED_INPUT_USD_PER_MILLION
        + Decimal(usage["output_tokens"]) * OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)
    return format(value, "f")


def _validate_transport(
    records: object,
    *,
    expected_session_id: str | None = None,
    allow_historical_schema: bool = False,
) -> dict[str, Any]:
    if not isinstance(records, list) or len(records) != 1:
        raise LiveE2E1Error("e2e1_live_attempt_cap_invalid")
    row = records[0]
    if not isinstance(row, dict) or set(row) != TRANSPORT_ATTEMPT_FIELDS:
        raise LiveE2E1Error("e2e1_live_transport_record_invalid")
    try:
        started_at = datetime.fromisoformat(row["started_at"])
        completed_at = datetime.fromisoformat(row["completed_at"])
    except (TypeError, ValueError) as exc:
        raise LiveE2E1Error("e2e1_live_transport_record_invalid") from exc
    latency = row.get("latency_ms")
    request_text = row.get("request_body_utf8")
    response_text = row.get("response_body_utf8")
    if (
        row.get("schema_version") != "copilot-live-exact-transport-attempt-v1"
        or row.get("sequence") != 1
        or row.get("case_id") != "e2e1_live_website_chat"
        or row.get("effort") != EFFORT
        or row.get("case_attempt") != 1
        or started_at.tzinfo is None
        or completed_at.tzinfo is None
        or completed_at < started_at
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or latency < 0
        or row.get("method") != "POST"
        or row.get("url") != PROVIDER_ENDPOINT
        or not isinstance(request_text, str)
        or row.get("request_body_sha256") != _text_sha256(request_text)
        or row.get("request_headers_persisted") is not False
        or row.get("response_headers_persisted") is not False
        or row.get("authorization_persisted") is not False
    ):
        raise LiveE2E1Error("e2e1_live_transport_record_invalid")
    authorization = _provider_request_authorization(
        request_text,
        expected_session_id=expected_session_id,
        allow_historical_schema=allow_historical_schema,
    )
    if (
        row.get("response_status_code") != 200
        or not isinstance(response_text, str)
        or row.get("response_body_sha256") != _text_sha256(response_text)
        or row.get("transport_error_code") is not None
    ):
        raise LiveE2E1Error("e2e1_live_provider_response_not_completed")
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LiveE2E1Error("e2e1_live_provider_response_invalid") from exc
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise LiveE2E1Error("e2e1_live_provider_response_invalid")
    _usage(records)
    return authorization


def _browser_transport_gate(
    exit_code: int,
    records: list[dict[str, Any]],
    *,
    expected_session_id: str | None = None,
) -> int | None:
    if exit_code != 0:
        return exit_code
    _validate_transport(records, expected_session_id=expected_session_id)
    return None


def _build_trace(
    scenario_dir: Path,
    provider_io: dict[str, Any],
    usage: dict[str, int],
    *,
    request_authorization: dict[str, Any],
    execution_mode: str = "live_openai",
    historical_provider_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    browser = json.loads((scenario_dir / "browser_assertions.json").read_text(encoding="utf-8"))
    copilot_io = json.loads((scenario_dir / "product_copilot_http_io.json").read_text(encoding="utf-8"))
    draft_io = json.loads((scenario_dir / "product_draft_http_io.json").read_text(encoding="utf-8"))
    preview_io = json.loads((scenario_dir / "product_preview_http_io.json").read_text(encoding="utf-8"))
    proposal = json.loads(preview_io["response_body_utf8"])["session"]["proposal"]
    records = provider_io["records"]
    live_provider = execution_mode == "live_openai"
    browser_session_id = browser.get("session_id")
    if not isinstance(browser_session_id, str) or (
        live_provider and request_authorization.get("session_id") != browser_session_id
    ):
        raise LiveE2E1Error("e2e1_live_browser_session_binding_invalid")
    trace = {
        "schema_version": "e2e1-live-copilot-browser-trace-v1",
        "execution_mode": execution_mode,
        "authorization_scope": "synthetic_california_coast_single_chat",
        "browser_prompt_sha256": _text_sha256(
            json.loads((scenario_dir / "browser_context.json").read_text())["approved_prompt"]
        ),
        "session_id": browser_session_id,
        "client_message_id": browser["client_message_id"],
        "copilot_api_request_sha256": copilot_io["request_body_sha256"],
        "copilot_api_response_sha256": copilot_io["response_body_sha256"],
        "draft_api_request_sha256": draft_io["request_body_sha256"],
        "draft_api_response_sha256": draft_io["response_body_sha256"],
        "preview_api_request_sha256": preview_io["request_body_sha256"],
        "preview_api_response_sha256": preview_io["response_body_sha256"],
        "provider": "openai",
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "provider_transport_attempts": len(records) if live_provider else 0,
        "provider_calls_current_run": len(records) if live_provider else 0,
        "provider_request_body_sha256": (records[-1]["request_body_sha256"] if live_provider else None),
        "provider_response_body_sha256": (records[-1]["response_body_sha256"] if live_provider else None),
        "provider_request_authorization": (request_authorization if live_provider else None),
        "interpretation_sha256": _sha256(scenario_dir / "typed_interpretation.json"),
        "conversation_turn_id": browser["conversation_turn_id"],
        "draft_operation_id": browser["draft_operation"]["operation_id"],
        "draft_content_hash": proposal["draft_content_hash"],
        "preview_request_id": proposal["compiled_request"]["source_request_id"],
        "runtime_execution_id": proposal["runtime_execution"]["execution_id"],
        "runtime_execution_sha256": _sha256(scenario_dir / "runtime_execution.json"),
        "runtime_request_content_hash": proposal["runtime_execution"]["request_content_hash"],
        "executed_method": proposal["runtime_execution"]["planner_run"]["method_executed"],
        "evaluation_executed": proposal["runtime_execution"]["evaluation_executed"],
        "parent_plan_id": browser["parent_plan_id"],
        "child_plan_id": browser["child_plan_id"],
        "child_plan_content_hash": browser["child_plan_content_hash"],
        "diff_id": browser["diff_id"],
        "certificate_id": browser["certificate_id"],
        "geography_hash": browser["rendered_geometry_hash"],
        "rendered_map_plan_id": browser["rendered_map_plan_id"],
        "rendered_geometry_hash": browser["rendered_geometry_hash"],
        **usage,
        "billed_cost_usd": _cost(usage) if live_provider else "0",
        "accept_or_keep_invoked": browser["accept_or_keep_invoked"],
        "assertions_passed": browser["assertions_passed"],
    }
    if historical_provider_evidence is not None:
        trace["historical_provider_evidence"] = historical_provider_evidence
    return trace


def _file_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in {MANIFEST_NAME, REPORT_NAME}:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "exact_content_location": relative,
            }
        )
    return rows


def _source_rows() -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "size_bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha256(ROOT / relative),
        }
        for relative in SOURCE_INPUTS
    ]


def build_manifest(
    output_dir: Path,
    *,
    command: list[str],
    pytest_exit_code: int,
    health: dict[str, Any],
    trace: dict[str, Any],
    base_url: str,
    source_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sources = source_inputs if source_inputs is not None else _source_rows()
    execution_mode = trace["execution_mode"]
    live_provider = execution_mode == "live_openai"
    manifest = {
        "schema_version": "e2e1-live-copilot-file-io-manifest-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_lifecycle": {
            "artifact": "run_lifecycle.json",
            "record": json.loads((output_dir / "run_lifecycle.json").read_text(encoding="utf-8")),
        },
        "authorization": {
            "confirmed": live_provider,
            "scope": "synthetic_california_coast_single_chat",
            "execution_mode": execution_mode,
            "current_run_provider_calls": trace["provider_calls_current_run"],
            "prompt_sha256": trace["browser_prompt_sha256"],
            "model": MODEL,
            "reasoning_effort": EFFORT,
            "logical_message_cap": 1,
            "transport_attempt_cap": 1 if live_provider else 0,
            "accept_or_keep_allowed": False,
            "exact_provider_request": trace["provider_request_authorization"],
            "historical_provider_evidence": trace.get("historical_provider_evidence"),
        },
        "base_url": base_url,
        "command": command,
        "pytest_exit_code": pytest_exit_code,
        "assertions_passed": pytest_exit_code == 0 and trace["assertions_passed"],
        "health": {
            "build_id": health.get("build_id"),
            "map": health.get("components", {}).get("map", {}).get("code"),
            "routing": health.get("components", {}).get("routing", {}).get("code"),
            "openai": health.get("components", {}).get("openai", {}).get("code"),
        },
        "logical_copilot_messages": 1,
        "provider_transport_attempts": trace["provider_transport_attempts"],
        "usage": {
            "input_tokens": trace["input_tokens"],
            "cached_input_tokens": trace["cached_input_tokens"],
            "uncached_input_tokens": trace["uncached_input_tokens"],
            "output_tokens": trace["output_tokens"],
            "total_tokens": trace["total_tokens"],
        },
        "pricing": {
            "currency": "USD",
            "input_usd_per_million_tokens": str(INPUT_USD_PER_MILLION),
            "cached_input_usd_per_million_tokens": str(CACHED_INPUT_USD_PER_MILLION),
            "output_usd_per_million_tokens": str(OUTPUT_USD_PER_MILLION),
            "billed_cost_usd": trace["billed_cost_usd"],
        },
        "files": _file_rows(output_dir),
        "source_inputs": sources,
        "trace": trace,
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    return manifest


def render_report(output_dir: Path, manifest: dict[str, Any]) -> str:
    scenario_dir = output_dir / SCENARIO
    execution_mode = manifest["authorization"]["execution_mode"]
    live_provider = execution_mode == "live_openai"
    sections = [
        ("Exact browser context and input", "browser_context.json"),
        ("Exact product Copilot API input and output", "product_copilot_http_io.json"),
        (
            "Exact OpenAI LLM input and output"
            if live_provider
            else "Historical exact OpenAI LLM input and output replayed locally",
            "provider_exact_llm_io.json",
        ),
        ("Exact staged-draft API input and output", "product_draft_http_io.json"),
        ("Exact preview API input and output", "product_preview_http_io.json"),
        ("Exact runtime optimizer input, child, and independent evaluation", "runtime_execution.json"),
        ("Exact session reload response", "product_reload_http_io.json"),
        ("Exact cross-layer trace", "trace.json"),
    ]
    lines = [
        "# E2E-1L Exact Website, File, and LLM Input/Output Report",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Assertions passed: {str(manifest['assertions_passed']).lower()}",
        f"- Execution mode: {execution_mode}",
        f"- Model: {MODEL}",
        f"- Reasoning effort: {EFFORT}",
        f"- Logical browser messages: {manifest['logical_copilot_messages']}",
        f"- Current-run provider calls: {manifest['provider_transport_attempts']}",
        "- Runtime binding: run_lifecycle.json, pytest_junit.xml, and each actual browser environment receipt",
        f"- Input tokens: {manifest['usage']['input_tokens']}",
        f"- Cached input tokens: {manifest['usage']['cached_input_tokens']}",
        f"- Uncached input tokens: {manifest['usage']['uncached_input_tokens']}",
        f"- Output tokens: {manifest['usage']['output_tokens']}",
        f"- Total tokens: {manifest['usage']['total_tokens']}",
        f"- Current-run estimated API cost: USD {manifest['pricing']['billed_cost_usd']}",
        "- Cost basis: observed token usage multiplied by configured token rates; not invoice-confirmed. Legacy billed_cost_usd fields contain this estimate, not a billing receipt.",
        f"- Manifest content hash: {manifest['content_hash']}",
        "",
        (
            "Only the approved synthetic California Coast context was sent during "
            "this live run. Exact bodies are preserved below."
            if live_provider
            else "No provider request was sent during this replay. The OpenAI body "
            "below is historical evidence replayed locally; its lineage and hashes "
            "are retained and its historical usage and cost are labeled separately."
        ),
        "Headers, API keys, cookies, and session mutation tokens were not persisted. Accept and Keep were not invoked.",
        "",
        "## Exact file input and output",
        "",
        "| Path | Bytes | SHA-256 | Exact content |",
        "| --- | ---: | --- | --- |",
    ]
    for row in manifest["files"]:
        lines.append(
            f"| {row['path']} | {row['size_bytes']} | {row['sha256']} | "
            f"[{row['exact_content_location']}]({row['exact_content_location']}) |"
        )
    lines += ["", "## Exact source inputs", "", "| Path | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    for row in manifest["source_inputs"]:
        lines.append(f"| {row['path']} | {row['size_bytes']} | {row['sha256']} |")
    historical = manifest["authorization"].get("historical_provider_evidence")
    if historical is not None:
        lines += [
            "",
            "## Historical provider evidence used for replay",
            "",
            f"- Provider calls in historical evidence: {historical['provider_transport_attempts']}",
            f"- Historical input tokens: {historical['usage']['input_tokens']}",
            f"- Historical output tokens: {historical['usage']['output_tokens']}",
            f"- Historical total tokens: {historical['usage']['total_tokens']}",
            f"- Historical estimated API cost: USD {historical['billed_cost_usd']}",
            f"- Historical provider document content hash: {historical['provider_io_content_hash']}",
            f"- Historical provider file SHA-256: {historical['provider_io_file_sha256']}",
        ]
    for title, name in sections:
        exact = (scenario_dir / name).read_text(encoding="utf-8").rstrip()
        lines += ["", f"## {title}", "", "~~~json", exact, "~~~"]
    return "\n".join(lines).rstrip() + "\n"


def _preserve_failed_run(
    root: Path,
    *,
    exit_code: int,
    command: list[str],
    health: dict[str, Any],
    source_inputs: list[dict[str, Any]],
    execution_mode: str,
    current_records: list[dict[str, Any]],
) -> None:
    """Preserve observed partial I/O without manufacturing a passing trace."""
    scenario = root / SCENARIO
    observed = {}
    for name in ("browser_failure.json", "browser_assertions.json", "runtime_execution.json", "map_render_ack.json"):
        path = scenario / name
        if path.is_file():
            observed[name] = json.loads(path.read_text(encoding="utf-8"))
    usage = None
    # A timeout/cancellation record confirms neither provider completion nor
    # billing. Only a complete response resolves that uncertainty; a request
    # rejected before authorization never reaches this boundary.
    completion_unknown = (scenario / "provider_request_authorized.json").is_file() and not any(
        record.get("response_status_code") is not None and record.get("transport_error_code") is None
        for record in current_records
    )
    if not current_records and not completion_unknown:
        usage = dict(ZERO_USAGE)
    elif all(record.get("response_status_code") == 200 for record in current_records):
        try:
            usage = _usage(current_records)
        except (LiveE2E1Error, ValueError, TypeError, KeyError):
            # Missing/incomplete provider usage is unknown, never an invented zero.
            pass
    trace = {
        "schema_version": "e2e1-live-failure-trace-v1",
        "assertions_passed": False,
        "execution_mode": execution_mode,
        "pytest_exit_code": exit_code,
        "generated_at": datetime.now(UTC).isoformat(),
        "current_provider_transport_records": len(current_records),
        "current_provider_successful_responses": sum(
            record.get("response_status_code") == 200 for record in current_records
        ),
        "current_usage": usage,
        "current_billed_cost_usd": ("0" if not any(usage.values()) else _cost(usage)) if usage is not None else None,
        "provider_completion_unknown": completion_unknown,
        "observed_browser_artifacts": observed,
        "missing_success_artifacts": [
            name
            for name in ("browser_assertions.json", "runtime_execution.json", "map_render_ack.json")
            if not (scenario / name).is_file()
        ],
    }
    _write_json(root / "failure_trace.json", trace)
    manifest = {
        "schema_version": "e2e1-live-failure-file-io-manifest-v1",
        "assertions_passed": False,
        "command": command,
        "health": health,
        "source_inputs": source_inputs,
        "files": _file_rows(root),
        "trace": trace,
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    _write_json(root / MANIFEST_NAME, manifest)
    lines = [
        "# Failed real website integration run",
        "",
        "Assertions passed: false. This run is not acceptance evidence.",
        "",
        "Cost fields are estimates from observed usage and configured rates, not invoice-confirmed billing; null means unknown.",
        "",
        f"Execution mode: {execution_mode}",
        f"Pytest exit code: {exit_code}",
        "",
        "## Exact file inputs and outputs",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    lines += [
        f"| [{row['path']}]({row['path']}) | {row['size_bytes']} | {row['sha256']} |" for row in manifest["files"]
    ]
    lines += ["", "## Source inputs", "", "| File | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| {row['path']} | {row['size_bytes']} | {row['sha256']} |" for row in source_inputs]
    for name in (
        "provider_exact_llm_io.json",
        "product_copilot_http_io.json",
        "product_draft_http_io.json",
        "product_preview_http_io.json",
        "browser_failure.json",
    ):
        path = scenario / name
        if path.is_file():
            lines += ["", f"## Exact {name}", "", "~~~json", path.read_text(encoding="utf-8").rstrip(), "~~~"]
    lines += ["", "## Observed failure trace", "", "~~~json", json.dumps(trace, indent=2, sort_keys=True), "~~~", ""]
    (root / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


def _begin_lifecycle(root: Path, execution_mode: str, base_url: str) -> dict[str, Any]:
    record = {
        "schema_version": "e2e1-live-run-lifecycle-v1",
        "run_id": root.name,
        "execution_mode": execution_mode,
        "base_url": base_url,
        "process_id": os.getpid(),
        "observed_at": datetime.now(UTC).isoformat(),
        "server_ready_at": None,
        "pytest_started_at": None,
        "pytest_completed_at": None,
        "server_stopped_at": None,
        "finalized_at": None,
        "finalized": False,
        "pytest_exit_code": None,
        "command": [],
    }
    _write_json(root / "run_lifecycle.json", record)
    return record


@contextmanager
def _failure_evidence(
    root: Path,
    *,
    source_inputs: list[dict[str, Any]],
    execution_mode: str,
    current_records: list[dict[str, Any]],
    health: dict[str, Any],
):
    """Keep partial transport evidence for every runner failure, not only pytest red."""
    try:
        yield
    except BaseException as error:
        try:
            if execution_mode == "live_openai":
                provider = {
                    "schema_version": "e2e1-live-provider-exact-io-v1",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "scope": "synthetic_california_coast_single_chat",
                    "model": MODEL,
                    "reasoning_effort": EFFORT,
                    "request_headers_persisted": False,
                    "response_headers_persisted": False,
                    "authorization_persisted": False,
                    "records": current_records,
                }
                provider["content_hash"] = _canonical_sha256(provider)
                _write_json(root / SCENARIO / "provider_exact_llm_io.json", provider)
            _write_json(
                root / "runner_failure.json",
                {
                    "schema_version": "e2e1-live-runner-failure-v1",
                    "assertions_passed": False,
                    "error_type": type(error).__name__,
                    "failure_code": str(error)
                    if re.fullmatch(r"e2e1_[a-z0-9_]+", str(error))
                    else "e2e1_runner_exception",
                    "observed_at": datetime.now(UTC).isoformat(),
                },
            )
            lifecycle_path = root / "run_lifecycle.json"
            lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8")) if lifecycle_path.is_file() else {}
            _preserve_failed_run(
                root,
                exit_code=lifecycle.get("pytest_exit_code") or 1,
                command=lifecycle.get("command", []),
                health=health,
                source_inputs=source_inputs,
                execution_mode=execution_mode,
                current_records=current_records,
            )
        except Exception as preservation_error:
            # A secondary disk/serialization error must not replace the actual cause.
            error.add_note(f"Failure evidence preservation also failed: {type(preservation_error).__name__}")
        raise


def _finish_lifecycle(root: Path, record: dict[str, Any], thread: threading.Thread) -> None:
    record["finalized"] = not thread.is_alive()
    if record["finalized"]:
        record["server_stopped_at"] = datetime.now(UTC).isoformat()
        record["finalized_at"] = datetime.now(UTC).isoformat()
    _write_json(root / "run_lifecycle.json", record)


def run_live(
    *, output_dir: Path | None, confirmed: bool, port: int | None = None, prompt: str = APPROVED_PROMPT
) -> int:
    if not confirmed:
        raise LiveE2E1Error("e2e1_live_billed_authorization_required")
    if prompt not in (APPROVED_PROMPT, COMBINED_PROMPT):
        raise LiveE2E1Error("e2e1_live_selected_prompt_invalid")
    selected_port = port if port is not None else 8127
    if not 1 <= selected_port <= 65_535:
        raise LiveE2E1Error("e2e1_live_port_invalid")
    if not _port_available(selected_port):
        raise LiveE2E1Error(f"e2e1_live_port_{selected_port}_occupied")
    base_url = f"http://{HOST}:{selected_port}"
    run_id = datetime.now(UTC).strftime("e2e1_live_copilot_%Y%m%dT%H%M%S%fZ")
    root = (output_dir or ROOT / "results" / "e2e" / run_id).resolve()
    if root.exists() and any(root.iterdir()):
        raise LiveE2E1Error("e2e1_live_output_directory_not_empty")
    scenario_dir = root / SCENARIO
    scenario_dir.mkdir(parents=True, exist_ok=True)
    source_inputs: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    with _failure_evidence(
        root, source_inputs=source_inputs, execution_mode="live_openai", current_records=current_records, health=health
    ):
        source_inputs.extend(_source_rows())

        load_local_environment(ROOT)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LiveE2E1Error("e2e1_live_openai_not_configured")
        os.environ["PRODUCT_COPILOT_ADAPTER"] = "openai"
        os.environ["OPENAI_COPILOT_MODEL"] = MODEL
        os.environ["PRODUCT_APP_ORIGIN"] = base_url

        transport = ExactIOTransport(
            request_validator=_one_provider_attempt_validator(
                scenario_dir / "provider_request_authorized.json", expected_prompt=prompt
            )
        )
        transport.records = current_records
        transport.begin_case(effort=EFFORT, case_id="e2e1_live_website_chat")
        http_client = httpx.AsyncClient(transport=transport, timeout=60)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            max_retries=0,
            timeout=60,
            http_client=http_client,
        )
        adapter = OpenAICopilotAdapter(
            model=MODEL,
            api_key="injected-client",
            timeout_seconds=60,
            history_messages=0,
            history_characters=0,
            reasoning_effort=EFFORT,
            client=client,
        )

        exit_code = 1
        command: list[str] = []
        lifecycle = _begin_lifecycle(root, "live_openai", base_url)
        with tempfile.TemporaryDirectory(prefix="e2e1-live-") as temporary:
            temp_root = Path(temporary)
            app = create_product_app(
                repository_root=ROOT,
                state_root=temp_root / "state",
                application_port=selected_port,
                copilot=adapter,
            )
            app.router.add_event_handler("shutdown", http_client.aclose)
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=HOST,
                    port=selected_port,
                    log_level="warning",
                    access_log=False,
                )
            )
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                health.update(_wait_ready(thread, base_url))
                lifecycle["server_ready_at"] = datetime.now(UTC).isoformat()
                environment = os.environ.copy()
                environment["PRODUCT_E2E_BASE_URL"] = base_url
                environment["PRODUCT_E2E_RUN_ID"] = root.name
                environment["PRODUCT_E2E_LIVE_COPILOT_EVIDENCE_DIR"] = str(scenario_dir)
                environment["PRODUCT_E2E_LIVE_COPILOT_PROMPT"] = prompt
                environment.setdefault("PRODUCT_E2E_VIEWPORT_CHECKS", "1")
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    BROWSER_TEST,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(temp_root / "pytest"),
                    "--junitxml",
                    str(root / "pytest_junit.xml"),
                ]
                lifecycle["command"] = command
                lifecycle["pytest_started_at"] = datetime.now(UTC).isoformat()
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
                )
                exit_code = completed.returncode
                lifecycle["pytest_completed_at"] = datetime.now(UTC).isoformat()
                lifecycle["pytest_exit_code"] = exit_code
                (root / "pytest_output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
                _write_json(
                    scenario_dir / "provider_exact_llm_io.json",
                    {
                        "schema_version": "e2e1-live-provider-exact-io-v1",
                        "generated_at": datetime.now(UTC).isoformat(),
                        "scope": "synthetic_california_coast_single_chat",
                        "model": MODEL,
                        "reasoning_effort": EFFORT,
                        "request_headers_persisted": False,
                        "response_headers_persisted": False,
                        "authorization_persisted": False,
                        "records": transport.records,
                    },
                )
            finally:
                server.should_exit = True
                thread.join(timeout=15)
                _finish_lifecycle(root, lifecycle, thread)
                if thread.is_alive() and sys.exc_info()[0] is None:
                    raise LiveE2E1Error("e2e1_live_server_shutdown_failed")

        provider_io = {
            "schema_version": "e2e1-live-provider-exact-io-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "scope": "synthetic_california_coast_single_chat",
            "model": MODEL,
            "reasoning_effort": EFFORT,
            "request_headers_persisted": False,
            "response_headers_persisted": False,
            "authorization_persisted": False,
            "records": transport.records,
        }
        provider_io["content_hash"] = _canonical_sha256(provider_io)
        _write_json(scenario_dir / "provider_exact_llm_io.json", provider_io)
        browser = (
            json.loads((scenario_dir / "browser_assertions.json").read_text(encoding="utf-8")) if exit_code == 0 else {}
        )
        browser_failure = _browser_transport_gate(
            exit_code,
            transport.records,
            expected_session_id=browser.get("session_id"),
        )
        if browser_failure is not None:
            _preserve_failed_run(
                root,
                exit_code=exit_code,
                command=command,
                health=health,
                source_inputs=source_inputs,
                execution_mode="live_openai",
                current_records=transport.records,
            )
            print((root / "pytest_output.txt").read_text(encoding="utf-8"))
            return browser_failure
        request_authorization = _validate_transport(
            transport.records,
            expected_session_id=browser["session_id"],
        )
        usage = _usage(transport.records)
        trace = _build_trace(
            scenario_dir,
            provider_io,
            usage,
            request_authorization=request_authorization,
        )
        _write_json(
            scenario_dir / "usage_and_cost.json",
            {
                **usage,
                "input_usd_per_million_tokens": str(INPUT_USD_PER_MILLION),
                "cached_input_usd_per_million_tokens": str(CACHED_INPUT_USD_PER_MILLION),
                "output_usd_per_million_tokens": str(OUTPUT_USD_PER_MILLION),
                "billed_cost_usd": trace["billed_cost_usd"],
            },
        )
        _write_json(scenario_dir / "trace.json", trace)
        (root / "server_log_excerpt.txt").write_text(
            "disposable_loopback_server_started=true\n"
            "openai_component=openai_configured\n"
            "accept_or_keep_invoked=false\n"
            "disposable_loopback_server_stopped=true\n",
            encoding="utf-8",
        )
        manifest = build_manifest(
            root,
            command=command,
            pytest_exit_code=exit_code,
            health=health,
            trace=trace,
            base_url=base_url,
            source_inputs=source_inputs,
        )
        _write_json(root / MANIFEST_NAME, manifest)
        (root / REPORT_NAME).write_text(render_report(root, manifest), encoding="utf-8")

        from verify_e2e1_live_copilot_chat import verify

        verified = verify(root)
        print(json.dumps(verified, indent=2, sort_keys=True))
        return exit_code


def replay_recorded_provider_output(
    *,
    provider_io_path: Path,
    output_dir: Path,
    port: int | None = None,
) -> int:
    """Replay a captured strict OpenAI result through the browser with zero provider calls."""
    selected_port = port if port is not None else 8127
    if not 1 <= selected_port <= 65_535:
        raise LiveE2E1Error("e2e1_live_port_invalid")
    if not _port_available(selected_port):
        raise LiveE2E1Error(f"e2e1_live_port_{selected_port}_occupied")
    base_url = f"http://{HOST}:{selected_port}"
    root = output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise LiveE2E1Error("e2e1_live_output_directory_not_empty")
    scenario_dir = root / SCENARIO
    scenario_dir.mkdir(parents=True, exist_ok=True)
    source_inputs: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    with _failure_evidence(
        root,
        source_inputs=source_inputs,
        execution_mode="recorded_exact_output_replay",
        current_records=current_records,
        health=health,
    ):
        source_inputs.extend(_source_rows())
        provider_io_path = provider_io_path.resolve()
        provider_io_file_sha256 = _sha256(provider_io_path)
        provider_io = json.loads(provider_io_path.read_text(encoding="utf-8"))
        historical_request_authorization = _validate_transport(
            provider_io.get("records"),
            allow_historical_schema=True,
        )
        if provider_io.get("content_hash") != _canonical_sha256(
            {key: value for key, value in provider_io.items() if key != "content_hash"}
        ):
            raise LiveE2E1Error("e2e1_live_replay_provider_hash_invalid")
        _write_json(scenario_dir / "provider_exact_llm_io.json", provider_io)
        replayed_provider_io_file_sha256 = _sha256(scenario_dir / "provider_exact_llm_io.json")
        os.environ["PRODUCT_COPILOT_ADAPTER"] = "openai"
        os.environ["OPENAI_COPILOT_MODEL"] = MODEL
        os.environ["OPENAI_API_KEY"] = "recorded-output-replay"
        os.environ["PRODUCT_APP_ORIGIN"] = base_url
        adapter = RecordedCopilotAdapter(provider_io)
        exit_code = 1
        command = [
            sys.executable,
            "-m",
            "pytest",
            BROWSER_TEST,
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        lifecycle = _begin_lifecycle(root, "recorded_exact_output_replay", base_url)
        from e2e1_provider_isolation import ProviderIsolation

        with (
            ProviderIsolation(root / "provider_isolation.json", root.name) as guard,
            tempfile.TemporaryDirectory(prefix="e2e1-replay-") as temporary,
        ):
            temp_root = Path(temporary)
            app = create_product_app(
                repository_root=ROOT,
                state_root=temp_root / "state",
                application_port=selected_port,
                copilot=adapter,
            )
            runtime = app.state.product_runtime
            if (
                runtime.service.copilot is not adapter
                or type(adapter) is not RecordedCopilotAdapter
                or runtime.config.copilot_adapter != "openai"
                or os.environ.get("OPENAI_API_KEY") != "recorded-output-replay"
            ):
                raise LiveE2E1Error("e2e1_live_replay_adapter_mismatch")
            guard.record.update(
                effective_adapter="recorded_exact_output_replay",
                actual_adapter="RecordedCopilotAdapter",
                api_key_disabled=True,
                app_initialized=True,
            )
            guard._write()
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=HOST,
                    port=selected_port,
                    log_level="warning",
                    access_log=False,
                )
            )
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                health.update(_wait_ready(thread, base_url))
                lifecycle["server_ready_at"] = datetime.now(UTC).isoformat()
                environment = os.environ.copy()
                environment["PRODUCT_E2E_BASE_URL"] = base_url
                environment["PRODUCT_E2E_RUN_ID"] = output_dir.name
                environment["PRODUCT_E2E_LIVE_COPILOT_EVIDENCE_DIR"] = str(scenario_dir)
                environment["PRODUCT_E2E_LIVE_COPILOT_PROMPT"] = historical_request_authorization["user_prompt"]
                environment.setdefault("PRODUCT_E2E_VIEWPORT_CHECKS", "1")
                command += ["--basetemp", str(temp_root / "pytest"), "--junitxml", str(root / "pytest_junit.xml")]
                lifecycle["command"] = command
                lifecycle["pytest_started_at"] = datetime.now(UTC).isoformat()
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
                )
                exit_code = completed.returncode
                lifecycle["pytest_completed_at"] = datetime.now(UTC).isoformat()
                lifecycle["pytest_exit_code"] = exit_code
                (root / "pytest_output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            finally:
                guard.record["server_started"] = server.started
                server.should_exit = True
                thread.join(timeout=15)
                guard.record["server_stopped"] = not thread.is_alive()
                _finish_lifecycle(root, lifecycle, thread)
                if thread.is_alive() and sys.exc_info()[0] is None:
                    raise LiveE2E1Error("e2e1_live_server_shutdown_failed")
        if exit_code != 0:
            _preserve_failed_run(
                root,
                exit_code=exit_code,
                command=command,
                health=health,
                source_inputs=source_inputs,
                execution_mode="recorded_exact_output_replay",
                current_records=[],
            )
            print((root / "pytest_output.txt").read_text(encoding="utf-8"))
            return exit_code
        browser = json.loads((scenario_dir / "browser_assertions.json").read_text(encoding="utf-8"))
        if not isinstance(browser.get("session_id"), str):
            raise LiveE2E1Error("e2e1_live_browser_session_binding_invalid")
        historical_usage = _usage(provider_io["records"])
        historical_provider_evidence = {
            "schema_version": "e2e1-live-historical-provider-evidence-v1",
            "provider_transport_attempts": len(provider_io["records"]),
            "provider_io_source_path": str(provider_io_path),
            "provider_io_file_sha256": provider_io_file_sha256,
            "replayed_provider_io_file_sha256": replayed_provider_io_file_sha256,
            "provider_io_content_hash": provider_io["content_hash"],
            "provider_request_body_sha256": provider_io["records"][0]["request_body_sha256"],
            "provider_response_body_sha256": provider_io["records"][0]["response_body_sha256"],
            "request_authorization": historical_request_authorization,
            "usage": historical_usage,
            "billed_cost_usd": _cost(historical_usage),
        }
        trace = _build_trace(
            scenario_dir,
            provider_io,
            dict(ZERO_USAGE),
            request_authorization=historical_request_authorization,
            execution_mode="recorded_exact_output_replay",
            historical_provider_evidence=historical_provider_evidence,
        )
        _write_json(
            scenario_dir / "usage_and_cost.json",
            {
                "schema_version": "e2e1-live-replay-usage-and-cost-v1",
                "execution_mode": "recorded_exact_output_replay",
                "current_run_provider_calls": 0,
                "current_run_usage": dict(ZERO_USAGE),
                "current_run_billed_cost_usd": "0",
                "input_usd_per_million_tokens": str(INPUT_USD_PER_MILLION),
                "cached_input_usd_per_million_tokens": str(CACHED_INPUT_USD_PER_MILLION),
                "output_usd_per_million_tokens": str(OUTPUT_USD_PER_MILLION),
                "historical_provider_evidence": historical_provider_evidence,
            },
        )
        _write_json(scenario_dir / "trace.json", trace)
        (root / "server_log_excerpt.txt").write_text(
            "disposable_loopback_server_started=true\n"
            "recorded_exact_provider_output_replayed=true\n"
            "provider_calls_during_browser_replay=0\n"
            "accept_or_keep_invoked=false\n"
            "disposable_loopback_server_stopped=true\n",
            encoding="utf-8",
        )
        manifest = build_manifest(
            root,
            command=command,
            pytest_exit_code=exit_code,
            health=health,
            trace=trace,
            base_url=base_url,
            source_inputs=source_inputs,
        )
        _write_json(root / MANIFEST_NAME, manifest)
        (root / REPORT_NAME).write_text(render_report(root, manifest), encoding="utf-8")
        seed = dict(manifest)
        content_hash = seed.pop("content_hash")
        if (
            content_hash != _canonical_sha256(seed)
            or manifest["authorization"]["execution_mode"] != "recorded_exact_output_replay"
            or manifest["provider_transport_attempts"] != 0
            or manifest["usage"] != ZERO_USAGE
            or manifest["pricing"]["billed_cost_usd"] != "0"
            or trace["provider_calls_current_run"] != 0
            or trace["historical_provider_evidence"]["provider_io_content_hash"] != provider_io["content_hash"]
            or _sha256(scenario_dir / "provider_exact_llm_io.json") != replayed_provider_io_file_sha256
        ):
            raise LiveE2E1Error("e2e1_live_replay_manifest_invalid")
        from verify_e2e1_live_copilot_chat import verify

        verified = verify(root)
        print(
            json.dumps(
                {
                    "assertions_passed": True,
                    "execution_mode": "recorded_exact_output_replay",
                    "provider_calls_current_run": 0,
                    "current_run_billed_cost_usd": "0",
                    "historical_provider_io_content_hash": provider_io["content_hash"],
                    "output_dir": str(root),
                    "independent_verification": verified,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--confirm-billed-call",
        action="store_true",
        help="Authorize exactly one live OpenAI Responses API attempt.",
    )
    mode.add_argument(
        "--replay-provider-io",
        type=Path,
        help="Replay one captured exact provider document with zero provider calls.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=("replacement", "combined"),
        default="replacement",
        help="Select one exact synthetic website prompt, not arbitrary provider input.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Loopback app port; defaults to the map-provider-authorized port 8127.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replay_provider_io is not None:
        if args.profile != "replacement":
            raise LiveE2E1Error("e2e1_live_replay_profile_is_bound_to_record")
        run_id = datetime.now(UTC).strftime("e2e1_replay_copilot_%Y%m%dT%H%M%S%fZ")
        output_dir = args.output_dir or ROOT / "results" / "e2e" / run_id
        return replay_recorded_provider_output(
            provider_io_path=args.replay_provider_io,
            output_dir=output_dir,
            port=args.port,
        )
    return run_live(
        output_dir=args.output_dir,
        confirmed=args.confirm_billed_call,
        port=args.port,
        prompt=COMBINED_PROMPT if args.profile == "combined" else APPROVED_PROMPT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
