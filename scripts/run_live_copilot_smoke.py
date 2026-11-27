"""Run one provenance-bound Copilot API smoke message.

This script creates a disposable local session and makes one logical Copilot
message. It persists only typed metadata and explicit protected state snapshots;
raw message text, provider prose, session tokens, and provider errors are omitted.
Invoking the script can make a billed provider call when OpenAI mode is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_copilot_prompt import write_summary  # noqa: E402
from run_live_copilot_prompt_evaluation import (  # noqa: E402
    _canonical_sha256,
    build_source_snapshot,
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
_SMOKE_MESSAGE = "Review the safer repair"


def protected_state(session: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(session, dict) or any(field not in session for field in PROTECTED_STATE_FIELDS):
        raise ValueError("smoke_session_contract_invalid")
    return {field: session[field] for field in PROTECTED_STATE_FIELDS}


def _typed_turn(turn: dict[str, Any]) -> dict[str, Any]:
    interpretation = turn.get("interpretation")
    if not isinstance(interpretation, dict):
        raise ValueError("smoke_interpretation_missing")
    intents = interpretation.get("intents")
    highlights = interpretation.get("highlights")
    if not isinstance(intents, list) or not all(isinstance(intent, dict) for intent in intents):
        raise ValueError("smoke_intents_invalid")
    if not isinstance(highlights, dict):
        raise ValueError("smoke_highlights_invalid")
    return {
        "state": turn.get("state"),
        "provider": turn.get("provider"),
        "model": turn.get("model"),
        "prompt_version": turn.get("prompt_version"),
        "prompt_sha256": turn.get("prompt_sha256"),
        "interpretation_state": interpretation.get("state"),
        "typed_intents": intents,
        "permission": interpretation.get("permission"),
        "highlights": highlights,
        "error_code": turn.get("error_code"),
    }


def build_smoke_evidence(
    *,
    started_at: str,
    completed_at: str,
    health: dict[str, Any],
    session_id: str,
    client_message_id: str,
    before_session: dict[str, Any],
    after_session: dict[str, Any],
    turn: dict[str, Any],
) -> dict[str, Any]:
    before = protected_state(before_session)
    after = protected_state(after_session)
    evidence = {
        "schema_version": "copilot-live-smoke-evidence-v2",
        "started_at": started_at,
        "completed_at": completed_at,
        "logical_messages": 1,
        "provider_attempts": None,
        "provider_attempts_note": "The product API does not expose adapter retry metadata; the fixed-24 runner records attempts per case.",
        "health_status": health.get("status"),
        "health_core_ready": health.get("core_ready"),
        "protected_state_scope": list(PROTECTED_STATE_FIELDS),
        "protected_state_before": before,
        "protected_state_after": after,
        "protected_state_before_sha256": _canonical_sha256(before),
        "protected_state_after_sha256": _canonical_sha256(after),
        "protected_state_unchanged": before == after,
        "session_id_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
        "client_message_id_sha256": hashlib.sha256(client_message_id.encode()).hexdigest(),
        "message_fixture": "review_safer_repair_v1",
        "message_sha256": hashlib.sha256(_SMOKE_MESSAGE.encode()).hexdigest(),
        "turn": _typed_turn(turn),
        "source_snapshot": build_source_snapshot(),
        "raw_request_content_persisted": False,
        "raw_provider_content_persisted": False,
    }
    evidence["content_hash"] = _canonical_sha256(evidence)
    return evidence


def _request_json(
    *,
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Accept": "application/json", "Origin": base_url}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-Session-Token"] = token
    request = Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=45) as response:  # noqa: S310 - loopback URL is validated below
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"smoke_http_{exc.code}") from None
    except URLError:
        raise RuntimeError("smoke_transport_failed") from None
    if not isinstance(value, dict):
        raise RuntimeError("smoke_response_invalid")
    return value


def run_smoke(base_url: str) -> dict[str, Any]:
    if base_url not in {"http://127.0.0.1:8127", "http://localhost:8127"}:
        raise ValueError("smoke_loopback_url_required")
    started_at = datetime.now(UTC).isoformat()
    health = _request_json(base_url=base_url, method="GET", path="/api/health")
    created = _request_json(base_url=base_url, method="POST", path="/api/sessions", payload={})
    session = created.get("session")
    token = created.get("mutation_token")
    if not isinstance(session, dict) or not isinstance(token, str):
        raise RuntimeError("smoke_session_create_invalid")
    session_id = session.get("session_id")
    if not isinstance(session_id, str):
        raise RuntimeError("smoke_session_id_invalid")
    before = _request_json(base_url=base_url, method="GET", path=f"/api/sessions/{session_id}", token=token).get(
        "session"
    )
    if not isinstance(before, dict):
        raise RuntimeError("smoke_before_state_invalid")
    client_message_id = f"client_message_{uuid4().hex}"
    response = _request_json(
        base_url=base_url,
        method="POST",
        path=f"/api/sessions/{session_id}/copilot/messages",
        token=token,
        payload={
            "expected_revision": before["revision"],
            "client_message_id": client_message_id,
            "message": _SMOKE_MESSAGE,
        },
    )
    after = _request_json(base_url=base_url, method="GET", path=f"/api/sessions/{session_id}", token=token).get(
        "session"
    )
    turn = response.get("turn")
    if not isinstance(after, dict) or not isinstance(turn, dict):
        raise RuntimeError("smoke_result_invalid")
    return build_smoke_evidence(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        health=health,
        session_id=session_id,
        client_message_id=client_message_id,
        before_session=before,
        after_session=after,
        turn=turn,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8127")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = run_smoke(args.base_url.rstrip("/"))
    write_summary(args.output, evidence)
    print(
        json.dumps(
            {
                "protected_state_unchanged": evidence["protected_state_unchanged"],
                "status": "written",
            }
        )
    )
    return 0 if evidence["protected_state_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
